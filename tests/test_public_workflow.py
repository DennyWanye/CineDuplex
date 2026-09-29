import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cineduplex.data import workflow
from cineduplex.data.nas_source import nas_root
from cineduplex.contracts import safe_child


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def test_asset_identity_rejects_corruption_and_wrong_git_blob(self):
        p=self.root/'asset';p.write_bytes(b'abc')
        spec=dict(size=3,sha256=hashlib.sha256(b'abc').hexdigest(),
                  git_blob_sha1=hashlib.sha1(b'blob 3\0abc').hexdigest())
        workflow.verify_asset(p,spec)
        p.write_bytes(b'abd')
        with self.assertRaises(ValueError):workflow.verify_asset(p,spec)
        p.write_bytes(b'abc')
        with self.assertRaises(ValueError):workflow.verify_asset(p,{**spec,'git_blob_sha1':'0'*40})

    def test_existing_verified_asset_does_not_download(self):
        p=self.root/'asset';p.write_bytes(b'abc')
        spec=dict(destination='asset',size=3,sha256=hashlib.sha256(b'abc').hexdigest())
        with patch.object(workflow,'Source',side_effect=AssertionError('must not download')):
            self.assertEqual(workflow.acquire(self.root,spec),p)

    def test_bad_existing_asset_is_not_overwritten(self):
        p=self.root/'asset';p.write_bytes(b'bad')
        with patch.object(workflow,'Source',side_effect=AssertionError('must not download')):
            with self.assertRaises(ValueError):
                workflow.acquire(self.root,dict(destination='asset',size=3,sha256='0'*64))
        self.assertEqual(p.read_bytes(),b'bad')

    def test_no_local_mount_fallback(self):
        with patch('subprocess.check_output',return_value='/dev/disk1 on / (apfs)\n'):
            with self.assertRaisesRegex(RuntimeError,'mount absent'):nas_root(self.root)

    def test_refuse_path_escape_and_symlink(self):
        with self.assertRaises(ValueError):safe_child(self.root,'../escape')
        (self.root/'link').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):safe_child(self.root,'link/asset')

    def test_existing_review_and_revision_are_preserved(self):
        for name,fn in [('pilot10-current',workflow.consolidate),('pilot10-revision2',workflow.revised)]:
            p=self.root/'output'/name/'scenes.jsonl';p.parent.mkdir(parents=True);p.write_text('keep review')
            with patch.object(workflow,'nas_root',return_value=self.root):
                with self.assertRaises(FileExistsError):fn(self.root)
            self.assertEqual(p.read_text(),'keep review')

    def test_status_dispatch_does_not_prepare_data(self):
        with patch.object(workflow,'status') as status,patch.object(workflow,'prepare_inputs') as prepare:
            self.assertEqual(workflow.main(['status','--root',str(self.root)]),0)
            status.assert_called_once_with(str(self.root));prepare.assert_not_called()

    def test_manifest_and_vendor_fixed_sources(self):
        spec=json.loads((workflow.PROJECT/'configs/data_sources.json').read_text())
        destinations=[x['destination'] for x in spec['assets']]
        self.assertEqual(len(destinations),len(set(destinations)))
        for asset in spec['assets']:
            self.assertEqual(len(asset['revision']),40)
            self.assertGreater(asset['size'],0)
            self.assertTrue('sha256' in asset or 'git_blob_sha1' in asset)
            self.assertFalse(Path(asset['destination']).is_absolute())
        manifest=json.loads((workflow.PROJECT/'configs/vendor-source-manifest.json').read_text())
        for name,expected in manifest['files'].items():
            actual=hashlib.sha256((workflow.PROJECT/'vendor/lychee-fd-source'/name).read_bytes()).hexdigest()
            self.assertEqual(actual,expected,name)


if __name__=='__main__':unittest.main()
