import unittest
from cineduplex.data.codec_decode_cpu import neutral_reference


class ReferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.scene = dict(scene_id='S03', source_revision='fixed', coverage_requested='sadness', split_group='G2')
        self.turn = dict(speaker_id='08', utterance_id='target', text='target words',
                         emotion='sad', start_sample=0, end_sample=160000, overlaps=[])

    def pair(self, sid='S04', **updates):
        s = dict(self.scene, scene_id=sid, coverage_requested='neutral')
        t = dict(self.turn, utterance_id='reference', text='other words', emotion='neutral')
        t.update(updates)
        return s, t

    def test_same_speaker_different_scene_and_text(self):
        valid = self.pair()
        s, t = neutral_reference([valid, (self.scene, self.turn)], self.scene, self.turn)
        self.assertEqual((s['scene_id'], t['utterance_id']), ('S04', 'reference'))

    def test_no_self_or_bad_reference_fallback(self):
        for pair in [self.pair(sid='S03'), self.pair(text='target words'),
                     self.pair(speaker_id='07'), self.pair(overlaps=['other']),
                     self.pair(emotion='sad'), self.pair(end_sample=8000),
                     self.pair(original_emotion='sad')]:
            with self.subTest(pair=pair):
                with self.assertRaises(ValueError):
                    neutral_reference([pair, (self.scene, self.turn)], self.scene, self.turn)

    def test_suppression_caption_scene_cannot_supply_neutral_control(self):
        s,t=self.pair();s['coverage_requested']='suppression'
        with self.assertRaises(ValueError):
            neutral_reference([(s,t)],self.scene,self.turn)

    def test_different_group_does_not_prove_same_speaker(self):
        s,t=self.pair();s['split_group']='G3'
        with self.assertRaises(ValueError):
            neutral_reference([(s,t)],self.scene,self.turn)


if __name__ == '__main__':
    unittest.main()
