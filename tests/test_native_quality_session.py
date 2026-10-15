import unittest
import sys
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch
from cineduplex.train.native_quality_session import wrap_factory


class NativeQualityLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.original=lambda value: value
        self.fw=NS(init_speaking_processor=self.original,model=NS())
        def output(fw):
            fw.init_speaking_processor=lambda value: value+1
            fw._cineduplex_output_contract=True
            return {}
        self.output=NS(install=output)
        self.context=NS(install=lambda session,detector:{})

    def test_sequential_calls_and_original_factory_restoration(self):
        closed=[]
        create=wrap_factory(lambda fw:NS(close=lambda:closed.append(True)),self.context,self.output,None)
        for _ in range(2):
            session=create(self.fw)
            with self.assertRaises(RuntimeError):create(self.fw)
            session.close()
            self.assertIs(self.fw.init_speaking_processor,self.original)
            self.assertFalse(self.fw._cineduplex_output_contract)
        self.assertEqual(len(closed),2)

    def test_context_failure_closes_native_session_and_restores(self):
        closed=[]
        def fail(*args):raise ValueError('context mismatch')
        create=wrap_factory(lambda fw:NS(close=lambda:closed.append(True)),NS(install=fail),self.output,None)
        with self.assertRaisesRegex(ValueError,'context mismatch'):create(self.fw)
        self.assertEqual(closed,[True]);self.assertIs(self.fw.init_speaking_processor,self.original)

    def test_close_failure_still_releases_framework(self):
        def fail():raise ValueError('close failed')
        session=wrap_factory(lambda fw:NS(close=fail),self.context,self.output,None)(self.fw)
        with self.assertRaisesRegex(ValueError,'close failed'):session.close()
        self.assertIsNone(self.fw._cineduplex_quality_session)
        self.assertIs(self.fw.init_speaking_processor,self.original)

    def test_merge_conditioning_is_owned_by_one_call(self):
        self.fw.model._cineduplex_merge_conditioning=NS(shape=(1,123))
        session=wrap_factory(lambda fw:NS(close=lambda:None),self.context,self.output,None)(self.fw)
        self.assertIsNone(self.fw.model._cineduplex_merge_conditioning)
        self.fw.model._cineduplex_merge_conditioning=NS(shape=(1,30))
        session.close()
        self.assertIsNone(self.fw.model._cineduplex_merge_conditioning)

    def test_failed_close_restores_experimental_rms_forward(self):
        original=lambda x:x
        layer=NS(forward=lambda x:x+1,_cineduplex_quality_previous_rms=original)
        self.fw.model.modules=lambda:[layer]
        def fail():raise ValueError('native close failed')
        session=wrap_factory(lambda fw:NS(close=fail),self.context,self.output,None)(self.fw)
        with self.assertRaisesRegex(ValueError,'native close failed'):session.close()
        self.assertIs(layer.forward,original)
        self.assertFalse(hasattr(layer,'_cineduplex_quality_previous_rms'))

    def test_graph_close_failure_still_restores_native_framework_and_rms(self):
        original=lambda x:x
        layer=NS(forward=lambda x:x+1,_cineduplex_quality_previous_rms=original)
        self.fw.model.modules=lambda:[layer]
        def fail(model):raise ValueError('graph teardown failed')
        self.fw.model._cineduplex_native_graph_close=fail
        session=wrap_factory(lambda fw:NS(close=lambda:None),self.context,self.output,None)(self.fw)
        with self.assertRaisesRegex(ValueError,'graph teardown failed'):session.close()
        self.assertIs(layer.forward,original)
        self.assertIsNone(self.fw._cineduplex_quality_session)
        self.assertIs(self.fw.init_speaking_processor,self.original)

    def test_reference_prefix_is_scoped_to_session_creation(self):
        native=NS(SYSTEM_MESSAGE_PREFIX='existing product instruction')
        runtime=ModuleType('lychee_fd.runtime');runtime.vllm_generation=native
        observed=[]
        def create(fw):
            observed.append(native.SYSTEM_MESSAGE_PREFIX)
            return NS(close=lambda:None)
        with patch.dict(sys.modules, {'lychee_fd.runtime':runtime}):
            session=wrap_factory(create,self.context,self.output,None,
                                 upstream_reference_prefix=True)(self.fw)
        self.assertEqual(observed,['<|BOT|>system\nYou are a helpful assistant.<|EOT|>'])
        self.assertEqual(native.SYSTEM_MESSAGE_PREFIX,'existing product instruction')
        session.close()

    def test_reference_prefix_creation_failure_restores_product_instruction(self):
        native=NS(SYSTEM_MESSAGE_PREFIX='existing product instruction')
        runtime=ModuleType('lychee_fd.runtime');runtime.vllm_generation=native
        def fail(fw):raise ValueError('native creation failed')
        with patch.dict(sys.modules, {'lychee_fd.runtime':runtime}):
            with self.assertRaisesRegex(ValueError,'native creation failed'):
                wrap_factory(fail,self.context,self.output,None,
                             upstream_reference_prefix=True)(self.fw)
        self.assertEqual(native.SYSTEM_MESSAGE_PREFIX,'existing product instruction')
        self.assertIsNone(self.fw._cineduplex_quality_session)
        self.assertIs(self.fw.init_speaking_processor,self.original)

if __name__=='__main__':unittest.main()
