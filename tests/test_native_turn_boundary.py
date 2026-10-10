"""Turn completion evidence without loading a model or GPU libraries."""
import sys
import types
import unittest
from unittest.mock import patch

from cineduplex.train.native_turn_policy import TurnBoundary, install


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.boundary = TurnBoundary(900, 800)
        self.boundary.observe({'type': 'state_change', 'from': 'L', 'to': 'S'})

    def finish(self, **fields):
        return self.boundary.observe({'type': 'state_change', 'from': 'S', 'to': 'L',
                                      'reason': 'control_sl', **fields}, speech_tail=0)

    def test_eos_survives_padding_of_speech_tail(self):
        self.boundary.observe({'type': 'speaking_token', 'stoken': 900, 'text_token': 42})
        self.assertFalse(self.finish()['interrupt'])
        self.assertEqual(self.finish()['turn_boundary']['decision'], 'preserve_completed')

    def test_generation_end_survives_padding_without_a_token_event(self):
        self.boundary.observe({'type': 'speaking_done', 'reason': 'eos'})
        self.assertFalse(self.finish()['interrupt'])

    def test_text_eot_does_not_certify_audio_completion(self):
        self.boundary.observe({'type': 'speaking_token', 'stoken': 901, 'text_token': 800})
        result = self.finish()
        self.assertTrue(result['interrupt'])
        self.assertTrue(result['turn_boundary']['text_complete'])

    def test_explicit_native_interrupt_is_preserved(self):
        self.boundary.observe({'type': 'speaking_done', 'reason': 'eos'})
        self.assertTrue(self.finish(interrupt=True)['interrupt'])

    def test_new_response_cannot_reuse_previous_completion(self):
        self.boundary.observe({'type': 'speaking_done', 'reason': 'eos'})
        self.boundary.observe({'type': 'state_change', 'from': 'L', 'to': 'S'})
        self.assertTrue(self.finish()['interrupt'])

    def test_quiet_input_does_not_override_native_cancellation(self):
        self.boundary.input_window([0.] * 6400)
        result = self.finish()
        self.assertTrue(result['interrupt'])
        self.assertEqual(result['turn_boundary']['input_rms_dbfs'], -120.)

    def test_real_session_wrapper_remembers_eos_before_native_tail_padding(self):
        class Tensor:
            def __getitem__(self, index):
                return types.SimpleNamespace(item=lambda: 0)

        class Session:
            generated_stoken_ids = Tensor()
            def advance_stream(self, audio, **kwargs):
                yield {'type': 'state_change', 'from': 'L', 'to': 'S', 'reason': 'control_ss'}
                yield {'type': 'speaking_token', 'stoken': 900, 'text_token': 42}
                yield {'type': 'state_change', 'from': 'S', 'to': 'L', 'reason': 'control_sl', 'interrupt': False}

        class Framework:
            tts_end_id = 900
            eos_token_id = 800
            model = types.SimpleNamespace()
            def create_incremental_stream_session(self, **kwargs):
                return Session()

        class Processor:
            def __call__(self, *args):
                return None

        module = types.ModuleType('lychee_fd.runtime.hf_v9_realtime')
        module.HFRealtimeV9GenerationFramework = Framework
        module.SpeakingControlLogitsProcessor = Processor
        packages = {name: types.ModuleType(name) for name in ('lychee_fd', 'lychee_fd.runtime')}
        with patch.dict(sys.modules, {**packages, module.__name__: module}):
            install()
            session = Framework().create_incremental_stream_session()
            events = list(session.advance_stream([0.] * 6400))
        self.assertFalse(events[-1]['interrupt'])
        self.assertTrue(events[-1]['turn_boundary']['audio_complete'])


if __name__ == '__main__':
    unittest.main()
