"""Listening PAD must not alter speaking, audio or control inputs."""
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from cineduplex.train.native_turn_policy import install_listening_alignment
from cineduplex.train.native_listening_contract import install as install_contract


class ListeningAlignmentTests(unittest.TestCase):
    def test_only_listening_channels_use_native_pad_processors(self):
        self.check_contract(install_listening_alignment)

    def test_pinned_quality_contract_preserves_audio_control_and_speaking(self):
        self.check_contract(install_contract)

    def check_contract(self, install):
        fw = NS(text_pad_token_id=17, stoken_pad_token_id=23)
        factory = NS(init_listening_pad_processor=lambda fw: (
            ('fixed', fw.text_pad_token_id), ('fixed', fw.stoken_pad_token_id)))
        module = NS(SingleTurnGenerationFramework=factory)
        seen = []
        session = NS(fw=fw, listening_state='l',
                     _stream_model=lambda **kwargs: seen.append(kwargs) or iter(()))
        with patch.dict(sys.modules, {'lychee_fd.runtime.vllm_generation': module}):
            install(session)
        audio, control, cache, speaking_text, speaking_speech = [object() for _ in range(5)]
        args = dict(audio_input_ids=audio, control_logits_processor=control,
                    past_key_values=cache, logits_processor=speaking_text,
                    stoken_logits_processor=speaking_speech, max_new_tokens=1)
        list(session._stream_model(**args))
        self.assertEqual(seen[-1]['logits_processor'], ('fixed', 17))
        self.assertEqual(seen[-1]['stoken_logits_processor'], ('fixed', 23))
        for key in ('audio_input_ids', 'control_logits_processor', 'past_key_values'):
            self.assertIs(seen[-1][key], args[key])
        for state in ('s', 'b'):
            session.listening_state = state
            list(session._stream_model(**args))
            self.assertEqual(seen[-1], args)
        self.assertIs(args['logits_processor'], speaking_text)


if __name__ == '__main__':
    unittest.main()
