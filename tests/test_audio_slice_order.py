"""Focused ASR ordering contracts; no microphone, cloud, or GPU use."""
import unittest

from cineduplex.guardrail.aliyun_guardrail import (
    AliyunGuardrail, Config, GuardrailError, chronological_slices,
)
from cineduplex.guardrail.call_guardrail import audio_text_alignment, speech_progress_can_release, PendingResponse


class AudioSliceOrderTests(unittest.TestCase):
    def test_vendor_array_order_does_not_change_spoken_order(self):
        first = {'StartTime': 0, 'EndTime': 10, 'Text': '第一句'}
        second = {'StartTime': 10, 'EndTime': 20, 'Text': '第二句'}
        ordered, meta = chronological_slices([second, first])
        self.assertEqual(ordered, [first, second])
        self.assertTrue(dict(meta)['reordered'])

    def test_absolute_clock_is_not_mixed_with_relative_clock(self):
        with self.assertRaises(GuardrailError):
            chronological_slices([{'StartTime': 0, 'EndTime': 10},
                                  {'StartTimestamp': 100, 'EndTimestamp': 200}])
        with self.assertRaises(GuardrailError):
            chronological_slices([{'StartTime': True, 'EndTime': 10},
                                  {'StartTime': 20, 'EndTime': 10}])

    def test_reordering_never_discards_risk_or_duplicate_time_slices(self):
        client = AliyunGuardrail(Config(enabled=True, access_key_id='test', access_key_secret='test'))
        parts = [
            {'StartTime': 10, 'EndTime': 20, 'RiskLevel': 'none', 'Text': '后一句'},
            {'StartTime': 0, 'EndTime': 10, 'RiskLevel': 'none', 'Text': '前一句'},
            {'StartTime': 0, 'EndTime': 10, 'RiskLevel': 'none',
             'Text': '', 'Labels': 'sexual_sounds'},
        ]
        replies = iter([{'Data': {'TaskId': 'test'}},
                        {'Data': {'RiskLevel': 'none', 'SliceDetails': parts}}])
        client.rpc = lambda *a, **k: next(replies)
        client._upload_wav = lambda *a: {}
        decision = client.check_audio(b'\x01\x00', sample_rate=24000, data_id='test')
        self.assertEqual(decision.transcript, '前一句\n\n后一句')
        self.assertFalse(decision.allowed)
        self.assertIn('sexual_sounds', decision.labels)
        self.assertNotIn('前一句', str(decision.record()))
        self.assertEqual(decision.record()['audio_order']['slice_count'], 3)

    def test_alignment_exposes_short_prefix_without_exposing_content(self):
        comparison = audio_text_alignment('第一句。第二句。第三句。', '第一句', prefix=True)
        self.assertTrue(comparison['consistent'])
        self.assertEqual(comparison['expected_chars'], 9)
        self.assertEqual(comparison['compared_chars'], 3)
        self.assertFalse(audio_text_alignment('', '')['consistent'])
        self.assertNotIn('第一句', str(comparison))

    def test_long_pcm_without_word_progress_waits_but_final_pause_can_finish(self):
        pending=PendingResponse('p',0)
        pending.prefix_matched_chars=9;pending.prefix_progress_end=48000*5
        unchanged={'matched_chars':9}
        self.assertTrue(speech_progress_can_release(pending,unchanged,48000*6,final=False))
        self.assertFalse(speech_progress_can_release(pending,unchanged,48000*9,final=False))
        self.assertTrue(speech_progress_can_release(pending,{'matched_chars':15},48000*9,final=False))
        self.assertTrue(speech_progress_can_release(pending,unchanged,48000*9,final=True))

    def test_complete_speech_requires_more_than_a_matching_start(self):
        self.assertTrue(audio_text_alignment('第一句第二句第三句第四句','第一句第二句',prefix=True)['consistent'])
        self.assertFalse(audio_text_alignment('第一句第二句第三句第四句','第一句第二句')['consistent'])


if __name__ == '__main__':
    unittest.main()
