"""Full-duplex replay must distinguish safe cancellation from lost replies."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('recorded_replay',
    Path(__file__).parents[1]/'scripts/replay_native_recorded_call.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def interrupted():
    return [
        dict(type='event_start', event_kind='response', event_id='old', epoch=1, seq=1),
        dict(type='ingress_speech', input_seq=5, revision=2, seq=2),
        dict(type='stop_audio', reason='ingress_confirmed_speech', epoch=2,
             input_seq=5, ingress_revision=2, seq=3),
        dict(type='event_start', event_kind='response', event_id='latest', epoch=2, seq=4),
        dict(type='event_end', event_id='latest', seq=5),
    ]


class ResponseAccountingTests(unittest.TestCase):
    def test_confirmed_barge_in_and_latest_complete_are_accounted(self):
        r = replay.response_outcomes(interrupted())
        self.assertEqual(r['accounted'], 2)
        self.assertTrue(r['latest_reply_complete'])
        self.assertEqual(r['stale_release_sequences'], [])

    def test_cancellation_without_real_ingress_is_missing_response(self):
        r = replay.response_outcomes([e for e in interrupted() if e['type'] != 'ingress_speech'])
        self.assertEqual(r['accounted'], 1)
        self.assertEqual(r['confirmed_user_cancelled'], {})

    def test_late_audio_after_cancel_is_not_accepted(self):
        events = interrupted()+[dict(type='audio', event_id='old', seq=6)]
        self.assertEqual(replay.response_outcomes(events)['stale_release_sequences'], [6])

    def test_missing_latest_reply_is_not_hidden_by_old_cancellation(self):
        r = replay.response_outcomes(interrupted()[:-1])
        self.assertFalse(r['latest_reply_complete'])
        self.assertEqual(r['accounted'], 1)

    def test_duplicate_end_does_not_count_as_another_response(self):
        events = interrupted()+[dict(type='event_end', event_id='latest', seq=6)]
        self.assertEqual(replay.response_outcomes(events)['accounted'], 2)


if __name__ == '__main__':
    unittest.main()
