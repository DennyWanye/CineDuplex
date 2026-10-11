"""Turn decisions use confirmed speech, not a loudness threshold or raw SL."""
import unittest

from cineduplex.train.speech_activity import SpeechEvidence


class SpeechEvidenceTests(unittest.TestCase):
    def frames(self, evidence, probability, count):
        return [evidence.observe(probability) for _ in range(count)]

    def test_noise_spikes_cannot_start_or_interrupt_a_turn(self):
        e = SpeechEvidence()
        for _ in range(20):
            self.frames(e, .9, 2)
            self.frames(e, .05, 4)
        self.assertEqual(e.utterance, 0)
        self.assertFalse(e.can_reply(0))
        self.assertFalse(e.can_interrupt(0))

    def test_question_must_finish_before_reply_and_is_answered_once(self):
        e = SpeechEvidence()
        self.frames(e, .95, 30)
        self.assertFalse(e.can_reply(0))
        self.frames(e, .05, 9)
        self.assertFalse(e.can_reply(0))
        e.observe(.05)
        self.assertFalse(e.can_reply(0))
        self.frames(e, .05, 10)
        self.assertTrue(e.can_reply(0))
        self.assertFalse(e.can_reply(1))

    def test_new_speech_interrupts_but_original_question_does_not(self):
        e = SpeechEvidence()
        self.frames(e, .9, 20)
        self.assertFalse(e.can_interrupt(1))
        self.frames(e, .05, 10)
        self.frames(e, .9, 3)
        self.assertTrue(e.can_interrupt(1))
        self.assertFalse(e.can_reply(1))

    def test_short_pause_does_not_split_a_question_into_an_interrupt(self):
        e = SpeechEvidence()
        self.frames(e, .9, 10)
        self.frames(e, .1, 6)
        self.frames(e, .9, 10)
        self.assertEqual(e.utterance, 1)
        self.assertFalse(e.can_interrupt(1))

    def test_reply_deadline_requires_unanswered_speech_and_resets_on_new_speech(self):
        e = SpeechEvidence()
        self.frames(e, .01, 100)
        self.assertFalse(e.reply_overdue(0))
        self.frames(e, .9, 20)
        self.frames(e, .01, 39)
        self.assertFalse(e.reply_overdue(0))
        e.observe(.01)
        self.assertTrue(e.reply_overdue(0))
        self.assertFalse(e.reply_overdue(1))
        self.frames(e, .9, 3)
        self.assertFalse(e.reply_overdue(0))


if __name__ == '__main__':
    unittest.main()
