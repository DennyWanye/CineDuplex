import unittest

from cineduplex.train.native_audio_context import ReceivedUtteranceWindows


class ReceivedWindowsTest(unittest.TestCase):
    def test_long_idle_does_not_fill_next_utterance(self):
        b = ReceivedUtteranceWindows(74)
        for i in range(100):
            b.append(i, 0)
        self.assertEqual(b.chunks, [98, 99])
        for i in range(100, 107):
            b.append(i, 1)
        self.assertEqual((b.start, b.end), (98, 107))
        b.refreshed(1)
        for i in range(107, 207):
            b.append(i, 1)
        for i in range(207, 212):
            b.append(i, 2)
        self.assertEqual(b.chunks, list(range(205, 212)))
        self.assertEqual((b.start, b.end), (205, 212))

    def test_pending_speech_keeps_short_pauses_until_refresh(self):
        b = ReceivedUtteranceWindows(74)
        for i in range(12):
            b.append(i, 1)
        self.assertEqual(b.chunks, list(range(12)))
        b.refreshed(1)
        self.assertEqual((b.start, b.end), (12, 12))

    def test_capacity_preserves_absolute_cache_cursor(self):
        b = ReceivedUtteranceWindows(74)
        for i in range(90):
            b.append(i, 1)
        self.assertEqual((b.start, b.end, len(b.chunks)), (16, 90, 74))


if __name__ == '__main__':
    unittest.main()
