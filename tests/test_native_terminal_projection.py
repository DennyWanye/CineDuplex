import unittest
from types import SimpleNamespace as NS

from cineduplex.train.native_terminal_projection import observe_output, terminal_observed


class DistributedTerminalTests(unittest.TestCase):
    def test_leader_processor_state_does_not_change_collective_branch(self):
        ranks = [NS(processor=NS(eos_token_id=9, has_eos=(rank == 3))) for rank in range(4)]
        self.assertEqual([terminal_observed(p) for p in ranks], [False] * 4)
        for p in ranks:
            observe_output(p, {'text_token': 42})
        self.assertEqual([terminal_observed(p) for p in ranks], [False] * 4)
        for p in ranks:
            observe_output(p, {'text_token': 9})
        self.assertEqual([terminal_observed(p) for p in ranks], [True] * 4)
        self.assertTrue(all(p.processor.has_eos for p in ranks))

    def test_terminal_persists_across_windows_but_not_new_answer(self):
        p = NS(processor=NS(eos_token_id=9, has_eos=False))
        observe_output(p, {'text_token': 9})
        observe_output(p, {'text_token': 3})
        self.assertTrue(terminal_observed(p))
        self.assertFalse(terminal_observed(NS(processor=p.processor)))
        self.assertFalse(terminal_observed(None))

    def test_missing_or_noninteger_broadcast_token_fails_before_next_step(self):
        p = NS(processor=NS(eos_token_id=9, has_eos=False))
        for value in (None, True, '9', 9.0):
            with self.assertRaises(ValueError):
                observe_output(p, {'text_token': value})
        self.assertFalse(terminal_observed(p))


if __name__ == '__main__':
    unittest.main()
