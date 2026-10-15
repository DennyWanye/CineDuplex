"""Native listening samples survive cache rebuilds without a Mac Torch import."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np


class Tensor(np.ndarray):
    def new_full(self, shape, value):
        return np.full(shape, value, dtype=self.dtype).view(Tensor)


def tensor(values):
    return np.array(values, dtype=np.int64).view(Tensor)


class MergeConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch=NS(cat=lambda rows,axis:np.concatenate(rows,axis=axis).view(Tensor),
                 where=lambda *args:np.where(*args).view(Tensor))
        spec=importlib.util.spec_from_file_location('merge_contract',
            Path(__file__).parents[1]/'src/cineduplex/train/generation_stream.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'torch':torch}):spec.loader.exec_module(module)
        cls.condition=staticmethod(module.merge_conditioning)

    def test_discarded_listening_text_is_replayed_instead_of_visible_pad(self):
        model=NS(_cineduplex_merge_conditioning=tensor([[-1,-1,777,-1]]))
        ids=tensor([[10,11,0,0,30]])
        actual=self.condition(model,ids,tensor([[88]]),0,5)
        self.assertEqual(actual.tolist(),[[11,0,777,30,88]])
        self.assertEqual(model._cineduplex_merge_conditioning.tolist(),actual.tolist())
        self.assertEqual(ids.tolist(),[[10,11,0,0,30]])

    def test_new_steps_append_to_preserved_native_conditioning(self):
        model=NS(_cineduplex_merge_conditioning=tensor([[10,11,777]]))
        actual=self.condition(model,tensor([[1,2,0,0]]),tensor([[99]]),3,1)
        self.assertEqual(actual.tolist(),[[99]])
        self.assertEqual(model._cineduplex_merge_conditioning.tolist(),[[10,11,777,99]])

    def test_first_prefill_uses_native_shift_and_sampled_next_text(self):
        model=NS()
        actual=self.condition(model,tensor([[10,11,12]]),tensor([[13]]),0,3)
        self.assertEqual(actual.tolist(),[[11,12,13]])


if __name__=='__main__':unittest.main()
