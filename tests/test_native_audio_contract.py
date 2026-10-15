import unittest

from cineduplex.train.native_audio_contract import native_audio_contract, require_native_audio_contract


class NativeAudioContractTest(unittest.TestCase):
    def test_published_clock_and_delay_are_preserved(self):
        contract=native_audio_contract({'control_token_chunk_size':10,'stoken_delay_num':10})
        self.assertEqual(contract['window_samples'],6400)
        self.assertEqual(contract['window_second'],.4)
        self.assertEqual(contract['stoken_delay_num'],10)

    def test_historical_24_second_features_are_rejected(self):
        config={'control_token_chunk_size':10,'stoken_delay_num':10}
        old=native_audio_contract(config);old['window_second']=24
        with self.assertRaisesRegex(ValueError,'regenerate'):
            require_native_audio_contract({'native_audio_contract':old},config)
        with self.assertRaises(ValueError):
            require_native_audio_contract({'rows':[]},config)

    def test_current_contract_accepts_without_changing_source(self):
        config={'control_token_chunk_size':10,'stoken_delay_num':10}
        current={'native_audio_contract':native_audio_contract(config),'rows':['keep']}
        self.assertEqual(require_native_audio_contract(current,config)['window_second'],.4)
        self.assertEqual(current['rows'],['keep'])


if __name__=='__main__':unittest.main()
