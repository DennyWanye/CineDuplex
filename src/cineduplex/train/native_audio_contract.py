"""Native streaming audio timing shared by feature export and validation."""


def native_audio_contract(config):
    chunk = config['control_token_chunk_size']
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 2:
        raise ValueError('native control chunk must be an integer >= 2')
    # The aligned training input interleaves 80ms encoder features and padding
    # on the 40ms speech/control grid. An encoder window is one control chunk.
    return {'schema': 1, 'sample_rate': 16000, 'samples_per_step': 640,
            'control_token_chunk_size': chunk, 'window_samples': chunk * 640,
            'window_second': chunk * .04, 'align_audio_input': True,
            'stoken_delay_num': config['stoken_delay_num']}


def require_native_audio_contract(payload, config):
    expected = native_audio_contract(config)
    if payload.get('native_audio_contract') != expected:
        raise ValueError('feature audio timing differs from native runtime; '
                         'regenerate into a new feature directory, preserve old artifacts')
    return expected
