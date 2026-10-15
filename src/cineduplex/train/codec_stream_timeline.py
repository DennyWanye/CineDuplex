"""Keep flow excitation on the audio timeline when its attention cache shrinks.

The original flow derives its noise-table offset from attention-cache length.
Token2wav caps that cache, so later chunks reuse the same excitation window.
This adapter preserves the original weights, noise table, solver and sampler;
only the position used to read that immutable table follows emitted frames.
"""
import types


def install(codec):
    if getattr(codec, '_absolute_noise_cursor_installed', False):
        return
    decoder = codec.flow.decoder
    original_chunk = decoder.forward_chunk
    original_reset = codec.reset
    original_pcm = codec.pcm

    def forward_chunk(current, mu, spks, cond, n_timesteps=10,
                      temperature=1., cnn_cache=None, att_cache=None):
        cursor = getattr(current, '_absolute_noise_cursor', None)
        if cursor is None:
            return original_chunk(mu, spks, cond, n_timesteps=n_timesteps,
                                  temperature=temperature, cnn_cache=cnn_cache,
                                  att_cache=att_cache)
        import torch
        frames = int(mu.size(2))
        offset = cursor['offset']
        if type(offset) is not int or offset < 0 or frames <= 0:
            raise ValueError('invalid flow audio position')
        if offset + frames > current.rand_noise.shape[-1]:
            raise ValueError('flow noise timeline capacity exceeded')
        z = current.rand_noise[:, :, offset:offset+frames] * temperature
        t_span = torch.linspace(0, 1, n_timesteps+1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span*.5*torch.pi)
        result = current.solve_euler_chunk(x=z, t_span=t_span, mu=mu,
            spks=spks, cond=cond, cnn_cache=cnn_cache, att_cache=att_cache)
        cursor['calls'].append({'noise_offset': offset, 'frames': frames,
            'attention_cache_frames': int(att_cache.shape[4]) if att_cache is not None else 0})
        cursor['offset'] += frames
        return result

    def reset(current):
        original_reset()
        current._absolute_noise_frames = int(current.prompt[2].shape[1])
        current._absolute_noise_trace = []

    def pcm(current, codes, last=False):
        if not hasattr(current, '_absolute_noise_frames'):
            raise RuntimeError('reset flow timeline before decoding an utterance')
        if getattr(decoder, '_absolute_noise_cursor', None) is not None:
            raise RuntimeError('concurrent flow noise cursor')
        cursor = {'offset': current._absolute_noise_frames, 'calls': []}
        decoder._absolute_noise_cursor = cursor
        try:
            result = original_pcm(codes, last=last)
            if len(cursor['calls']) != 1:
                raise RuntimeError('streaming codec flow-call contract changed')
            current._absolute_noise_frames = cursor['offset']
            current._absolute_noise_trace.extend(cursor['calls'])
            return result
        finally:
            decoder._absolute_noise_cursor = None

    decoder.forward_chunk = types.MethodType(forward_chunk, decoder)
    codec.reset = types.MethodType(reset, codec)
    codec.pcm = types.MethodType(pcm, codec)
    codec._absolute_noise_cursor_installed = True
    codec._absolute_noise_original = (original_chunk, original_reset, original_pcm)


def uninstall(codec):
    if not getattr(codec, '_absolute_noise_cursor_installed', False):
        return
    decoder = codec.flow.decoder
    if getattr(decoder, '_absolute_noise_cursor', None) is not None:
        raise RuntimeError('cannot restore a codec during flow decoding')
    decoder.forward_chunk, codec.reset, codec.pcm = codec._absolute_noise_original
    del codec._absolute_noise_cursor_installed
    del codec._absolute_noise_original
