"""Local CPU speech evidence for turn-taking; never discard microphone samples.

The pinned Silero ONNX asset stays on NAS. Its 16 kHz streaming contract is
512 samples, 64 context samples and recurrent state (2, 1, 128), as documented
by https://github.com/snakers4/silero-vad. Model/license provenance is retained
with the deployment assets. No audio is sent to a speech detection service.
"""
import hashlib

MODEL_SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'


class SpeechEvidence:
    """Hysteresis with a new utterance ID, independent of loudness and SL score."""
    # Eight 32ms windows require 256ms of strong speech evidence.
    # A 96ms transient must not create a reply or cancel active playback.
    START_FRAMES = 8

    def __init__(self):
        self.active = False
        self.utterance = 0
        self.high_frames = 0
        self.quiet_frames = 0
        self.probability = 0.

    def observe(self, probability):
        self.probability = float(probability)
        started = False
        if probability >= .5:
            self.high_frames += 1
            self.quiet_frames = 0
            if not self.active and self.high_frames >= self.START_FRAMES:
                self.active = True
                self.utterance += 1
                started = True
        elif probability < .35:
            self.high_frames = 0
            self.quiet_frames += 1
            if self.quiet_frames >= 10:
                self.active = False
        else:
            self.high_frames = 0
            self.quiet_frames = 0
        return started

    def can_reply(self, answered_utterance):
        return (self.utterance > answered_utterance and not self.active
                and self.quiet_frames >= 20)

    def reply_overdue(self, answered_utterance):
        # A native KL can persist indefinitely after a real question. Admit
        # SS only after 1.28 s of confirmed silence, never on an empty call.
        return self.can_reply(answered_utterance) and self.quiet_frames >= 40

    def can_interrupt(self, response_utterance):
        return self.active and self.utterance > response_utterance


def load_detector(path):
    import onnxruntime as ort
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != MODEL_SHA256:
        raise ValueError('speech detector model identity')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(source, sess_options=options,
                               providers=['CPUExecutionProvider'])


class StreamingSpeech:
    def __init__(self, detector):
        import numpy as np
        self.detector = detector
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.evidence = SpeechEvidence()

    def feed(self, audio):
        import numpy as np
        self.pending = np.concatenate((self.pending, np.asarray(audio, dtype=np.float32)))
        started = False
        count = 0
        for offset in range(0, len(self.pending) - 511, 512):
            frame = self.pending[offset:offset + 512].reshape(1, 512)
            values = np.concatenate((self.context, frame), axis=1)
            probability, self.state = self.detector.run(None, {
                'input': values, 'state': self.state, 'sr': np.array(16000, dtype=np.int64)})
            self.context = frame[:, -64:].copy()
            started |= self.evidence.observe(float(probability.reshape(-1)[0]))
            count += 512
        self.pending = self.pending[count:].copy()
        return started
