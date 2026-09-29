"""Measured pilot acoustics with explicit heuristic limits; no emotion inference."""
from __future__ import annotations
import argparse
import json
import re
import wave

from cineduplex.contracts import digest, write_json
from .nas_source import nas_root


def extract(root):
    root = nas_root(root)
    print('prosody: importing NAS numpy', flush=True)
    import numpy as np
    out = root/'output/pilot10-current'
    scenes = [json.loads(s) for s in (out/'scenes.jsonl').read_text().splitlines()]
    rows = []
    for scene in scenes:
        for turn in scene['turns']:
            source = root/turn['audio']['path']
            if digest(source) != turn['audio']['sha256']:
                raise ValueError('source audio changed')
            with wave.open(str(source)) as stream:
                if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
                    raise ValueError('expected 16k mono PCM16')
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float64)/32768
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError('empty/nonfinite source waveform')
            # Non-overlapping 20ms energy windows; the final short window is real.
            counts = np.array([len(samples[i:i+320]) for i in range(0,len(samples),320)])
            rms = np.array([np.sqrt(np.mean(samples[i:i+320]**2)) for i in range(0,len(samples),320)])
            db = 20*np.log10(np.maximum(rms,1e-10))
            threshold = max(-60.0, float(np.percentile(db,95))-35.0)
            quiet = db < threshold
            spans = []
            first = None
            for idx, value in enumerate([*quiet, False]):
                if value and first is None:
                    first = idx
                elif not value and first is not None:
                    start = first*320
                    end = min(idx*320,len(samples))
                    if end-start >= 1600:
                        spans.append(dict(start_sample=start,end_sample=end,duration_seconds=(end-start)/16000))
                    first = None
            # 40ms normalized autocorrelation, 10ms hop. A periodicity estimate,
            # not calibrated voicing confidence or validated emotional pitch.
            frequencies = []
            strengths = []
            for start in range(0, max(0,len(samples)-640+1), 160):
                x = samples[start:start+640]
                if 20*np.log10(max(float(np.sqrt(np.mean(x*x))),1e-10)) < threshold:
                    continue
                x = x-x.mean()
                corr = np.correlate(x,x,mode='full')[len(x)-1:]
                if corr[0] <= 1e-12:
                    continue
                lags = np.arange(32,267)  # 60–500 Hz search range at 16k.
                energies = np.cumsum(x*x)
                left = energies[len(x)-lags-1]
                right = energies[-1]-energies[lags-1]
                norm = corr[lags]/np.sqrt(np.maximum(left*right,1e-20))
                local = [j for j in range(1,len(norm)-1)
                         if norm[j]>=norm[j-1] and norm[j]>norm[j+1] and norm[j]>=0.65]
                if not local:
                    continue
                best = max(float(norm[j]) for j in local)
                j = next(j for j in local if norm[j]>=max(0.65,0.9*best))
                frequencies.append(float(16000/lags[j]))
                strengths.append(float(norm[j]))
            text = re.sub(r'<[^>]*>|\[[^\]]*\]|\([^)]*\)', '', turn['text'])
            units = len(re.findall(r'[\u3400-\u9fff]|[A-Za-z0-9]+',text))
            duration = len(samples)/16000
            row = dict(scene_id=scene['scene_id'],utterance_id=turn['utterance_id'],
                audio_path=turn['audio']['path'],input_sha256=turn['audio']['sha256'],
                label_source='measured_audio_and_explicit_acoustic_heuristics',
                confidence=None,confidence_status='NOT_CALIBRATED',available_at='utterance_end',
                use_scope='offline characterization; no causal online control or emotion approval',
                duration_seconds=duration,sample_rate=16000,
                rms=float(np.sqrt(np.mean(samples*samples))),
                mean_square_energy=float(np.mean(samples*samples)),
                peak_amplitude=float(np.max(np.abs(samples))),
                clipped_sample_fraction=float(np.mean(np.abs(samples)>=32767/32768)),
                rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(samples*samples))),1e-10))),
                frame_rms_dbfs_p05_p50_p95=[float(v) for v in np.percentile(db,[5,50,95])],
                dynamic_range_db_p95_minus_p05=float(np.percentile(db,95)-np.percentile(db,5)),
                low_energy_threshold_dbfs=threshold,
                low_energy_sample_ratio=float(counts[quiet].sum()/len(samples)),
                low_energy_spans_ge_100ms=spans,
                low_energy_note='Energy threshold only; not verified silence, speech segmentation or breaths.',
                pitch_hz_p10_p50_p90=[float(v) for v in np.percentile(frequencies,[10,50,90])] if frequencies else None,
                pitch_periodic_frames=len(frequencies),
                pitch_mean_periodicity=float(np.mean(strengths)) if strengths else None,
                pitch_note='Normalized autocorrelation heuristic, 40ms/10ms, 60–500Hz; octave errors and unvoiced/noisy failures remain unreviewed.',
                transcript_units=units,transcript_units_per_second=units/duration,
                rate_note='CJK characters plus Latin/alphanumeric words per full clip second; not syllables per speech-active second.',
                original_emotion=turn.get('original_emotion',turn.get('emotion')),
                source_captions=turn.get('source_captions'),
                performance_inference=dict(tension=None,suppression=None,vulnerability=None,
                    confidence=None,status='NOT_INFERRED_NO_VALIDATED_JUDGE'),
                auditory_review='NOT_PERFORMED',training_ready=False)
            rows.append(row)
            print('prosody:',turn['utterance_id'],'measured',flush=True)
    if len(rows)!=28 or len({r['utterance_id'] for r in rows})!=28:
        raise ValueError('expected exactly 28 unique current pilot utterances')
    path=out/'prosody.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    if [json.loads(x) for x in path.read_text().splitlines()] != rows:
        raise ValueError('saved feature readback mismatch')
    write_json(out/'prosody-readback.json',dict(status='PASSED',rows=len(rows),
        source_audio_hashes_verified=True,exact_saved_rows_readback=True,
        output_sha256=digest(path),heuristics_validated_against_human_labels=False,
        scope='actual measurements and source labels, not pseudo-performance or emotion validation'))
    print('prosody: 28 measurements saved and exactly read back',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    extract(parser.parse_args().root)
