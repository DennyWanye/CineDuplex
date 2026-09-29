# Capability and verification status

Publication snapshot: 2026-09-29. This is a data-preparation MVP.

| Capability | Observed result in the original Mac/NAS environment | Limit |
|---|---|---|
| Candidate package | 10 scenes, 28 utterances, 2018 raw codes | Not a certified training set |
| Duration | 80.49225 seconds of utterances; 100.3363125 seconds of scene windows | Gaps included only in the latter |
| CPU codec | Actual encoding, reconstruction and WAV readback | No listening or CUDA parity approval |
| Frontend comparison | 29 inputs, 2077 codes identical to pinned S3Tokenizer 0.2.0 Torch CPU frontend | Includes one extra S09 span; not 29 independent scenes |
| Dataset/collator | 14 actual rows, covering 10 source scenes, saved/reloaded and collated | Import-adapted original algorithms on CPU, no full model |
| HF review export | 28 rows exactly saved and reloaded | Review schema, not a training export |
| Prosody | 28 source-backed records measured and read back | Heuristics not calibrated, performance judge not run |
| Training-ready | 0/10 | Hearing and structure/use acceptance pending |

The above describes private execution receipts on NAS; raw receipts and samples are not distributed in this public repository. It is not a fresh-clone end-to-end reproduction claim. Publication checks are listed in the agent handoff document.

## Remaining conditions

- All original/reconstructed utterances require actual listening review.
- S08 uses original neutral labels and caption clues; suppression is not confirmed.
- S01–S08 are expressive mixture excerpts, not continuous isolated duplex tracks.
- S09/S10 retain actual channels, but crosstalk/music/event quality and split assignment need review.
- The original loader rebuilds timing. S09's 16s window produced 9.2s user input; S10's 13s window produced 12s. Do not claim preserved wall-clock timing.
- No complete trainer, trained model, human-validated pseudo-performance labels, or inference deployment is included.
- Generic tools are narrowly tested. A new machine still needs dependency, source access, NAS and ffmpeg qualification.
- NAS intermittent latency occurred; successful small reads do not establish a root-cause fix.

There is no running production job in the original handoff. The scheduled task is paused by user request. Do not resume it automatically.
