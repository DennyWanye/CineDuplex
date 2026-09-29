# CineDuplex agent entry point

Read README.md, docs/AGENT_HANDOFF.zh-CN.md, docs/DATA_PIPELINE.zh-CN.md,
and docs/STATUS.md before operating this repository.

- This public release is a data-preparation MVP, not an implemented trainer.
- Local docs/EXECUTION_CHECKPOINT.md, when present, contains private session state.
  Latest user instructions override historical deployment/lease instructions.
- The original scheduled job is PAUSED at the user's request. Do not restart it,
  take a GPU lease, operate a production service, or begin training implicitly.
- No plan-test or subagents by default. Finish the main implementation before
  broad regression. Do only relevant checks; report actual evidence in Chinese.
- This implementation currently supports the actual macOS SMB mount
  /Volumes/media. All new downloads, dependency caches, temporary files, and
  data outputs must be under /Volumes/media/CineDuplex (or another explicit
  task directory on that same verified mount). Never fall back to local disk.
- Inspect existing processes and artifacts before launching work. Do not rerun
  completed production to refresh a report. Never force-unmount a busy share.
- Preserve source revisions, original annotations, hashes and license records.
  Do not publish source audio, transcripts, model weights or credentials here.
- Separate expressive utterance examples from continuous duplex conversations.
  Never call mixed or zero-filled audio independent speaker recordings.
- Do not use HumDial-FDBench test data for training. Suppression remains a
  candidate until actual listening review supports it. No invented confidence.
- Actual codec/loader checks do not certify hearing quality, CUDA parity,
  model training, original wall-clock timing, or clean isolated channels.
- Before public pushes, inspect the staged allowlist and run publication tests.
  Keep machine-specific configurations, private checkpoints and runtime evidence
  ignored. Vendor code retains its upstream license and notices.
