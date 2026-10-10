#!/usr/bin/env python3
"""Align bounded rank telemetry from one real call; never infer listening quality."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--call', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve earlier evidence')
    ranks, sources = [], []
    for rank in range(4):
        path = args.root / 'rank-profile' / args.call / f'rank{rank}.jsonl'
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('rank metadata budget')
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if any(row['rank'] != rank for row in rows):
            raise ValueError('rank identity differs')
        indexed = {row['input_seq']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate input window')
        ranks.append(indexed)
        sources.append({'rank': rank, 'bytes': len(raw), 'rows': len(rows),
                        'sha256': hashlib.sha256(raw).hexdigest()})
    common = sorted(set.intersection(*(set(rank) for rank in ranks)))
    if not common:
        raise ValueError('no shared input windows')
    def compact(row):
        return {key: row.get(key) for key in
                ('rank', 'input_seq', 'window_wall_ms', 'setup_ms',
                 'command_dispatch_ms', 'slow_cpu_calls', 'stages',
                 'allocator_delta', 'major_faults', 'cpu_system_ms',
                 'thread_schedule_delta_ns',
                 'thread_schedule_delta_runtime_ns_wait_ns_count')}
    slow = sorted(common, key=lambda seq: max(rank[seq]['window_wall_ms']
                                              for rank in ranks), reverse=True)[:8]
    report = {'call': args.call, 'sources': sources,
              'common_windows': len(common),
              'same_window_ids': all(set(rank) == set(ranks[0]) for rank in ranks),
              'allocator_retries': [sum(row['allocator_delta']['num_alloc_retries']
                                        for row in rank.values()) for rank in ranks],
              'rank_mean_wall_ms': [statistics.mean(row['window_wall_ms']
                                                    for row in rank.values()) for rank in ranks],
              'slow_windows': [{'input_seq': seq,
                                'ranks': [compact(rank[seq]) for rank in ranks],
                                # Durations use one host clock at a time. An
                                # unequal gap supports an arrival/setup skew,
                                # but does not identify a network root cause.
                                'gap_from_previous_finish_ms': [
                                    (rank[seq]['at'] - rank[seq - 1]['at']) * 1000
                                    - rank[seq]['window_wall_ms']
                                    if seq - 1 in rank else None for rank in ranks]}
                               for seq in slow],
              'caveats': ['nested CUDA stages overlap and include waits',
                          'unsampled collective GPU time is unknown',
                          'peer dispatch time includes waiting for next input',
                          'no subjective audio acceptance']}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'slow_windows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
