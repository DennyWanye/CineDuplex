#!/usr/bin/env python3
"""Inspect configuration, or explicitly probe with a fixed harmless sentence."""
import argparse
import json
import uuid

from cineduplex.guardrail.aliyun_guardrail import Config, AliyunGuardrail, GuardrailError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Make one paid text-only cloud request')
    args = parser.parse_args()
    try:
        config = Config.from_env()
        result = {'configuration': config.status(), 'live_request': False}
        if args.live:
            client = AliyunGuardrail(config)
            decision = client.check_text('你好，请介绍一下春天的天气。', direction='output',
                                         session_id=str(uuid.uuid4()), chat_id=str(uuid.uuid4()))
            result.update(live_request=True, decision=decision.record())
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except GuardrailError as exc:
        print(json.dumps({'error': str(exc), 'cloud_verified': False}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
