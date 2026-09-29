"""Validate supplied canonical samples, never fabricate training labels."""
from pathlib import Path
import hashlib
import json
import wave

from cineduplex.contracts import digest, safe_child


def speech_tokens(value, vocab_size):
    if not isinstance(value, list) or any(type(x) is not int or not 0 <= x < vocab_size for x in value):
        raise ValueError("speech tokens must be list[int] within the raw codec vocabulary")
    return value


def validate_record(record, root):
    if record.get("schema_version") != "cineduplex.scene.v1":
        raise ValueError("unknown dataset schema")
    for field in ("scene_id", "split_group", "rights_ref"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"missing {field}")
    if record.get("sample_rate") != 16000 or type(record.get("duration_samples")) is not int or record["duration_samples"] <= 0:
        raise ValueError("16kHz positive integer sample duration required")
    rights = json.loads(safe_child(root, record["rights_ref"]).read_text())
    if "training" not in rights.get("allowed_uses", []) or not rights.get("source_id") or not rights.get("license_id"):
        raise ValueError("missing declared training rights")
    for name in ("input_audio", "target_audio"):
        asset = record[name]
        path = safe_child(root, asset["path"])
        sha = asset["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError("actual lowercase SHA256 required")
        if digest(path) != sha:
            raise ValueError(f"audio hash mismatch: {name}")
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getframerate() != 16000 or audio.getnframes() != record["duration_samples"]:
                raise ValueError("require two mono PCM WAV tracks with equal sample timeline")
    if not isinstance(record.get("turns"), list) or not record["turns"]:
        raise ValueError("empty turns cannot be counted as valid training data")
    if record.get("annotation_status") != "reviewed":
        raise ValueError("annotations not reviewed")
    return {"scene_id": record["scene_id"], "split_group": record["split_group"],
            "audio_asset_integrity": "VERIFIED", "rights": "DECLARATION_PRESENT_NOT_LEGAL_ADJUDICATION",
            "training_ready": False, "remaining": ["turn schema and causal alignment", "raw codec tokens", "official collator"]}


def validate_jsonl(config):
    records, seen = [], set()
    path = safe_child(config["root"], config["manifest"])
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = validate_record(json.loads(line), config["root"])
        if value["scene_id"] in seen:
            raise ValueError(f"duplicate scene_id at line {lineno}")
        seen.add(value["scene_id"])
        records.append(value)
    if not records:
        raise ValueError("no samples provided")
    return {"records": records, "count": len(records), "training_ready": False,
            "scope": "audio/hash/rights declaration ingress only; causal/collator gates remain open"}
