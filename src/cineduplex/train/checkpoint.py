"""Integrity verifier for committed local checkpoint manifests; not a trainer."""
from pathlib import Path
from cineduplex.contracts import safe_child, read_json, digest, canonical_hash, Blocked


def verify(root):
    manifest = read_json(safe_child(root, "manifest.json"))
    complete = read_json(safe_child(root, "COMPLETE.json"))
    if manifest.get("schema") != "cineduplex.checkpoint.v1" or manifest.get("kind") != "training_resume":
        raise Blocked("not a training resume checkpoint")
    if complete.get("manifest_sha256") != canonical_hash(manifest):
        raise Blocked("completion is not bound to this manifest")
    world = manifest.get("world_size")
    if type(world) is not int or world < 1 or len(manifest.get("ranks", [])) != world:
        raise Blocked("invalid world size or missing rank receipt")
    if {r["rank"] for r in manifest["ranks"]} != set(range(world)):
        raise Blocked("rank receipts are not complete")
    files = manifest.get("files", [])
    if not files or len({f["path"] for f in files}) != len(files):
        raise Blocked("empty or duplicate checkpoint file list")
    for f in files:
        path = safe_child(root, f["path"])
        if not path.is_file() or path.stat().st_size != f["size"] or digest(path) != f["sha256"]:
            raise Blocked(f"missing or corrupt checkpoint file: {f['path']}")
    return {"integrity": "VERIFIED", "manifest_sha256": canonical_hash(manifest),
            "world_size": world, "files": len(files), "resume_numerics": "NOT_TESTED"}
