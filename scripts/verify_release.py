"""Verify every distributed file against PUBLIC_FILES.json; no downloads or fits."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "PUBLIC_FILES.json").read_text())
    entries = manifest["files"]
    seen = set()
    for entry in entries:
        relative = PurePosixPath(entry["file"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Invalid or duplicate release path")
        seen.add(str(relative))
        path = root / str(relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or nonregular release file: " + str(relative))
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Changed release file: " + str(relative))
    return {"verified_files": len(entries), "new_fits": False, "data_downloads": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root)))
