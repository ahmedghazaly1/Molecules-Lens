"""Uploaded GENTRL folder checkpoints and their selected roles."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

STORE = Path(__file__).resolve().parents[2] / "models"
MANIFEST = STORE / "active.json"
ROLES = {"base", "pretrained", "after_rl"}
REQUIRED = {"enc.model", "dec.model", "lp.model", "order.pkl"}
MAX_FILE = 100 * 1024 * 1024
MAX_TOTAL = 300 * 1024 * 1024
_lock = threading.Lock()


def _manifest() -> dict:
    if not MANIFEST.is_file():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def list_checkpoints() -> dict:
    with _lock:
        return {role: item["label"] for role, item in _manifest().items()
                if role in ROLES and (STORE / item["id"]).is_dir()}


def selected(role: str) -> Path | None:
    with _lock:
        item = _manifest().get(role)
    if not item:
        return None
    path = STORE / item["id"]
    return path if path.is_dir() and all((path / name).is_file() for name in REQUIRED) else None


def clear(role: str) -> None:
    if role not in ROLES:
        raise ValueError("Unknown checkpoint role")
    with _lock:
        entries = _manifest()
        entries.pop(role, None)
        STORE.mkdir(parents=True, exist_ok=True)
        temporary = MANIFEST.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries), encoding="utf-8")
        temporary.replace(MANIFEST)


def _activate(role: str, folder: Path, label: str) -> dict:
    if role not in ROLES:
        raise ValueError("Unknown checkpoint role")
    missing = REQUIRED - {p.name for p in folder.iterdir() if p.is_file() and p.stat().st_size > 0}
    if missing:
        raise ValueError("Checkpoint is missing: " + ", ".join(sorted(missing)))
    if sum((folder / name).stat().st_size for name in REQUIRED) > MAX_TOTAL:
        raise ValueError("Checkpoint exceeds the 300 MB limit")
    with _lock:
        entries = _manifest()
        entries[role] = {"id": folder.name, "label": label[:120]}
        STORE.mkdir(parents=True, exist_ok=True)
        temporary = MANIFEST.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries), encoding="utf-8")
        temporary.replace(MANIFEST)
    return {"role": role, "label": label[:120]}


def upload_zip(role: str, source: Path, label: str) -> dict:
    if role not in ROLES:
        raise ValueError("Unknown checkpoint role")
    target = STORE / uuid.uuid4().hex
    target.mkdir(parents=True)
    try:
        with zipfile.ZipFile(source) as archive:
            matches = {name: [] for name in REQUIRED}
            for entry in archive.infolist():
                if not entry.is_dir() and Path(entry.filename).name in REQUIRED:
                    matches[Path(entry.filename).name].append(entry)
            if any(len(found) != 1 for found in matches.values()):
                raise ValueError("ZIP must contain exactly one copy of each checkpoint file")
            parents = {str(Path(found[0].filename).parent) for found in matches.values()}
            if len(parents) != 1:
                raise ValueError("The four checkpoint files must be in one folder")
            if sum(found[0].file_size for found in matches.values()) > MAX_TOTAL:
                raise ValueError("Checkpoint exceeds the 300 MB limit")
            for name, found in matches.items():
                if found[0].file_size <= 0 or found[0].file_size > MAX_FILE:
                    raise ValueError(f"Invalid size for {name}")
                with archive.open(found[0]) as reader, (target / name).open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
        return _activate(role, target, label)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def upload_folder_file(upload_id: str, filename: str, source: Path) -> None:
    if not (len(upload_id) == 32 and all(c in "0123456789abcdef" for c in upload_id)):
        raise ValueError("Invalid upload ID")
    if filename not in REQUIRED:
        raise ValueError("Unexpected checkpoint file")
    if source.stat().st_size == 0 or source.stat().st_size > MAX_FILE:
        raise ValueError("Checkpoint file must be between 1 byte and 100 MB")
    target = STORE / upload_id
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / (filename + ".upload")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target / filename)
    finally:
        temporary.unlink(missing_ok=True)


def finish_folder(role: str, upload_id: str, label: str) -> dict:
    if not (len(upload_id) == 32 and all(c in "0123456789abcdef" for c in upload_id)):
        raise ValueError("Invalid upload ID")
    return _activate(role, STORE / upload_id, label)
