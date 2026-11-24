"""Score SMILES with the MoleculeX pIC50 checkpoint, in that project's venv."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

PIC50_ROOT = Path(
    os.environ.get(
        "PIC50_ROOT",
        r"D:\PROJECTS\MoleculeX pIC50\MoleculeX-Predictive-Transformer-Model",
    )
)
PIC50_PYTHON = Path(
    os.environ.get("PIC50_PYTHON", str(PIC50_ROOT / ".venv" / "Scripts" / "python.exe"))
)
PIC50_SCRIPT = PIC50_ROOT / "train_predict_pic50.py"


class Pic50Scorer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None

    def _start(self) -> None:
        if not PIC50_PYTHON.is_file() or not PIC50_SCRIPT.is_file():
            raise FileNotFoundError(f"pIC50 python or script missing: {PIC50_PYTHON} / {PIC50_SCRIPT}")
        log = Path(os.environ.get("PIC50_LOG", str(PIC50_ROOT / "pic50_worker_stderr.log")))
        self._proc = subprocess.Popen(
            [str(PIC50_PYTHON), "-u", str(PIC50_SCRIPT), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log.open("w", encoding="utf-8"),
            text=True,
            cwd=str(PIC50_ROOT),
        )

    def _read_json(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("pIC50 worker closed stdout")
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def score(self, smiles: list[str]) -> list[float | None]:
        if not smiles:
            return []
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
                ready = self._read_json()
                if not ready.get("ok"):
                    raise RuntimeError(ready.get("error", "pIC50 worker failed to load"))
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(json.dumps({"smiles": smiles}) + "\n")
            self._proc.stdin.flush()
            data = self._read_json()
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "pIC50 scoring failed"))
            values = data.get("predictions") or []
            if len(values) != len(smiles):
                raise RuntimeError("pIC50 worker returned the wrong number of scores")
            return [float(value) for value in values]


scorer = Pic50Scorer()
