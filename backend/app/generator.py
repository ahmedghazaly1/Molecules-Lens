"""Sampler interface: GENTRL checkpoint when the worker starts, mock otherwise."""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
from pathlib import Path
from typing import Protocol

MOCK_SMILES = [
    "CCO",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CCN(CC)CCOC(=O)C1=CC=CC=C1N",
    "CC(C)NCC(O)COC1=CC=CC2=C1C=CC=C2",
    "CN1CCC(CC1)C2=C(OC3=CC=CC=C23)C4=CC=CC=C4",
    "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3",
    "C1=CC=C(C=C1)CCN",
    "CC(C)(C)NCC(C1=CC=C(C=C1)O)O",
    "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2",
    "CC1=CC(=NO1)C2=CC=CC=C2",
    "C1CCC(CC1)N",
    "CC(=O)NC1=CC=C(C=C1)O",
    "C1=CC=C2C(=C1)C=CC=C2O",
    "COC1=CC=CC=C1OC",
]

GENTRL_ROOT = Path(
    os.environ.get(
        "GENTRL_ROOT",
        r"D:\Downloads\GENTRL ver2\Code\Data-handling-for-drug-discovery-using-deep-learning-main",
    )
)
GENTRL_PYTHON = Path(
    os.environ.get("GENTRL_PYTHON", str(GENTRL_ROOT / ".venv" / "Scripts" / "python.exe"))
)
GENTRL_WORKER = GENTRL_ROOT / "examples" / "sample_worker.py"
MODEL_WORKER = Path(__file__).resolve().parent / "model_worker.py"


class MoleculeSampler(Protocol):
    def generate(self, n: int, stage: str = "after_rl") -> list[dict]:
        ...


class MockSampler:
    source = "mock"

    def generate(self, n: int, stage: str = "after_rl") -> list[dict]:
        if n <= 0:
            return []
        if n <= len(MOCK_SMILES):
            chosen = random.sample(MOCK_SMILES, n)
        else:
            chosen = MOCK_SMILES + [random.choice(MOCK_SMILES) for _ in range(n - len(MOCK_SMILES))]
        return [{"smiles": smiles, "reward": None} for smiles in chosen]


class GentrlSampler:
    """Talks to a long-lived GENTRL process that already loaded the checkpoint."""

    source = "gentrl"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None

    def _start(self) -> None:
        if not GENTRL_PYTHON.is_file() or not GENTRL_WORKER.is_file():
            raise FileNotFoundError(
                f"GENTRL python or worker missing: {GENTRL_PYTHON} / {GENTRL_WORKER}"
            )
        log = Path(__file__).resolve().parents[1] / "worker_stderr.log"
        self._proc = subprocess.Popen(
            [str(GENTRL_PYTHON), "-u", str(MODEL_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log.open("w", encoding="utf-8"),
            text=True,
            cwd=str(GENTRL_ROOT),
            env={**os.environ, "GENTRL_ROOT": str(GENTRL_ROOT),
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )

    def _read_json(self) -> dict:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("GENTRL worker closed stdout")
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)


    def _ensure(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start()
            ready = self._read_json()
            if not ready.get("ok"):
                raise RuntimeError(ready.get("error", "GENTRL worker failed to load"))

    def _request(self, payload: dict) -> dict:
        with self._lock:
            self._ensure()
            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            data = self._read_json()
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "GENTRL request failed"))
            return data

    def generate(self, n: int, stage: str = "after_rl") -> list[dict]:
        if n <= 0:
            return []
        data = self._request({"n": n, "stage": stage})
        if data.get("stage") != stage:
            raise RuntimeError("Restart the MoleculeLens API to activate the model stage selector.")
        molecules = data.get("molecules") or []
        return SampleBatch([
            {"smiles": item["smiles"], "reward": item.get("reward")}
            for item in molecules
            if item.get("smiles")
        ], data.get("attempts"), data.get("checkpoint"), data.get("screening"))

    def explain_reward(self, smiles: str) -> dict:
        data = self._request({"cmd": "explain", "smiles": smiles})
        if "shap" not in data:
            raise RuntimeError("GENTRL worker has no reward explanation. Restart the API so it reloads the worker.")
        return {
            "reward": data["reward"],
            "baseline": data["baseline"],
            "inputs": data["inputs"],
            "reference_means": data["reference_means"],
            "shap": data["shap"],
            "structure_similarity": data.get("structure_similarity"),
            "property_similarity": data.get("property_similarity"),
            "reference": data.get("reference"),
            "reference_rows": data.get("reference_rows"),
        }


class FallbackSampler:
    """Prefer GENTRL; use mock only if the worker cannot load or sample."""

    def __init__(self) -> None:
        self._gentrl = GentrlSampler()
        self._mock = MockSampler()
        self.source = "gentrl"

    def generate(self, n: int, stage: str = "after_rl") -> list[dict]:
        molecules = self._gentrl.generate(n, stage)
        self.source = "gentrl"
        return molecules

    def explain_reward(self, smiles: str) -> dict:
        return self._gentrl.explain_reward(smiles)


def build_sampler() -> MoleculeSampler:
    return FallbackSampler()


class SampleBatch(list):
    def __init__(self, values, attempts=None, checkpoint=None, screening=None):
        super().__init__(values)
        self.attempts = attempts
        self.checkpoint = checkpoint
        self.screening = screening


sampler: MoleculeSampler = build_sampler()
