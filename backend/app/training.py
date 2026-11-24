"""Train-tab status. The buttons run NewPretraining.ipynb and train_rl.ipynb."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from app.checkpoints import clear, list_checkpoints, selected

DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_RL_ITERATIONS = 50

GENTRL_ROOT = Path(
    os.environ.get(
        "GENTRL_ROOT",
        r"D:\Downloads\GENTRL ver2\Code\Data-handling-for-drug-discovery-using-deep-learning-main",
    )
)
EXAMPLES = GENTRL_ROOT / "examples"
CHECKPOINT = EXAMPLES / "gentrl_trained_rl.pth"
BASE_CHECKPOINT = EXAMPLES / "saved_gentrl_zinc"
PRETRAIN_CHECKPOINT = EXAMPLES / "saved_gentrl"
GENTRL_PYTHON = Path(os.environ.get("GENTRL_PYTHON", str(GENTRL_ROOT / ".venv" / "Scripts" / "python.exe")))
NOTEBOOK_RUNNER = Path(__file__).resolve().parent / "run_notebook.py"
PRETRAIN_NOTEBOOK = EXAMPLES / "NewPretraining.ipynb"
RL_NOTEBOOK = EXAMPLES / "train_rl.ipynb"

_lock = threading.Lock()
_proc: subprocess.Popen[str] | None = None
_status = "idle"
_stage = "idle"
_message = "No run is in progress. Generate samples from the checkpoint below."
_error: str | None = None
_rec: list[float] = []
_reward: list[float] = []
_checkpoint: str | None = None
_device = ""
_step = 0
_total = 0
_dataset = ""
_phase = ""
_run_action = ""
_input_checkpoint: str | None = None
_input_label: str | None = None

def _read_worker(proc: subprocess.Popen[str]) -> None:
    global _status, _message, _error
    if proc.stdout is None:
        return
    decoder = json.JSONDecoder()
    for line in proc.stdout:
        start = line.find("{")
        while start != -1:
            try:
                event, offset = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                break
            start = line.find("{", start + max(offset, 1))
            if isinstance(event, dict):
                _apply_event(event)
    code = proc.wait()
    with _lock:
        if _status == "running":
            _status = "idle"
            if code != 0 and not _error:
                _error = f"Pretrain stopped ({code}). See examples/train_stderr.log"
                _message = _error


def _apply_event(event: dict) -> None:
    global _status, _stage, _message, _error, _checkpoint, _rec, _reward, _device, _step, _total, _dataset, _phase
    kind = event.get("event")
    with _lock:
        if kind == "device":
            _device = str(event.get("device") or "")
        elif kind == "progress":
            _step = int(event.get("step") or 0)
            _total = int(event.get("total") or _total)
            if event.get("phase"):
                _phase = str(event.get("phase"))
            if event.get("dataset"):
                _dataset = Path(str(event.get("dataset"))).name
            if event.get("device"):
                _device = str(event.get("device"))
        elif kind == "stage":
            _stage = str(event.get("stage") or "pretrain")
            if _stage == "rl":
                _message = "Reinforcement learning is running."
            else:
                _message = f"Running {Path(str(event.get('dataset') or 'pretrain.ipynb')).name}"
        elif kind == "log":
            text = str(event.get("message") or "").strip()
            if text:
                _message = text
        elif kind == "metrics" and event.get("stage") != "rl":
            rec = (event.get("metrics") or {}).get("rec")
            if rec is not None:
                # GENTRL reports mean reconstruction log-probability (`rec`).
                # Negating it gives the reconstruction loss shown in the UI.
                _rec.append(-float(rec))
        elif kind == "metrics" and event.get("stage") == "rl":
            reward = (event.get("metrics") or {}).get("mean_reward")
            if reward is not None:
                _reward.append(float(reward))
        elif kind == "done":
            completed_phase = _run_action
            clear("base" if completed_phase == "zinc" else "pretrained" if completed_phase == "continue" else "after_rl")
            completed_checkpoint = (
                BASE_CHECKPOINT if completed_phase == "zinc"
                else PRETRAIN_CHECKPOINT if completed_phase == "continue"
                else CHECKPOINT
            )
            if completed_checkpoint.exists():
                _checkpoint = str(completed_checkpoint)
            _status = "idle"
            _phase = completed_phase
            _message = (
                "Reinforcement learning finished. Generate will sample from this checkpoint."
                if completed_phase == "rl"
                else "Base model training finished. Pretraining and reinforcement learning can now use this checkpoint."
                if completed_phase == "zinc"
                else "Pretraining finished. The fine-tuned checkpoint was saved."
            )
            _stage = "done"
        elif kind == "error":
            _status = "idle"
            _error = str(event.get("message") or "Pretrain failed")
            _message = _error


def _progress_fields() -> dict:
    return {
        "device": _device,
        "step": _step,
        "total": _total,
        "dataset": _dataset,
        "phase": _phase,
    }


def _safe_dataset(name: str) -> str:
    filename = Path(name).name
    if filename != name or not filename.lower().endswith(".csv"):
        raise RuntimeError("Choose a CSV file name.")
    if not (EXAMPLES / filename).is_file():
        raise RuntimeError(f"{filename} is not in the GENTRL examples folder.")
    return filename


def _train_settings(epochs: int, batch_size: int, learning_rate: float, rl_iterations: int) -> dict:
    if not 1 <= epochs <= 50:
        raise RuntimeError("Epochs must be between 1 and 50.")
    if not 1 <= batch_size <= 64:
        raise RuntimeError("Batch size must be between 1 and 64.")
    if not 1e-5 <= learning_rate <= 1e-3:
        raise RuntimeError("Learning rate must be between 0.00001 and 0.001.")
    if rl_iterations < 1:
        raise RuntimeError("RL iterations must be at least 1.")
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "rl_iterations": rl_iterations,
    }


def _launch_notebook(
    notebook: Path,
    phase: str,
    message: str,
    dataset_name: str = "",
    settings: dict | None = None,
) -> dict:
    global _proc, _status, _stage, _message, _error, _phase, _device, _step, _total, _dataset, _rec, _reward, _checkpoint, _run_action, _input_checkpoint, _input_label
    if not notebook.is_file():
        raise RuntimeError(f"Missing {notebook.name}")
    with _lock:
        if _status == "running" and _proc is not None and _proc.poll() is None:
            raise RuntimeError("A training run is already going")
        if not GENTRL_PYTHON.is_file():
            raise RuntimeError("GENTRL Python is not installed")
        _error = None
        _checkpoint = None
        if phase in {"zinc", "continue"}:
            _rec = []
        _reward = []
        _stage = phase
        _phase = phase
        _run_action = phase
        source_role = "base" if phase == "continue" else "pretrained" if phase == "rl" else None
        source_key = "base_checkpoint" if phase == "continue" else "pretrain_checkpoint" if phase == "rl" else None
        _input_checkpoint = str((settings or {}).get(source_key)) if source_key else None
        _input_label = (list_checkpoints().get(source_role) if source_role else None) or ("Built-in checkpoint" if source_role else None)
        _device = ""
        _step = 0
        _total = 0
        _dataset = dataset_name or notebook.name
        _message = message
        log_path = EXAMPLES / "train_stderr.log"
        command = [
            str(GENTRL_PYTHON),
            "-u",
            str(NOTEBOOK_RUNNER),
            str(notebook),
            phase,
            dataset_name,
            json.dumps(settings or _train_settings(
                DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_RL_ITERATIONS,
            )),
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log:
                _proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    cwd=str(notebook.parent),
                )
        except OSError as exc:
            _status = "idle"
            _error = f"Could not start training: {exc}"
            _message = _error
            raise RuntimeError(_error) from exc
        _status = "running"
        threading.Thread(target=_read_worker, args=(_proc,), daemon=True).start()
    return snapshot()


def start_zinc(settings: dict) -> dict:
    return _launch_notebook(
        PRETRAIN_NOTEBOOK,
        "zinc",
        "Training the GENTRL base model on zinc_plogp_300k_FinalSTEREO.csv. A separate checkpoint will be saved when training finishes.",
        settings=settings,
    )


def _require_base_checkpoint() -> None:
    checkpoint = selected("base") or BASE_CHECKPOINT
    required = ("enc.model", "dec.model", "lp.model", "order.pkl")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Base model checkpoint is incomplete ({', '.join(missing)}). "
            "Run Train Base Model first."
        )


def _require_pretrain_checkpoint() -> None:
    if selected("pretrained") is None:
        _require_base_checkpoint()
    checkpoint = selected("pretrained") or PRETRAIN_CHECKPOINT
    required = ("enc.model", "dec.model", "lp.model", "order.pkl")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Pretraining checkpoint is incomplete ({', '.join(missing)}). "
            "Run Start pretraining first."
        )
    base_checkpoint = selected("base") or BASE_CHECKPOINT
    if selected("pretrained") is None and (PRETRAIN_CHECKPOINT / "enc.model").stat().st_mtime < (base_checkpoint / "enc.model").stat().st_mtime:
        raise RuntimeError("The pretraining checkpoint predates the current base model. Run Start pretraining again before reinforcement learning.")


def start_pretrain(dataset: str, settings: dict) -> dict:
    filename = _safe_dataset(dataset)
    _require_base_checkpoint()
    settings = {**settings, "base_checkpoint": str(selected("base") or BASE_CHECKPOINT)}
    return _launch_notebook(
        PRETRAIN_NOTEBOOK,
        "continue",
        f"Continuing from the GENTRL base model checkpoint on {filename}.",
        filename,
        settings,
    )


def snapshot() -> dict:
    path = str(
        BASE_CHECKPOINT if _run_action == "zinc"
        else PRETRAIN_CHECKPOINT if _run_action == "continue"
        else CHECKPOINT
    )
    with _lock:
        return {
            "status": "running" if _status == "running" else "idle",
            "stage": _stage,
            "checkpoint": _checkpoint or path,
            "input_checkpoint": _input_checkpoint,
            "input_label": _input_label,
            "rec": list(_rec),
            "reward": list(_reward),
            "message": _message,
            "error": _error,
            **_progress_fields(),
        }


def start_rl(dataset: str, settings: dict) -> dict:
    filename = _safe_dataset(dataset)
    _require_pretrain_checkpoint()
    settings = {**settings, "pretrain_checkpoint": str(selected("pretrained") or PRETRAIN_CHECKPOINT)}
    return _launch_notebook(
        RL_NOTEBOOK,
        "rl",
        f"Running reinforcement learning from the pretrained GENTRL checkpoint on {filename}.",
        filename,
        settings,
    )


def stop_run() -> dict:
    global _status, _stage, _message, _error
    with _lock:
        proc = _proc
        running = _status == "running" and proc is not None and proc.poll() is None
        if running:
            _status = "idle"
            _stage = "stopped"
            _error = None
            _message = "Run stopped. The curves and the last saved checkpoint are unchanged."
        else:
            _message = "No run is in progress."
            proc = None
    if proc is None:
        return snapshot()
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], check=False, capture_output=True)
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    return snapshot()


def request_run(
    action: str,
    dataset: str = "Leshmania.csv",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    rl_iterations: int = DEFAULT_RL_ITERATIONS,
) -> dict:
    settings = _train_settings(epochs, batch_size, learning_rate, rl_iterations)
    if action == "zinc":
        return start_zinc(settings)
    if action == "pretrain":
        return start_pretrain(dataset, settings)
    if action == "rl":
        return start_rl(dataset, settings)
    raise RuntimeError("Unknown training action")


def save_reference(filename: str, text: str) -> dict:
    name = Path(filename).name
    if name != filename or not name.lower().endswith(".csv"):
        raise RuntimeError("Upload a .csv file.")
    (EXAMPLES / name).write_text(text, encoding="utf-8")
    return {"name": name}
