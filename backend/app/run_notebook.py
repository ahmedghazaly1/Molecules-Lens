"""Run a GENTRL ver2 notebook and send its printed lines back as JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _text(cell) -> str:
    source = cell.source
    return source if isinstance(source, str) else "".join(source)


def _set_text(cell, text: str) -> None:
    cell.source = text


def _apply_settings(nb, settings: dict) -> None:
    """Swap training numbers in the copy that is about to run."""
    epochs = int(settings.get("epochs", 10))
    batch_size = int(settings.get("batch_size", 8))
    learning_rate = float(settings.get("learning_rate", 1e-4))
    rl_iterations = int(settings.get("rl_iterations", 50))
    rate = format(learning_rate, ".12g")
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = _text(cell)
        text = re.sub(r"batch_size=\d+", f"batch_size={batch_size}", text)
        text = re.sub(r"num_epochs=\d+", f"num_epochs={epochs}", text)
        text = re.sub(r"lr=[0-9.eE+-]+", f"lr={rate}", text)
        text = re.sub(r"num_iterations=\d+", f"num_iterations={rl_iterations}", text)
        _set_text(cell, text)


def _running_lines(nb) -> list[str]:
    needles = ("batch_size=", "num_epochs=", "lr=", "num_iterations=", "RL_ITERATIONS =")
    lines = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for line in _text(cell).splitlines():
            if any(needle in line for needle in needles) and not line.lstrip().startswith("#"):
                lines.append(line.strip())
    return lines


BASE_CHECKPOINT = "saved_gentrl_zinc"


def _load_base_cell() -> str:
    return f"model.load('{BASE_CHECKPOINT}/')\nprint('Loaded GENTRL base model checkpoint')\n"


def _make_checkpoint_directories_portable(nb) -> None:
    """Replace notebook shell mkdir calls and create model-save folders on any OS."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = _text(cell)
        source = re.sub(
            r"(?m)^!\s*mkdir\s+-p\s+([^\s]+)\s*$",
            lambda match: f"from pathlib import Path\nPath({match.group(1)!r}).mkdir(parents=True, exist_ok=True)",
            source,
        )
        save_folders = re.findall(r"model\.save\(\s*['\"]([^'\"]+)['\"]\s*\)", source)
        if save_folders:
            setup = "from pathlib import Path\n" + "\n".join(
                f"Path({folder!r}).mkdir(parents=True, exist_ok=True)" for folder in save_folders
            )
            source = setup + "\n" + source
        _set_text(cell, source)


def _report_each_rl_iteration(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = _text(cell)
        if "model.train_as_rl(" not in source:
            continue
        source = re.sub(
            r"model\.train_as_rl\(([^\n)]*)\)",
            lambda match: match.group(0) if "verbose_step=" in match.group(1)
            else f"model.train_as_rl({match.group(1)}, verbose_step=1)",
            source,
        )
        _set_text(cell, source)


def _metrics_from_line(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for part in line.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lstrip("!").strip()
        if key not in {"rec", "mean_reward", "valid_perc", "loss", "kl"}:
            continue
        try:
            metrics[key] = float(value.strip())
        except ValueError:
            continue
    return metrics


def prepare_notebook(nb, mode: str, dataset: str, settings: dict | None = None):
    """Train the base model on MOSES molecules or continue from its checkpoint."""
    if mode == "rl" and any(
        "moleculelens-run" in cell.metadata.get("tags", []) for cell in nb.cells
    ):
        _prepare_new_rl(nb, dataset, settings or {})
        return
    if mode in {"zinc", "continue"} and any(
        "MOLECULELENS_300K_STAGE" in _text(cell) for cell in nb.cells
    ):
        _prepare_new_pretraining(nb, mode, dataset, settings or {})
        return
    _apply_settings(nb, settings or {})
    if dataset and dataset != "Leshmania.csv":
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            _set_text(cell, _text(cell).replace("Leshmania.csv", dataset))
    if mode == "zinc":
        kept = []
        in_reference_section = False
        for cell in nb.cells:
            source = _text(cell)
            if cell.cell_type == "markdown" and "Leshmania dataset" in source:
                in_reference_section = True
                continue
            if in_reference_section and "enc = gentrl.RNNEncoder" in source:
                in_reference_section = False
            if in_reference_section or "Leshmania_loader" in source:
                continue
            kept.append(cell)
            if "saved_gentrl_zinc" in source and "model.save" in source:
                break
        nb.cells = kept
        _make_checkpoint_directories_portable(nb)
        return
    if mode == "rl":
        # train_rl.ipynb loads saved_gentrl/, the checkpoint saved by pretraining.
        # The remaining cells are post-run analysis and can fail after a valid
        # checkpoint has already been saved. Keep the training path only.
        nb.cells = [nb.cells[i] for i in (0, 1, 2, 3, 5, 6, 10, 11, 13, 14, 16)]
        _report_each_rl_iteration(nb)
        _make_checkpoint_directories_portable(nb)
        return
    if mode != "continue":
        return
    kept = []
    for cell in nb.cells:
        text = _text(cell)
        if "dataset_v1.csv" in text or "Zinc_loader = DataLoader" in text:
            continue
        if "saved_gentrl_zinc" in text and "model.save" in text:
            continue
        if "train_as_vaelp(Zinc_loader" in text:
            _set_text(cell, _load_base_cell())
        kept.append(cell)
    nb.cells = kept
    _make_checkpoint_directories_portable(nb)


def _prepare_new_pretraining(nb, mode: str, dataset: str, settings: dict) -> None:
    """Pass the selected stage and settings to the adapted 300K notebook."""
    import nbformat

    config = "\n".join([
        f"MOLECULELENS_300K_STAGE = {mode!r}",
        f"TRAINING_DATASET = {dataset!r}",
        f"TRAINING_EPOCHS = {int(settings.get('epochs', 3))}",
        f"TRAINING_BATCH_SIZE = {int(settings.get('batch_size', 64))}",
        f"TRAINING_LEARNING_RATE = {float(settings.get('learning_rate', 1e-4))!r}",
        f"BASE_CHECKPOINT_INPUT = {str(settings.get('base_checkpoint', 'saved_gentrl_zinc'))!r}",
    ])
    nb.cells[0] = nbformat.v4.new_code_cell(config)
    if mode == "continue":
        for cell in nb.cells:
            if cell.cell_type == "code":
                _set_text(cell, _text(cell).replace(
                    "model.load('saved_gentrl_zinc/')",
                    "model.load(BASE_CHECKPOINT_INPUT)\n"
                    "    from pathlib import Path\n"
                    "    Path('moleculelens_loaded_checkpoint.txt').write_text(str(BASE_CHECKPOINT_INPUT), encoding='utf-8')",
                ))


def _prepare_new_rl(nb, dataset: str, settings: dict) -> None:
    """Execute only the tagged RL training cells with explicit run settings."""
    cells = [cell for cell in nb.cells if "moleculelens-run" in cell.metadata.get("tags", [])]
    if len(cells) != 6:
        raise ValueError("The RL notebook training cells are incomplete")
    config = "\n".join([
        f"RL_DATASET = {dataset!r}",
        f"RL_ITERATIONS = {int(settings.get('rl_iterations', 50))}",
        "RL_BATCH_SIZE = 50",
        "CHECKPOINT_EVERY = 25",
        "RUN_COMPARISON = False",
        f"PRETRAIN_CHECKPOINT_INPUT = {str(settings.get('pretrain_checkpoint', 'saved_gentrl'))!r}",
    ])
    cells[0] = nbformat.v4.new_code_cell(config)
    for cell in cells:
        if cell.cell_type == "code":
            source = _text(cell).replace('pretrain_checkpoint = Path("saved_gentrl")', 'pretrain_checkpoint = Path(PRETRAIN_CHECKPOINT_INPUT)')
            source = source.replace(
                "model.cpu()",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                "model = model.to(device)\n"
                "print('Model is on device:', next(model.parameters()).device, flush=True)",
            )
            source = source.replace(
                'model.load(str(pretrain_checkpoint))',
                "model.load(str(pretrain_checkpoint))\n"
                "Path('moleculelens_loaded_checkpoint.txt').write_text(str(pretrain_checkpoint), encoding='utf-8')",
            )
            _set_text(cell, source)
    nb.cells = cells


def main() -> None:
    notebook = Path(sys.argv[1]).resolve()
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    dataset = sys.argv[3] if len(sys.argv) > 3 else "Leshmania.csv"
    settings = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}
    with notebook.open(encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)
    prepare_notebook(nb, mode, dataset, settings)
    if mode in {"continue", "rl"}:
        # Capture lineage before training can overwrite a checkpoint.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.properties import load_reference
        inherited = load_reference(Path(settings["pretrain_checkpoint"])) if mode == "rl" else None
        backend = str(Path(__file__).resolve().parents[1])
        capture = ("_quality_smiles = frame['SMILES'].astype(str).tolist()\n_quality_outputs = [Path('saved_gentrl'), Path(final_checkpoint)]\n" if mode == "continue" else "_quality_smiles = list(reward.reference_smiles)\n_quality_outputs = [Path('gentrl_trained_rl.pth'), Path('saved_gentrl_after_rl')]\n")
        complete = mode == "continue" or bool(inherited and inherited.get("complete"))
        nb.cells.append(nbformat.v4.new_code_cell(
            f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {backend!r})\n"
            "from app.properties import save_reference as _save_quality_reference\n"
            + capture
            + "for _quality_output in _quality_outputs:\n"
            + f"    _save_quality_reference(_quality_output, _quality_smiles, {dataset!r}, {inherited!r}, complete={complete!r})\n"
            + "print('Saved target training reference for novelty checks', flush=True)\n"
        ))
    if mode in {"continue", "rl"}:
        (notebook.parent / "moleculelens_loaded_checkpoint.txt").unlink(missing_ok=True)
    running = _running_lines(nb)
    (notebook.parent / "moleculelens_run.txt").write_text(
        "\n".join(running) + "\n",
        encoding="utf-8",
    )
    stage = "rl" if mode == "rl" or "train_rl" in notebook.name else "pretrain"
    nb.cells.insert(0, nbformat.v4.new_code_cell(
        "print(" + json.dumps("MoleculeLens run: " + " | ".join(running)) + ", flush=True)\n"
    ))
    pending: list[str] = []

    def on_stream(msg) -> None:
        if msg.get("msg_type") != "stream":
            return
        pending.append(msg.get("content", {}).get("text") or "")
        if not pending[-1].endswith("\n"):
            return
        block = "".join(pending)
        pending.clear()
        emit({"event": "log", "message": block.strip()[:500]})
        for line in block.splitlines():
            if line.startswith("Model is on device:"):
                emit({"event": "device", "device": line.split(":", 1)[1].strip()})
            rl_progress = re.match(r"RL iteration (\d+)/(\d+) (started|completed)", line)
            if rl_progress:
                iteration, total, state = rl_progress.groups()
                emit({
                    "event": "progress",
                    "phase": "rl",
                    "step": int(iteration) if state == "completed" else int(iteration) - 1,
                    "total": int(total),
                })
            if line.startswith("sampled:"):
                parts = line.removeprefix("sampled:").strip().split(" of ")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    emit({
                        "event": "progress",
                        "phase": "sampling",
                        "step": int(parts[0]),
                        "total": int(parts[1]),
                    })
            if "rec:" not in line and "mean_reward:" not in line:
                continue
            metrics = _metrics_from_line(line)
            if metrics:
                emit({"event": "metrics", "stage": stage, "metrics": metrics})

    class StreamingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            on_stream(msg)
            return super().process_message(msg, cell, cell_index)

    # Printed by the kernel during model.sample(N). The notebook calls sample once
    # and stays silent until all N molecules exist.
    nb.cells.insert(0, nbformat.v4.new_code_cell(
        "import gentrl as _gentrl\n"
        "_orig_sample = _gentrl.GENTRL.sample\n"
        "def _sample_progress(self, num_samples):\n"
        "    if num_samples < 500:\n"
        "        return _orig_sample(self, num_samples)\n"
        "    out = []\n"
        "    done = 0\n"
        "    while done < num_samples:\n"
        "        take = min(50, num_samples - done)\n"
        "        out.extend(_orig_sample(self, take))\n"
        "        done += take\n"
        "        print(f'sampled: {done} of {num_samples}', flush=True)\n"
        "    return out\n"
        "_gentrl.GENTRL.sample = _sample_progress\n"
    ))
    emit({"event": "stage", "stage": stage, "dataset": notebook.name})
    client = StreamingClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.create_kernel_manager = lambda: KernelManager(
        kernel_name="python3",
        kernel_cmd=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    )
    try:
        client.execute()
    except Exception as exc:
        emit({"event": "error", "message": str(exc)})
        sys.exit(1)
    emit({"event": "done", "notebook": notebook.name})


if __name__ == "__main__":
    main()
