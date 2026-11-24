"""Launch the installed GENTRL worker with uploaded checkpoint selection."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from time import perf_counter

from app.checkpoints import selected
from app.screening import GenerationScreen


def main() -> None:
    root = Path(os.environ["GENTRL_ROOT"])
    script = root / "examples" / "sample_worker.py"
    sys.path.insert(0, str(root / "examples"))
    worker = runpy.run_path(str(script))
    entry = worker["main"]
    original = entry.__globals__["_checkpoint_file"]
    active_checkpoint = None

    def checkpoint_file(stage=None):
        nonlocal active_checkpoint
        role = "pretrained" if stage == "before_rl" else "after_rl" if stage in (None, "after_rl") else None
        choice = selected(role) if role else None
        if stage is None and choice is None:
            choice = selected("pretrained")
        active_checkpoint = choice or original(stage)
        return active_checkpoint

    entry.__globals__["_checkpoint_file"] = checkpoint_file
    # Count every decoded sample, including duplicates and property-filter rejects.
    import json
    import builtins
    from rdkit import Chem
    state = {}
    original_sample = entry.__globals__["sample_valid"]

    def sample_counted(model, n, reward_fn):
        state.clear()
        state.update(attempted=0, valid=0)
        if active_checkpoint is None:
            raise RuntimeError("Cannot screen candidates without an active checkpoint")
        screen = GenerationScreen(Path(active_checkpoint))
        sample = model.sample
        parsed_round: dict[str, Chem.Mol | None] = {}

        def counted(*args, **kwargs):
            values = sample(*args, **kwargs)
            parsed_round.clear()
            for value in values:
                state["attempted"] += 1
                if value not in parsed_round:
                    parsed_round[value] = Chem.MolFromSmiles(value)
                mol = parsed_round[value]
                state["valid"] += int(mol is not None and mol.GetNumAtoms() > 0)
            return values

        def accepts_sample(smiles: str) -> bool:
            return screen.accept(parsed_round[smiles])

        model.sample = counted
        original_valid = original_sample.__globals__["valid"]
        original_sample.__globals__["valid"] = accepts_sample
        try:
            # Only candidates accepted by the screen count toward the requested total.
            started = perf_counter()
            molecules = original_sample(model, n, reward_fn, max_rounds=float("inf"))
            elapsed = perf_counter() - started
            if len(molecules) != n:
                raise RuntimeError(f"The sampler stopped with {len(molecules)} of {n} requested candidates.")
            screen.save()
            state["screening"] = {
                "requested": n,
                "reference_coverage": screen.reference_coverage,
                "rejections": dict(screen.rejections),
                "elapsed_seconds": elapsed,
                "accepted_per_minute": len(molecules) * 60 / elapsed if elapsed > 0 else None,
            }
            return molecules
        finally:
            original_sample.__globals__["valid"] = original_valid
            model.sample = sample

    def report(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str) and args[0].startswith('{'):
            data = json.loads(args[0])
            if "molecules" in data:
                data["screening"] = state.pop("screening")
                data["attempts"] = dict(state)
                data["checkpoint"] = str(active_checkpoint)
                args = (json.dumps(data),)
        builtins.print(*args, **kwargs)

    entry.__globals__["sample_valid"] = sample_counted
    entry.__globals__["print"] = report
    entry()


if __name__ == "__main__":
    main()
