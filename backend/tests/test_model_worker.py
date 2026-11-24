import io
import json
import unittest
from contextlib import redirect_stdout
from types import ModuleType
from unittest.mock import Mock, patch

from rdkit import Chem

from app import model_worker


class ModelWorkerTests(unittest.TestCase):
    def test_generation_keeps_sampling_until_1000_candidates_pass(self):
        # The worker adapter replaces globals in the external sampler module.
        external = ModuleType("sample_worker")
        external.json = json
        exec(
            """
def _checkpoint_file(stage):
    return 'test-checkpoint.pth'

def valid(smiles):
    return True

class Model:
    def __init__(self):
        self.round = 0

    def sample(self, size):
        self.round += 1
        return [f'[{self.round}CH4]']

def sample_valid(model, n, reward_fn, max_rounds=40):
    accepted = []
    rounds = 0
    while len(accepted) < n and rounds < max_rounds:
        rounds += 1
        for smiles in model.sample(128):
            if valid(smiles):
                accepted.append({'smiles': smiles})
    return accepted

def main():
    _checkpoint_file('after_rl')
    molecules = sample_valid(Model(), 1000, None)
    print(json.dumps({'molecules': molecules}))
""",
            external.__dict__,
        )
        screen = Mock(reference_coverage="complete", rejections={"physicochemical": 3000})
        screen.accept.side_effect = lambda mol: mol.GetAtomWithIdx(0).GetIsotope() % 4 == 0

        output = io.StringIO()
        with (
            patch.dict(model_worker.os.environ, {"GENTRL_ROOT": "."}),
            patch.object(model_worker.sys, "path", model_worker.sys.path.copy()),
            patch.object(model_worker.runpy, "run_path", return_value=external.__dict__),
            patch.object(model_worker, "selected", return_value=None),
            patch.object(model_worker, "GenerationScreen", return_value=screen),
            patch.object(model_worker, "perf_counter", side_effect=[20.0, 50.0]),
            patch.object(Chem, "MolFromSmiles", wraps=Chem.MolFromSmiles) as parse_smiles,
            redirect_stdout(output),
        ):
            model_worker.main()

        result = json.loads(output.getvalue())
        self.assertEqual(len(result["molecules"]), 1000)
        self.assertEqual(result["attempts"]["attempted"], 4000)
        self.assertEqual(parse_smiles.call_count, 4000)
        self.assertEqual(result["screening"]["requested"], 1000)
        self.assertEqual(result["screening"]["elapsed_seconds"], 30.0)
        self.assertEqual(result["screening"]["accepted_per_minute"], 2000.0)
        screen.save.assert_called_once()
