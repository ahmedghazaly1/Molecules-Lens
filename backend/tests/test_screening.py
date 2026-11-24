import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from app import screening


class GenerationScreenTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.store = self.root / "models"
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"model")
        self.enterContext(patch.object(screening, "STORE", self.store))
        self.enterContext(patch.dict(screening.os.environ, {"MOLECULELENS_ACCEPTED_CSV": ""}))

    def test_screen_rejects_known_invalid_alerted_and_low_quality_molecules(self):
        reference = {"complete": False, "smiles": ["CCO"]}
        with patch.object(screening, "load_reference", return_value=reference):
            screen = screening.GenerationScreen(self.checkpoint)
        self.assertEqual(screen.reference_coverage, "partial")
        for smiles in ["", "OCC", "Oc1ccccc1O", "CO", "c1ccccc1"]:
            self.assertFalse(screen.accept(Chem.MolFromSmiles(smiles)), smiles)
        self.assertEqual(dict(screen.rejections), {
            "invalid": 1, "training_match": 1, "structural_alert": 1, "physicochemical": 1, "sa_or_qed": 1,
        })

    def test_logs_and_qed_boundaries_use_unrounded_scores(self):
        screen = screening.GenerationScreen(self.checkpoint)
        with patch.object(screening, "estimated_logs", side_effect=[-4.01, -1.0]):
            self.assertFalse(screen.accept(Chem.MolFromSmiles("CCOc1ccc(NC(C)=O)cc1")))
            self.assertTrue(screen.accept(Chem.MolFromSmiles("COc1ccc(CCNC(C)=O)cc1")))
        with patch.object(screening.QED, "qed", return_value=0.6):
            self.assertFalse(screen.accept(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")))
        self.assertEqual(dict(screen.rejections), {"logs": 1, "sa_or_qed": 1})

    def test_library_seeds_scaffolds_and_persists_across_requests(self):
        (self.root / "RIPK1_FINAL_ACCEPTED.csv").write_text("SMILES\nCc1ccccc1\n", encoding="utf-8")
        with patch.object(screening, "MAX_PER_SCAFFOLD", 2):
            screen = screening.GenerationScreen(self.checkpoint)
            self.assertTrue(screen.accept(Chem.MolFromSmiles("CCOc1ccc(NC(C)=O)cc1")))
            screen.save()
            resumed = screening.GenerationScreen(self.checkpoint)
            self.assertFalse(resumed.accept(Chem.MolFromSmiles("CC(=O)Nc1ccc(OCC)cc1")))
            self.assertFalse(resumed.accept(Chem.MolFromSmiles("COc1ccc(CCNC(C)=O)cc1")))
        self.assertEqual(dict(resumed.rejections), {"duplicate": 1, "scaffold_cap": 1})
