import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import properties as quality


class QualityTests(unittest.TestCase):
    def test_validity_uses_attempts_not_returned_candidates(self):
        cards = [{"smiles": "CCO"}, {"smiles": "OCC"}]
        result = quality.evaluate(cards, {"attempted": 10, "valid": 7}, None)
        self.assertEqual(result["validity"], 0.7)
        self.assertEqual(result["diversity"], 0)
        self.assertIsNone(result["novelty"])
        self.assertIsNone(cards[0]["quality"]["novel"])

    def test_novelty_matches_canonical_structures(self):
        cards = [{"smiles": "OCC"}, {"smiles": "CCN"}]
        result = quality.evaluate(cards, None, {"smiles": ["CCO"], "complete": True})
        self.assertEqual(result["novelty"], 0.5)
        self.assertFalse(cards[0]["quality"]["novel"])
        self.assertTrue(cards[1]["quality"]["novel"])
        self.assertGreater(result["diversity"], 0)
        self.assertIsNone(result["validity"])

    def test_empty_and_singleton_batches(self):
        result = quality.evaluate([], None, None)
        self.assertIsNone(result["chemical_pass"])
        self.assertIsNone(result["diversity"])
        result = quality.evaluate([{"smiles": "CCO"}], None, None)
        self.assertIsNone(result["diversity"])

    def test_pains_alert_and_pass(self):
        cards = [{"smiles": "CCO"}, {"smiles": "Oc1ccccc1O"}]
        result = quality.evaluate(cards, None, None)
        self.assertEqual(result["chemical_pass"], 0.5)
        self.assertTrue(cards[1]["quality"]["alerts"])

    def test_reference_bound_to_contents_and_inherits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in quality.REQUIRED:
                (checkpoint / name).write_bytes(b"model")
            with patch.object(quality, "REFERENCES", root / "references"):
                quality.save_reference(checkpoint, ["CCO", "OCC"], "pretrain.csv")
                first = quality.load_reference(checkpoint)
                self.assertEqual(first["smiles"], ["CCO"])
                quality.save_reference(checkpoint, ["CCN"], "fine-tune.csv", first)
                self.assertEqual(len(quality.load_reference(checkpoint)["smiles"]), 2)
                (checkpoint / "enc.model").write_bytes(b"different model")
                self.assertIsNone(quality.load_reference(checkpoint))

    def test_incomplete_lineage_does_not_claim_novelty(self):
        result = quality.evaluate([{"smiles": "CCO"}], None, {"smiles": ["CCN"], "complete": False})
        self.assertIsNone(result["novelty"])

    def test_parse_csv_and_smiles(self):
        self.assertEqual(quality.parse_dataset("id,SMILES\n1,CCO\n2,CCN\n"), ["CCO", "CCN"])
        self.assertEqual(quality.parse_dataset("CCO ethanol\nCCN ethylamine"), ["CCO", "CCN"])


if __name__ == "__main__":
    unittest.main()
