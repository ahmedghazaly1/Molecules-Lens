"""Apply the RIPK1 acceptance rules before a generated molecule is returned."""

import csv
import json
import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from app.checkpoints import STORE
from app.properties import _brenk_catalog, _pains_catalog, bbb_checks, checkpoint_key, estimated_logs, load_reference, sascorer


MAX_PER_SCAFFOLD = 20


def scaffold_for(mol: Chem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "ACYCLIC"


class GenerationScreen:
    def __init__(self, checkpoint: Path):
        self.library = STORE / "accepted_libraries" / f"{checkpoint_key(checkpoint)}.json"
        self.accepted = json.loads(self.library.read_text(encoding="utf-8")) if self.library.is_file() else {}
        reference = load_reference(checkpoint)
        self.reference = set(reference["smiles"]) if reference else set()
        self.reference_coverage = (
            "unavailable" if reference is None else "complete" if reference.get("complete") else "partial"
        )

        seed_path = os.environ.get("MOLECULELENS_ACCEPTED_CSV")
        seed = Path(seed_path) if seed_path else STORE.parent / "RIPK1_FINAL_ACCEPTED.csv"
        if seed_path or seed.is_file():
            with seed.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                column = next((name for name in reader.fieldnames or [] if name.strip().lower() == "smiles"), None)
                if column is None:
                    raise ValueError(f"Accepted library needs a SMILES column: {seed}")
                for row in reader:
                    mol = Chem.MolFromSmiles((row[column] or "").strip())
                    if mol is not None and mol.GetNumAtoms():
                        self.accepted.setdefault(Chem.MolToSmiles(mol), scaffold_for(mol))

        self.seen = set(self.accepted)
        self.scaffolds = Counter(self.accepted.values())
        self.rejections = Counter()
        self.new_count = 0

    def accept(self, mol: Chem.Mol | None) -> bool:
        reason = self._rejection_reason(mol)
        if reason is not None:
            self.rejections[reason] += 1
            return False
        return True

    def _rejection_reason(self, mol: Chem.Mol | None) -> str | None:
        if mol is None or not mol.GetNumAtoms():
            return "invalid"
        canonical = Chem.MolToSmiles(mol)
        if canonical in self.seen:
            return "duplicate"
        self.seen.add(canonical)
        if canonical in self.reference:
            return "training_match"
        if _pains_catalog.HasMatch(mol) or _brenk_catalog.HasMatch(mol):
            return "structural_alert"

        mw = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        checks = bbb_checks(
            mw, Descriptors.TPSA(mol), Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), rotatable_bonds, logp
        )
        if not all(check["ok"] for check in checks):
            return "physicochemical"
        if sascorer.calculateScore(mol) > 5.0 or QED.qed(mol) <= 0.6:
            return "sa_or_qed"
        if not -4.0 <= estimated_logs(mol, mw, logp, rotatable_bonds) <= -1.0:
            return "logs"

        scaffold = scaffold_for(mol)
        if self.scaffolds[scaffold] >= MAX_PER_SCAFFOLD:
            return "scaffold_cap"
        self.scaffolds[scaffold] += 1
        self.accepted[canonical] = scaffold
        self.new_count += 1
        return None

    def save(self) -> None:
        if not self.new_count:
            return
        self.library.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.library.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(self.accepted), encoding="utf-8")
            temporary.replace(self.library)
        finally:
            temporary.unlink(missing_ok=True)
