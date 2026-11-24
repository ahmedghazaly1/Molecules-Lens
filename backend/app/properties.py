from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from io import BytesIO

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from app.checkpoints import STORE, REQUIRED
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

_sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _sa_path not in sys.path:
    sys.path.append(_sa_path)
import sascorer  # noqa: E402

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_pains_catalog = FilterCatalog(_pains_params)

_brenk_params = FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_brenk_catalog = FilterCatalog(_brenk_params)

# Ertl & Schuffenhauer: 1 = easy to synthesise, 10 = hard. 6 is a common cutoff.
SA_EASY_MAX = 6.0


def lipinski_pass(mw: float, logp: float, hbd: int, hba: int) -> bool:
    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10


def bbb_checks(mw: float, tpsa: float, hbd: int, hba: int, rotatable_bonds: int, logp: float) -> list[dict]:
    return [
        {"name": "MW", "value": mw, "limit": "< 450 Da", "ok": mw < 450},
        {"name": "TPSA", "value": tpsa, "limit": "< 90 Å²", "ok": tpsa < 90},
        {"name": "HBD", "value": hbd, "limit": "< 3", "ok": hbd < 3},
        {"name": "HBA", "value": hba, "limit": "≤ 7", "ok": hba <= 7},
        {"name": "Rotatable bonds", "value": rotatable_bonds, "limit": "≤ 8", "ok": rotatable_bonds <= 8},
        {"name": "LogP", "value": logp, "limit": "1–4 inclusive", "ok": 1 <= logp <= 4},
    ]


def estimated_logs(mol: Chem.Mol, exact_mw: float, logp: float, rotatable_bonds: int) -> float:
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    aromatic_proportion = aromatic_atoms / max(1, mol.GetNumHeavyAtoms())
    return 0.16 - 0.63 * logp - 0.0062 * exact_mw + 0.066 * rotatable_bonds - 0.74 * aromatic_proportion


def molecule_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol


def png_base64(
    mol: Chem.Mol,
    size: tuple[int, int] = (280, 200),
    highlight: dict[int, tuple[float, float, float]] | None = None,
) -> str:
    import base64

    from rdkit.Chem.Draw import rdMolDraw2D

    drawn = Chem.Mol(mol)
    if drawn.GetNumConformers() == 0:
        AllChem.Compute2DCoords(drawn)
    if not highlight:
        img = Draw.MolToImage(drawn, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = 0.12
    options.fillHighlights = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        drawn,
        highlightAtoms=list(highlight),
        highlightAtomColors=highlight,
    )
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText()).decode("ascii")


def _logp_color(value: float, scale: float) -> tuple[float, float, float] | None:
    if scale <= 0 or abs(value) < 0.02:
        return None
    t = min(1.0, abs(value) / scale)
    strength = 0.45 + 0.55 * t
    if value > 0:
        return (1.0 * strength, 0.42 * strength, 0.05 * strength)
    return (0.12 * strength, 0.38 * strength, 0.95 * strength)


def _sa_color(penalty: float, scale: float) -> tuple[float, float, float] | None:
    if scale <= 0 or abs(penalty) < 0.05:
        return None
    t = min(1.0, abs(penalty) / scale)
    strength = 0.45 + 0.55 * t
    if penalty > 0:
        return (0.95 * strength, 0.35 * strength, 0.08 * strength)
    return (0.1 * strength, 0.55 * strength, 0.85 * strength)


def _logp_atoms(mol: Chem.Mol) -> list[dict]:
    contribs = Crippen._GetAtomContribs(mol)
    values = [float(pair[0]) for pair in contribs]
    return [
        {
            "atom": i,
            "symbol": mol.GetAtomWithIdx(i).GetSymbol(),
            "contribution": round(value, 3),
        }
        for i, value in enumerate(values)
    ]


def _sa_atoms(mol: Chem.Mol) -> list[dict]:
    from rdkit.Chem import rdFingerprintGenerator

    if sascorer._fscores is None:
        sascorer.readFragmentScores()
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    sascorer.mfpgen.GetSparseCountFingerprint(mol, additionalOutput=output)
    per_atom: dict[int, list[float]] = {atom.GetIdx(): [] for atom in mol.GetAtoms()}
    for bit, hits in output.GetBitInfoMap().items():
        score = float(sascorer._fscores.get(bit, -4.0))
        for atom_idx, _radius in hits:
            per_atom[int(atom_idx)].append(score)
    rows = []
    for atom_idx, scores in per_atom.items():
        mean = sum(scores) / len(scores) if scores else 0.0
        rows.append({"atom": atom_idx, "fragment_score": round(mean, 3)})
    return rows


def _pains_atoms(mol: Chem.Mol) -> tuple[str | None, list[int]]:
    match = _pains_catalog.GetFirstMatch(mol)
    if match is None:
        return None, []
    atoms: list[int] = []
    for sub in match.GetFilterMatches(mol):
        atoms.extend(int(pair.target) for pair in sub.atomPairs)
    desc = match.GetDescription() or "PAINS"
    return desc, sorted(set(atoms))


_QED_LABELS = {
    "MW": "Molecular weight",
    "ALOGP": "LogP",
    "HBA": "Hydrogen-bond acceptors",
    "HBD": "Hydrogen-bond donors",
    "PSA": "Polar surface area",
    "ROTB": "Rotatable bonds",
    "AROM": "Aromatic rings",
    "ALERTS": "Structural alerts",
}


def explain(smiles: str) -> dict | None:
    mol = molecule_from_smiles(smiles)
    if mol is None:
        return None
    logp_atoms = _logp_atoms(mol)
    sa_atoms = _sa_atoms(mol)
    logp_scale = max((abs(row["contribution"]) for row in logp_atoms), default=0.0) or 1.0
    sa_values = [row["fragment_score"] for row in sa_atoms]
    sa_mean = sum(sa_values) / len(sa_values) if sa_values else 0.0
    sa_scale = max((abs(value - sa_mean) for value in sa_values), default=0.0) or 1.0

    logp_colors = {}
    for row in logp_atoms:
        color = _logp_color(row["contribution"], logp_scale)
        if color is not None:
            logp_colors[row["atom"]] = color
    sa_colors = {}
    for row in sa_atoms:
        color = _sa_color(-(row["fragment_score"] - sa_mean), sa_scale)
        if color is not None:
            sa_colors[row["atom"]] = color
    alert, pains_atoms = _pains_atoms(mol)
    pains_colors = {idx: (0.95, 0.55, 0.55) for idx in pains_atoms}
    brenk_matches = _brenk_catalog.GetMatches(mol)
    brenk_colors = {
        int(pair.target): (0.95, 0.55, 0.55)
        for entry in brenk_matches
        for match in entry.GetFilterMatches(mol)
        for pair in match.atomPairs
    }

    props = QED.properties(mol)
    terms = []
    for name, raw in props._asdict().items():
        desirability = float(QED.ads(raw, QED.adsParameters[name]))
        terms.append(
            {
                "name": name,
                "label": _QED_LABELS[name],
                "value": round(float(raw), 3),
                "desirability": round(desirability, 3),
            }
        )
    weakest = min(terms, key=lambda term: term["desirability"])

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    checks = [
        {"name": "MW", "value": round(mw, 2), "limit": "≤ 500", "ok": mw <= 500},
        {"name": "LogP", "value": round(logp, 2), "limit": "≤ 5", "ok": logp <= 5},
        {"name": "HBD", "value": hbd, "limit": "≤ 5", "ok": hbd <= 5},
        {"name": "HBA", "value": hba, "limit": "≤ 10", "ok": hba <= 10},
    ]
    return {
        "smiles": smiles,
        "logp": round(logp, 2),
        "sa_score": round(float(sascorer.calculateScore(mol)), 2),
        "qed": round(float(QED.qed(mol)), 3),
        "logp_atoms": logp_atoms,
        "sa_atoms": sa_atoms,
        "images": {
            "logp": png_base64(mol, size=(520, 340), highlight=logp_colors),
            "sa": png_base64(mol, size=(520, 340), highlight=sa_colors),
            "pains": png_base64(mol, size=(520, 340), highlight=pains_colors or None),
            "brenk": png_base64(mol, size=(520, 340), highlight=brenk_colors or None),
        },
        "qed_terms": terms,
        "weakest_qed": weakest["name"],
        "lipinski": checks,
        "bbb": bbb_checks(Descriptors.ExactMolWt(mol), Descriptors.TPSA(mol), hbd, hba, Lipinski.NumRotatableBonds(mol), logp),
        "pains": alert is not None,
        "pains_alert": alert,
        "pains_atoms": pains_atoms,
        "brenk_alerts": [entry.GetDescription() for entry in brenk_matches],
        "similarity": similarity_map(mol),
    }


_REFERENCE_FPS: tuple[bytes, list[tuple[str, object]]] | None = None


def _reference_fps() -> tuple[str, list[tuple[str, object]]]:
    global _REFERENCE_FPS
    import json

    from rdkit.Chem import rdMolDescriptors
    from app.generator import GENTRL_ROOT

    path = GENTRL_ROOT / "examples" / "rl_reward_state.json"
    try:
        snapshot = path.read_bytes()
        payload = json.loads(snapshot)
        if payload.get("version") != 1 or not isinstance(payload.get("reference_smiles"), list):
            return "", []
        dataset = str(payload["dataset"])
    except (OSError, ValueError, KeyError, TypeError):
        return "", []
    if _REFERENCE_FPS is not None and _REFERENCE_FPS[0] == snapshot:
        return dataset, _REFERENCE_FPS[1]
    rows: list[tuple[str, object]] = []
    for smiles in payload["reference_smiles"]:
        if not isinstance(smiles, str):
            continue
        mol = molecule_from_smiles(smiles)
        if mol is not None:
            rows.append((smiles, rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
    _REFERENCE_FPS = snapshot, rows
    return dataset, rows


def _morgan(mol: Chem.Mol, atom_id: int):
    from rdkit.Chem.Draw import SimilarityMaps

    return SimilarityMaps.GetMorganFingerprint(mol, atomId=atom_id, radius=2, fpType="bv", nBits=2048)


def similarity_map(mol: Chem.Mol) -> dict | None:
    """Riniker–Landrum map: Dice similarity to the nearest saved reward reference.

    Weights are s_orig - s_mod after removing the Morgan bits that include each atom.
    Green supports the similarity. Pink works against it.
    """
    import base64

    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

    if mol.GetNumAtoms() < 2:
        return None
    probe_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    best_smiles = None
    best_score = -1.0
    dataset, references = _reference_fps()
    for smiles, fp in references:
        score = float(DataStructs.DiceSimilarity(probe_fp, fp))
        if score > best_score:
            best_score = score
            best_smiles = smiles
    if best_smiles is None:
        return None
    reference = molecule_from_smiles(best_smiles)
    if reference is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(520, 340)
    drawn, max_weight = SimilarityMaps.GetSimilarityMapForFingerprint(
        reference,
        mol,
        _morgan,
        drawer,
        metric=DataStructs.DiceSimilarity,
        colorMap=[(0.85, 0.35, 0.55), (0.95, 0.95, 0.95), (0.25, 0.65, 0.35)],
    )
    drawn.FinishDrawing()
    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(
        reference, mol, _morgan, metric=DataStructs.DiceSimilarity
    )
    return {
        "reference_dataset": dataset,
        "reference_smiles": best_smiles,
        "dice": round(best_score, 3),
        "max_weight": round(float(max_weight), 4),
        "weights": [round(float(weight), 4) for weight in weights],
        "image": base64.b64encode(drawn.GetDrawingText()).decode("ascii"),
    }


def molblock_3d(mol: Chem.Mol) -> str | None:
    posed = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(posed, params) != 0:
        if AllChem.EmbedMolecule(posed, useRandomCoords=True) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(posed, maxIters=200)
    except ValueError:
        pass
    return Chem.MolToMolBlock(posed)


def pains_alert(mol: Chem.Mol) -> str | None:
    match = _pains_catalog.GetFirstMatch(mol)
    if match is None:
        return None
    desc = match.GetDescription()
    return desc or "PAINS"


def describe(smiles: str, index: int, reward: float | None = None) -> dict | None:
    mol = molecule_from_smiles(smiles)
    if mol is None:
        return None
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    sa = float(sascorer.calculateScore(mol))
    qed = float(QED.qed(mol))
    alert = pains_alert(mol)
    exact_mw = Descriptors.ExactMolWt(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    checks = bbb_checks(exact_mw, Descriptors.TPSA(mol), hbd, hba, rotatable_bonds, logp)
    return {
        "id": f"mol-{index}",
        "smiles": smiles,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "logs": round(estimated_logs(mol, exact_mw, logp, rotatable_bonds), 2),
        "hbd": hbd,
        "hba": hba,
        "lipinski": lipinski_pass(mw, logp, hbd, hba),
        "bbb": all(check["ok"] for check in checks),
        "pains": alert is not None,
        "pains_alert": alert,
        "sa_score": round(sa, 2),
        "brenk_alerts": [entry.GetDescription() for entry in _brenk_catalog.GetMatches(mol)],
        "sa_easy": sa <= SA_EASY_MAX,
        "qed": round(qed, 3),
        "reward": None if reward is None else round(float(reward), 3),
        "image": png_base64(mol),
        "molblock": None,
    }


# Batch quality metrics and checkpoint-bound target-training references.
REFERENCES = STORE / "training_references"


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None and mol.GetNumAtoms() else None


def checkpoint_key(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path / name for name in REQUIRED) if path.is_dir() else [path]:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_reference(path: Path) -> dict | None:
    try:
        file = REFERENCES / (checkpoint_key(path) + ".json")
        return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None
    except (OSError, ValueError):
        return None


def save_reference(path: Path, smiles: list[str], label: str, inherited: dict | None = None, complete: bool = True) -> dict:
    values = {value for s in smiles if (value := canonical(s))}
    if not values:
        raise ValueError("Dataset contains no valid, nonempty molecules")
    values.update((inherited or {}).get("smiles", []))
    stages = list((inherited or {}).get("datasets", []))
    stages.append(label)
    data = {"smiles": sorted(values), "datasets": stages, "complete": complete,
            "scope": "pretraining and fine-tuning; base training excluded"}
    REFERENCES.mkdir(parents=True, exist_ok=True)
    target = REFERENCES / (checkpoint_key(path) + ".json")
    import uuid
    temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(target)
    return {"count": len(values), "datasets": stages}


def parse_dataset(text: str) -> list[str]:
    stream = io.StringIO(text.lstrip("\ufeff"))
    rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("Dataset is empty")
    header = [s.strip().lower() for s in rows[0]]
    if "smiles" in header:
        index = header.index("smiles")
        return [row[index].strip() for row in rows[1:] if len(row) > index]
    return [row[0].split()[0] for row in rows if row and row[0].strip()]


def evaluate(molecules: list[dict], attempts: dict | None, reference: dict | None) -> dict:
    known = set(reference["smiles"]) if reference and reference.get("complete") else None
    fingerprints = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for card in molecules:
        mol = Chem.MolFromSmiles(card["smiles"])
        alert = pains_alert(mol)
        card["quality"] = {"valid": True, "novel": canonical(card["smiles"]) not in known if known is not None else None,
                           "chemical_pass": alert is None, "alerts": [alert] if alert else []}
        fingerprints.append(generator.GetFingerprint(mol))
    count = len(molecules)
    # Average over distinct sample positions; repeated molecules reduce diversity.
    pairs = count * (count - 1) // 2
    similarity = sum(sum(DataStructs.BulkTanimotoSimilarity(fp, fingerprints[:i])) for i, fp in enumerate(fingerprints) if i)
    attempted = attempts.get("attempted", 0) if attempts else 0
    valid = attempts.get("valid", 0) if attempts else 0
    return {"validity": valid / attempted if attempted else None,
            "attempted": attempted, "valid_attempts": valid, "count": count,
            "novelty": sum(c["quality"]["novel"] for c in molecules) / count if known is not None and count else None,
            "diversity": 1 - similarity / pairs if pairs else None,
            "chemical_pass": sum(c["quality"]["chemical_pass"] for c in molecules) / count if count else None,
            "chemical_filter": "RDKit PAINS", "reference_count": len(known) if known is not None else None,
            "datasets": (reference or {}).get("datasets", []),
            "novelty_note": "Compared with pretraining and fine-tuning datasets; base training excluded." if known is not None else "Complete pretraining and fine-tuning dataset unavailable."}
