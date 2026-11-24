from fastapi import FastAPI, HTTPException, Query, Request
import csv
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.generator import sampler
from app.pic50 import scorer
from app.properties import describe, explain, molblock_3d, molecule_from_smiles
from app.properties import evaluate, load_reference, parse_dataset, save_reference as save_quality_reference
from app.training import request_run, save_reference, snapshot, stop_run
from app.checkpoints import MAX_TOTAL, clear, finish_folder, list_checkpoints, upload_folder_file, upload_zip
from app.checkpoints import selected

app = FastAPI(title="MoleculeLens", version="0.1.0")


async def _receive_file(request: Request, limit: int = MAX_TOTAL) -> Path:
    temporary = tempfile.NamedTemporaryFile(delete=False)
    path = Path(temporary.name)
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise ValueError("Upload exceeds the size limit")
            temporary.write(chunk)
        temporary.close()
        if not size:
            raise ValueError("Upload is empty")
        return path
    except Exception:
        temporary.close()
        path.unlink(missing_ok=True)
        raise


@app.get("/models")
def models():
    return list_checkpoints()


@app.post("/models/upload-zip")
async def model_upload_zip(request: Request, role: str, label: str = "Uploaded checkpoint"):
    try:
        path = await _receive_file(request)
        try:
            return upload_zip(role, path, label)
        finally:
            path.unlink(missing_ok=True)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/models/upload-file")
async def model_upload_file(request: Request, upload_id: str, name: str):
    try:
        path = await _receive_file(request, 100 * 1024 * 1024)
        try:
            upload_folder_file(upload_id, name, path)
        finally:
            path.unlink(missing_ok=True)
        return {"ok": True}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/models/finish-folder")
def model_finish_folder(role: str, upload_id: str, label: str):
    try:
        return finish_folder(role, upload_id, label)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/models/{role}")
def model_clear(role: str):
    try:
        clear(role)
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/models/{role}/training-data")
async def attach_training_data(role: str, request: Request, name: str = "Training dataset"):
    if role not in {"pretrained", "after_rl"}:
        raise HTTPException(status_code=400, detail="Choose a pretrained or fine-tuned checkpoint")
    checkpoint = selected(role)
    if checkpoint is None:
        raise HTTPException(status_code=400, detail="Upload this checkpoint first")
    path = await _receive_file(request, 20 * 1024 * 1024)
    try:
        return save_quality_reference(checkpoint, parse_dataset(path.read_text(encoding="utf-8-sig")), name)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    source = getattr(sampler, "source", "unknown")
    return {"ok": True, "sampler": source}


@app.post("/generate")
def generate(n: int = Query(default=12, ge=1), stage: Literal["before_rl", "after_rl"] = "after_rl"):
    try:
        sampled = sampler.generate(n, stage)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    molecules = []
    for i, item in enumerate(sampled):
        card = describe(item["smiles"], i, reward=item.get("reward"))
        if card:
            molecules.append(card)
    if len(molecules) != n:
        raise HTTPException(status_code=502, detail=f"Generation produced {len(molecules)} of {n} requested candidates.")
    pic50_values: list[float | None] = [None] * len(molecules)
    if molecules:
        try:
            pic50_values = scorer.score([card["smiles"] for card in molecules])
        except Exception as exc:
            print(f"pIC50 scoring failed ({exc!r})", flush=True)
    for card, pic50 in zip(molecules, pic50_values):
        card["pic50"] = pic50
    checkpoint = getattr(sampled, "checkpoint", None)
    reference = load_reference(Path(checkpoint)) if checkpoint else None
    quality = evaluate(molecules, getattr(sampled, "attempts", None), reference)
    quality["screening"] = getattr(sampled, "screening", None)
    return {
        "sampler": getattr(sampler, "source", "unknown"),
        "count": len(molecules),
        "molecules": molecules,
        "quality": quality,
    }


class ImportRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    format: Literal["smiles", "csv"]


@app.post("/import")
def import_molecules(body: ImportRequest):
    if body.format == "csv":
        try:
            reader = csv.DictReader(io.StringIO(body.text.lstrip("\ufeff"), newline=""))
            column = next((name for name in (reader.fieldnames or []) if name and name.strip().lower() == "smiles"), None)
            if column is None:
                raise ValueError("CSV needs a SMILES column")
            entries = [(number, (row.get(column) or "").strip()) for number, row in enumerate(reader, start=2)]
        except csv.Error as exc:
            raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        entries = [(number, line.strip()) for number, line in enumerate(body.text.splitlines(), start=1) if line.strip()]
    if not entries:
        raise HTTPException(status_code=400, detail="No molecules found")
    if len(entries) > 200:
        raise HTTPException(status_code=400, detail="Import at most 200 molecules at a time")
    molecules, errors = [], []
    for number, smiles in entries:
        try:
            card = describe(smiles, len(molecules) + 1) if smiles else None
        except (ValueError, RuntimeError):
            card = None
        if card is None:
            errors.append({"row": number, "smiles": smiles, "reason": "Invalid or empty SMILES"})
        else:
            card["id"] = f"import-{len(molecules) + 1}"
            molecules.append(card)
    if molecules:
        try:
            values = scorer.score([card["smiles"] for card in molecules])
        except Exception as exc:
            print(f"pIC50 scoring failed ({exc!r})", flush=True)
            values = [None] * len(molecules)
        for card, value in zip(molecules, values):
            card["pic50"] = value
    return {"molecules": molecules, "errors": errors}


class MoleculeRequest(BaseModel):
    smiles: str


@app.post("/molecule/3d")
def molecule_3d(body: MoleculeRequest):
    mol = molecule_from_smiles(body.smiles)
    if mol is None or not mol.GetNumAtoms():
        raise HTTPException(status_code=400, detail="Could not parse SMILES")
    return {"molblock": molblock_3d(mol)}


class ExplainRequest(MoleculeRequest):
    include_model_reward: bool = True


@app.post("/explain")
def explain_molecule(body: ExplainRequest):
    card = explain(body.smiles)
    if card is None:
        raise HTTPException(status_code=400, detail="Could not parse SMILES")
    if not body.include_model_reward:
        card["reward_shap"] = None
        card["reward_error"] = None
        card["similarity"] = None
        card["similarity_error"] = None
        return card
    try:
        card["reward_shap"] = sampler.explain_reward(body.smiles)
        card["reward_error"] = None
    except Exception as exc:
        card["reward_shap"] = None
        card["reward_error"] = str(exc)
    # A legacy reward has no saved molecule snapshot. Never imply that a
    # different reward state or a mutable CSV is its reference.
    reward = card["reward_shap"]
    similarity = card["similarity"]
    if reward is None or similarity is None or reward.get("reference") != similarity["reference_dataset"] or reward.get("structure_similarity") is None:
        card["similarity"] = None
        card["similarity_error"] = "No saved reference molecules are available for the active reward."
    else:
        card["similarity_error"] = None
    return card


class TrainRequest(BaseModel):
    action: str = Field(pattern="^(zinc|pretrain|rl)$")
    dataset: str = "Leshmania.csv"
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=64, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, ge=1e-5, le=1e-3)
    rl_iterations: int = Field(default=50, ge=1)


@app.get("/train/status")
def train_status():
    return snapshot()


@app.post("/train/stop")
def train_stop():
    return stop_run()


@app.post("/train/upload")
async def train_upload(request: Request, name: str = Query(default="reference.csv")):
    text = (await request.body()).decode("utf-8", errors="replace")
    try:
        return save_reference(name, text)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/start")
def train_start(body: TrainRequest):
    try:
        return request_run(
            body.action,
            body.dataset,
            epochs=body.epochs,
            batch_size=body.batch_size,
            learning_rate=body.learning_rate,
            rl_iterations=body.rl_iterations,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
