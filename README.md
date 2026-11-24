# MoleculeLens

Interactive dashboard for generating and inspecting molecules. GENTRL is an external model.

## Run

Terminal 1 — API:

```powershell
cd d:\PROJECTS\MoleculeLens\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Terminal 2 — UI:

```powershell
cd d:\PROJECTS\MoleculeLens\frontend
npm run dev
```

Open http://localhost:5173 and click **Generate 12 molecules**.

Generation applies the RIPK1 screening rules before returning candidates, including PAINS and Brenk, the physicochemical cutoffs, SA ≤ 5, QED > 0.6, and estimated LogS from −4.0 to −1.0. Each checkpoint has a persistent accepted library in `models/accepted_libraries/` for deduplication and the cap of 20 molecules per Bemis–Murcko scaffold. To seed it from an existing library, place `RIPK1_FINAL_ACCEPTED.csv` in the project root or set `MOLECULELENS_ACCEPTED_CSV` to the CSV path before starting the API.

Training-set matches are excluded using the datasets attached to the selected checkpoint. To exclude both RIPK1 reference datasets, attach a single CSV containing the combined SMILES through the checkpoint's **Training data** control. The Generate page reports if reference coverage is partial or unavailable. The sampler keeps drawing candidates until the requested number passes screening. Large requests can take longer, especially when few candidates pass the filters.

Completed batches show how many molecules were accepted per minute of sampling and screening. This rate excludes preparing the API response. A 3D conformer is calculated when you open a candidate's **3D** view.

## Run a staging deployment

Build the UI and run the single-origin staging app from the repository root. These commands assume the backend virtual environment is installed.

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
$env:PIC50_LOG = Join-Path $PWD 'backend/pic50_worker_stderr.log'
.\backend\.venv\Scripts\python.exe -m uvicorn app.staging:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. On another Windows host, set `GENTRL_ROOT`, `GENTRL_PYTHON`, `PIC50_ROOT`, and `PIC50_PYTHON` to the installed model projects and their Python executables. Keep the `models/` directory on persistent storage and retain the GENTRL example data and checkpoints needed by training. Run one API process because training status and worker handles live in process memory.

For remote testing, put an authenticated HTTPS proxy in front of the loopback listener and limit access to trusted testers. The API accepts checkpoint files that can contain Python pickle data, and training and checkpoint changes affect the shared instance.

Run these checks using the deployed HTTPS URL.

- Open Generate, Draw, Lens, and Training. Import a SMILES string and a CSV with one invalid row. Inspect a candidate in Lens and open its 2D and 3D views.
- Generate a small batch in both Before RL and After RL. Confirm candidate cards appear, predicted pIC50 contains numeric values, and an After RL candidate has a reward explanation. `/health` reports API liveness, so also verify the model results.
- In a disposable staging copy, upload a trusted checkpoint and run a short training job. Check progress and cancellation, then confirm that a completed checkpoint is available for generation and remains selected after a restart where applicable.

## Bring your own GENTRL checkpoint

In **Training workspace → Upload GENTRL checkpoints**, choose one of three slots:

- **Base model:** used as the starting point for pretraining.
- **Pretrained model:** used as the starting point for RL and for Before RL generation.
- **After-RL model:** used for After RL generation.

Select a folder or ZIP containing `enc.model`, `dec.model`, `lp.model`, and `order.pkl` in one directory. The uploaded folder's name is only a display label; the selected slot determines how MoleculeLens uses it. Uploaded checkpoints are stored under `models/` and remain selected after an API restart. Use **Built-in** to return a slot to the checkpoint saved in the GENTRL examples directory. A completed training run becomes the built-in checkpoint for its output stage.

Checkpoint files can contain Python pickle data. Only upload models from trusted sources.
