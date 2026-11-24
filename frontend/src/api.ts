export type Molecule = {
  quality?: { valid: boolean; novel: boolean | null; chemical_pass: boolean; alerts: string[] };
  id: string;
  smiles: string;
  mw: number;
  logp: number;
  logs?: number;
  hbd: number;
  hba: number;
  lipinski: boolean;
  bbb: boolean;
  pains: boolean;
  pains_alert: string | null;
  brenk_alerts: string[];
  sa_score: number;
  sa_easy: boolean;
  qed: number;
  reward: number | null;
  pic50: number | null;
  image: string;
  molblock: string | null;
};

export type GenerateResponse = {
  quality: BatchQuality;
  sampler: string;
  count: number;
  molecules: Molecule[];
};

export type ImportError = { row: number; smiles: string; reason: string };
export type ImportResponse = { molecules: Molecule[]; errors: ImportError[] };

export async function importMolecules(text: string, format: "smiles" | "csv"): Promise<ImportResponse> {
  const res = await fetch("/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, format }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Import failed (${res.status})`);
  }
  return res.json();
}

export type ModelStage = "before_rl" | "after_rl";

export type BatchQuality = {
  screening?: {
    requested: number;
    reference_coverage: "complete" | "partial" | "unavailable";
    elapsed_seconds?: number;
    accepted_per_minute?: number | null;
  } | null;
  validity: number | null;
  attempted: number;
  valid_attempts: number;
  count: number;
  novelty: number | null;
  diversity: number | null;
  chemical_pass: number | null;
  chemical_filter: string;
  reference_count: number | null;
  datasets: string[];
  novelty_note: string;
};

export async function attachTrainingData(role: CheckpointRole, file: File): Promise<{ count: number }> {
  const res = await fetch(`/models/${role}/training-data?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || "Could not attach training dataset");
  }
  return res.json();
}

export type CheckpointRole = "base" | "pretrained" | "after_rl";
export type CheckpointSlots = Partial<Record<CheckpointRole, string>>;

async function modelResponse(res: Response): Promise<{ role: CheckpointRole; label: string }> {
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Model upload failed (${res.status})`);
  }
  return res.json();
}

export async function fetchCheckpoints(): Promise<CheckpointSlots> {
  const res = await fetch("/models");
  if (!res.ok) throw new Error(`Could not read model uploads (${res.status})`);
  return res.json();
}

export async function clearCheckpoint(role: CheckpointRole): Promise<void> {
  const res = await fetch(`/models/${role}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Could not restore built-in checkpoint (${res.status})`);
}

export async function uploadCheckpoint(role: CheckpointRole, files: FileList, kind: "zip" | "folder"): Promise<{ role: CheckpointRole; label: string }> {
  if (kind === "zip") {
    const file = files[0];
    if (!file || !file.name.toLowerCase().endsWith(".zip")) throw new Error("Choose a ZIP file");
    return modelResponse(await fetch(`/models/upload-zip?role=${role}&label=${encodeURIComponent(file.name)}`, { method: "POST", body: file }));
  }
  const required = ["enc.model", "dec.model", "lp.model", "order.pkl"];
  const chosen = Array.from(files).filter((file) => required.includes(file.name));
  if (chosen.length !== 4 || new Set(chosen.map((file) => file.name)).size !== 4) {
    throw new Error("Choose a folder containing enc.model, dec.model, lp.model, and order.pkl");
  }
  const parents = new Set(chosen.map((file) => file.webkitRelativePath.slice(0, -file.name.length)));
  if (parents.size !== 1) throw new Error("The four checkpoint files must be in the same folder");
  const label = chosen[0].webkitRelativePath.split("/").at(-2) || "Uploaded checkpoint";
  const uploadId = crypto.randomUUID().replaceAll("-", "");
  for (const file of chosen) {
    const res = await fetch(`/models/upload-file?upload_id=${uploadId}&name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    await modelResponse(res);
  }
  return modelResponse(await fetch(`/models/finish-folder?role=${role}&upload_id=${uploadId}&label=${encodeURIComponent(label)}`, { method: "POST" }));
}

export type TrainStatus = {
  status: "idle" | "running";
  stage: string;
  checkpoint: string;
  input_checkpoint?: string | null;
  input_label?: string | null;
  rec: number[];
  reward: number[];
  message: string;
  error?: string | null;
  device?: string;
  step?: number;
  total?: number;
  dataset?: string;
  phase?: string;
};

export async function fetchTrainStatus(): Promise<TrainStatus> {
  const res = await fetch("/train/status");
  if (!res.ok) throw new Error(`Train status failed (${res.status})`);
  return res.json();
}

export async function stopTrain(): Promise<TrainStatus> {
  const res = await fetch("/train/stop", { method: "POST" });
  if (!res.ok) throw new Error(`Stop failed (${res.status})`);
  return res.json();
}

export async function uploadReference(file: File): Promise<{ name: string }> {
  const text = await file.text();
  const res = await fetch(`/train/upload?name=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: text,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export type TrainSettings = {
  epochs: number;
  batchSize: number;
  learningRate: number;
  rlIterations: number;
};

export async function requestTrain(
  action: "zinc" | "pretrain" | "rl",
  dataset = "Leshmania.csv",
  settings: TrainSettings = { epochs: 3, batchSize: 64, learningRate: 0.0001, rlIterations: 50 },
): Promise<TrainStatus> {
  const res = await fetch("/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      dataset,
      epochs: settings.epochs,
      batch_size: settings.batchSize,
      learning_rate: settings.learningRate,
      rl_iterations: settings.rlIterations,
    }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Train request failed (${res.status})`);
  }
  return res.json();
}

export type AtomContribution = {
  atom: number;
  symbol?: string;
  contribution?: number;
  fragment_score?: number;
};

export type QedTerm = {
  name: string;
  label: string;
  value: number;
  desirability: number;
};

export type PropertyCheck = {
  name: string;
  value: number;
  limit: string;
  ok: boolean;
};

export type RewardShap = {
  reward: number;
  structure_similarity?: number;
  property_similarity?: number;
  baseline: number;
  inputs: { qed: number; sa: number; logp: number };
  reference_means: { qed: number; sa: number; logp: number };
  shap: { qed: number; sa: number; logp: number };
  reference: string | null;
  reference_rows: { kept: number; dropped: number } | null;
};

export type Explanation = {
  smiles: string;
  logp: number;
  sa_score: number;
  qed: number;
  logp_atoms: AtomContribution[];
  sa_atoms: AtomContribution[];
  images: { logp: string; sa: string; pains: string; brenk: string };
  qed_terms: QedTerm[];
  weakest_qed: string;
  lipinski: PropertyCheck[];
  bbb: PropertyCheck[];
  pains: boolean;
  pains_alert: string | null;
  pains_atoms: number[];
  brenk_alerts: string[];
  similarity: {
    reference_dataset: string;
    reference_smiles: string;
    dice: number;
    max_weight: number;
    weights: number[];
    image: string;
  } | null;
  similarity_error: string | null;
  reward_shap: RewardShap | null;
  reward_error: string | null;
};

export async function explainMolecule(smiles: string, includeModelReward = true): Promise<Explanation> {
  const res = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles, include_model_reward: includeModelReward }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Explain failed (${res.status})`);
  }
  return res.json();
}

export async function generateBatch(n = 12, stage: ModelStage = "after_rl"): Promise<GenerateResponse> {
  const res = await fetch(`/generate?n=${n}&stage=${stage}`, { method: "POST" });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Generate failed (${res.status})`);
  }
  return res.json();
}

export async function fetchMolblock(smiles: string, signal: AbortSignal): Promise<string | null> {
  const res = await fetch("/molecule/3d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
    signal,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `3D structure failed (${res.status})`);
  }
  const data: { molblock: string | null } = await res.json();
  return data.molblock;
}
