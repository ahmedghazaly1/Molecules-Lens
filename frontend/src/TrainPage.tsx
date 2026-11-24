import { useEffect, useState } from "react";
import { attachTrainingData, clearCheckpoint, fetchCheckpoints, fetchTrainStatus, requestTrain, stopTrain, uploadCheckpoint, uploadReference, type CheckpointRole, type CheckpointSlots, type TrainStatus } from "./api";

function progressLine(status: TrainStatus | null): string {
  if (!status || (status.phase === "" && status.status !== "running" && status.stage !== "done")) {
    return "No live run.";
  }
  const where = status.device?.includes("cuda") ? "GPU" : status.device ? "CPU" : "";
  const file = status.dataset ? ` on ${status.dataset}` : "";
  const place = where ? ` on the ${where}` : "";
  const count = status.total ? ` ${status.step ?? 0} of ${status.total}` : "";
  if (status.status === "running" && status.phase === "sampling") {
    return `Sampling molecules${file}${place}.${count}.`;
  }
  if (status.status === "running" && status.phase === "zinc") {
    return `Training the GENTRL base model on molecules from the MOSES dataset${place}.${count}`;
  }
  if (status.status === "running" && status.phase === "rl") {
    return `Reinforcement learning${file}${place}.${status.total ? ` Iteration${count}.` : ""}`;
  }
  if (status.status === "running") {
    return `Pretraining${file}${place}.${status.total ? ` Pass${count}.` : ""}`;
  }
  if (status.stage === "done" && status.phase === "rl") {
    return `Reinforcement learning finished${file}. Generate samples from this checkpoint.`;
  }
  if (status.stage === "done" && status.phase === "zinc") {
    return "Base model training finished. The GENTRL checkpoint was saved.";
  }
  if (status.stage === "done") {
    return `Run finished${file}.`;
  }
  if (status.stage === "stopped") {
    return `Run stopped${file}.`;
  }
  return "No live run.";
}

function Curve({ title, caption, values, color }: { title: string; caption: string; values: number[]; color: string }) {
  const width = 640;
  const height = 240;
  const left = 72;
  const right = width - 20;
  const top = 12;
  const bottom = height - 48;
  const rawMin = values.length ? Math.min(...values) : 0;
  const rawMax = values.length ? Math.max(...values) : 1;
  const min = title === "Reward" ? 0 : Math.max(0, rawMin - (rawMax - rawMin || rawMax || 1) * 0.1);
  const max = title === "Reward" ? 1 : rawMax + (rawMax - rawMin || rawMax || 1) * 0.1;
  const span = max - min || 1;
  const formatTick = (value: number) => Number(value.toFixed(2)).toString();
  const d = values
    .map((value, index) => {
      const x = left + (values.length === 1 ? (right - left) / 2 : (index / (values.length - 1)) * (right - left));
      const y = bottom - ((value - min) / span) * (bottom - top);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const xTicks = values.length > 1 ? [0, Math.floor((values.length - 1) / 2), values.length - 1] : [0];

  return (
    <section className="chart-card">
      <h3>{title}</h3>
      <p className="muted">{caption}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" aria-label={`${title} by ${title === "Reward" ? "RL iteration" : "training step"}`}>
          {[0, 0.5, 1].map(fraction => {
            const y = bottom - fraction * (bottom - top);
            return <g key={fraction}><line x1={left} x2={right} y1={y} y2={y} stroke="#e2eae5" /><text x={left - 9} y={y + 4} textAnchor="end" fill="#60756b" fontSize="12">{formatTick(min + fraction * span)}</text></g>;
          })}
          <path d={`M${left},${top}V${bottom}H${right}`} fill="none" stroke="#83988c" strokeWidth="1.5" />
          {xTicks.map(index => {
            const x = left + (values.length <= 1 ? (right - left) / 2 : (index / (values.length - 1)) * (right - left));
            return <text key={index} x={x} y={bottom + 19} textAnchor="middle" fill="#60756b" fontSize="12">{index + 1}</text>;
          })}
          <text x={(left + right) / 2} y={height - 8} textAnchor="middle" fill="#425c4f" fontSize="12">{title === "Reward" ? "RL iteration" : "Training step"}</text>
          <text transform={`translate(17 ${(top + bottom) / 2}) rotate(-90)`} textAnchor="middle" fill="#425c4f" fontSize="12">{title}</text>
          <path d={d} fill="none" stroke={color} strokeWidth="2" />
          {values.length === 1 ? <circle cx={(left + right) / 2} cy={bottom - ((values[0] - min) / span) * (bottom - top)} r="5" fill={color} /> : null}
      </svg>
      {values.length === 0 ? <p className="muted">{title === "Reward" ? "Reward points appear during reinforcement learning." : "Loss points appear during base model training and pretraining."}</p> : null}
    </section>
  );
}

export function TrainPage() {
  const [dataset, setDataset] = useState("Leshmania.csv");
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [rlIterations, setRlIterations] = useState(50);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointSlots>({});
  const [referenceMessage, setReferenceMessage] = useState("");
  const [uploading, setUploading] = useState<CheckpointRole | null>(null);

  useEffect(() => {
    fetchCheckpoints().then(setCheckpoints).catch(() => undefined);
  }, []);

  async function onCheckpoint(role: CheckpointRole, files: FileList | null, kind: "zip" | "folder") {
    if (!files?.length) return;
    setError(null);
    setUploading(role);
    setReferenceMessage("");
    try {
      const saved = await uploadCheckpoint(role, files, kind);
      setCheckpoints((current) => ({ ...current, [role]: saved.label }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not upload checkpoint");
    } finally {
      setUploading(null);
    }
  }

  useEffect(() => {
    let stop = false;
    async function pull() {
      try {
        const next = await fetchTrainStatus();
        if (!stop) {
          setStatus(next);
          if (next.stage === "done") fetchCheckpoints().then(setCheckpoints).catch(() => undefined);
          setError(null);
        }
      } catch {
        if (!stop) setError("The training API is not running.");
      }
    }
    pull();
    const timer = window.setInterval(pull, 2000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, []);

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setError(null);
    try {
      const report = await uploadReference(file);
      setDataset(report.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read that file");
    }
  }

  async function onStop() {
    setError(null);
    try {
      setStatus(await stopTrain());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not stop the run");
    }
  }

  async function onRequest(action: "zinc" | "pretrain" | "rl") {
    setError(null);
    try {
      setStatus(await requestTrain(action, dataset, { epochs, batchSize, learningRate, rlIterations }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not request the run");
    }
  }

  const busy = status?.status === "running";

  return (
    <main className="page-content">
      <div className="page-heading"><div><p className="eyebrow">02 / OPTIMIZE</p><h1>Training workspace</h1><p>Prepare a reference set, configure a run, and monitor each stage in one place.</p></div></div>
      {error || status?.error ? <p className="error banner">{error || status?.error}</p> : null}
      <div className="training-steps">
        <div className="step-card surface-card"><span className="step-number">1</span><div><strong>Dataset</strong><small>{dataset}</small></div></div>
        <div className={`step-card surface-card ${status?.phase === "pretrain" ? "active" : ""}`}><span className="step-number">2</span><div><strong>Pretrain</strong><small>Learn reference chemistry</small></div></div>
        <div className={`step-card surface-card ${status?.phase === "rl" ? "active" : ""}`}><span className="step-number">3</span><div><strong>Reinforce</strong><small>Optimize reward</small></div></div>
      </div>
      <div className="training-layout">
        <section className="surface-card panel"><div className="section-head"><h2>Run configuration</h2><span className="mini-label">{busy ? "LOCKED DURING RUN" : "READY"}</span></div>
          <p className="muted">Use a reference CSV with valid SMILES before starting a run.</p>
          <div className="dataset-row"><span className="dataset-icon">▤</span><div><strong>{dataset}</strong><small>Reference dataset · CSV</small></div><label className="secondary upload-button">Change<input type="file" accept=".csv,text/csv" disabled={busy} onChange={(event) => onUpload(event.target.files?.[0])} /></label></div>
          <div className="checkpoint-uploads">
            <h3>Upload GENTRL checkpoints</h3>
            <p className="hint">Each folder needs enc.model, dec.model, lp.model, and order.pkl. Folder names can be anything. Upload only checkpoints from a trusted source.</p>
            {([
              ["base", "Base model", "Used to start pretraining"],
              ["pretrained", "Pretrained model", "Used to start RL and generate before RL"],
              ["after_rl", "After-RL model", "Used to generate after RL"],
            ] as const).map(([role, title, purpose]) => (
              <div className="checkpoint-upload-row" key={role}>
                <div><strong>{title}</strong><small>{purpose}</small><small>Active upload: {checkpoints[role] || "Built-in checkpoint"}</small></div>
                <div className="checkpoint-upload-actions">
                  <label className="secondary upload-button">Folder<input type="file" multiple {...{ webkitdirectory: "" }} disabled={busy || uploading !== null} onChange={(event) => { void onCheckpoint(role, event.target.files, "folder"); event.target.value = ""; }} /></label>
                  <label className="secondary upload-button">ZIP<input type="file" accept=".zip,application/zip" disabled={busy || uploading !== null} onChange={(event) => { void onCheckpoint(role, event.target.files, "zip"); event.target.value = ""; }} /></label>
                  {role !== "base" && checkpoints[role] ? <label className="secondary upload-button">Training data<input type="file" accept=".csv,.smi,.txt" disabled={busy || uploading !== null} onChange={async (event) => { const file = event.target.files?.[0]; event.target.value = ""; if (!file) return; setUploading(role); setError(null); setReferenceMessage(""); try { const saved = await attachTrainingData(role, file); setReferenceMessage(`${title}: ${saved.count} unique training molecules attached.`); } catch (err) { setError(err instanceof Error ? err.message : "Could not attach training data"); } finally { setUploading(null); } }} /></label> : null}
                  {checkpoints[role] ? <button className="secondary" type="button" disabled={busy || uploading !== null} onClick={async () => { try { await clearCheckpoint(role); setCheckpoints((current) => { const next = { ...current }; delete next[role]; return next; }); } catch (err) { setError(err instanceof Error ? err.message : "Could not clear checkpoint"); } }}>Built-in</button> : null}
                </div>
              </div>
            ))}
            <p className="hint">For an uploaded target model, attach one CSV or SMILES file containing all molecules used in its pretraining and fine-tuning. Base training is excluded. Attaching a file replaces the previous reference for that checkpoint. App training saves references automatically.</p>
            {referenceMessage ? <p className="hint" role="status">{referenceMessage}</p> : null}
            {uploading ? <p className="hint">Uploading {uploading} checkpoint…</p> : null}
          </div>
          <div className="field-grid">
            <label className="slider"><span>Epochs</span><input type="number" min={1} max={50} step={1} disabled={busy} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
            <label className="slider"><span>Batch size</span><input type="number" min={1} max={64} step={1} disabled={busy} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></label>
            <label className="slider"><span>Learning rate</span><input type="number" min={0.00001} max={0.001} step={0.00001} disabled={busy} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
            <label className="slider"><span>RL iterations</span><input type="number" min={1} step={1} disabled={busy} value={rlIterations} onChange={(event) => setRlIterations(Number(event.target.value))} /></label>
          </div>
          <p className="hint">Epochs, batch size, and learning rate apply to base model training and pretraining. RL iterations apply to reinforcement learning.</p>
          <div className="training-actions"><button className="secondary" disabled={busy} onClick={() => onRequest("zinc")}>Train Base Model</button><button className="primary" disabled={busy} onClick={() => onRequest("pretrain")}>Start pretraining →</button><button className="secondary" disabled={busy} onClick={() => onRequest("rl")}>Start reinforcement learning</button></div>
          <p className="hint">Train Base Model uses zinc_plogp_300k_FinalSTEREO.csv from the GENTRL examples folder. Pretraining starts from the base checkpoint; reinforcement learning starts from the checkpoint saved by pretraining. Base model training can take many hours on CPU.</p>
        </section>
        <section className="surface-card panel"><div className="section-head"><h2>Current run</h2><span className={`status-pill ${busy ? "running" : ""}`}>{busy ? "● RUNNING" : status?.stage === "zinc" ? "BASE MODEL" : (status?.stage || "IDLE").toUpperCase()}</span></div>
          <p className="muted">{progressLine(status)}</p><p className="muted">{status?.message}</p>
          <div className={`progress-track ${busy && (status?.step ?? 0) === 0 ? "progress-pending" : ""}`} role="progressbar" aria-label="Training progress" aria-valuemin={0} aria-valuemax={status?.total || undefined} aria-valuenow={status?.total ? status.step ?? 0 : undefined}><i style={{ width: `${status?.total ? Math.min(100, ((status.step ?? 0) / status.total) * 100) : 0}%` }} /></div>
          <div className="progress-label"><span>{status?.total ? `${status.step ?? 0} of ${status.total}` : "No active progress"}</span><strong>{status?.total ? `${Math.round(((status.step ?? 0) / status.total) * 100)}%` : "—"}</strong></div>
          <div className="run-stats"><div><strong>{status?.rec?.at(-1)?.toFixed(3) ?? "—"}</strong><span>Latest reconstruction loss</span></div><div><strong>{status?.device || "—"}</strong><span>Compute device</span></div><div><strong>{status?.phase || "—"}</strong><span>Current phase</span></div></div>
          {status?.phase === "continue" || status?.phase === "rl" ? (
            <p className="checkpoint">Starting checkpoint: <span>{status.input_label || checkpoints[status.phase === "continue" ? "base" : "pretrained"] || "Built-in checkpoint"}</span>{status.input_checkpoint ? <> · {status.input_checkpoint}</> : null}</p>
          ) : null}
          <p className="checkpoint">Output checkpoint: <span>{status?.checkpoint || "None yet"}</span></p>
          <button className="secondary" disabled={!busy} onClick={onStop}>Stop run</button>
        </section>
        <section className="training-charts"><Curve title="Reconstruction loss" caption="Negative reconstruction log-probability during base training or pretraining. Lower is better." values={status?.rec ?? []} color="#149b73" /><Curve title="Reward" caption="Mean RL score combines reference structure similarity (50%) with QED, SA, and LogP similarity (50%). Higher is better; scores range from 0 to 1." values={status?.reward ?? []} color="#e6a34d" /></section>
      </div>
    </main>
  );
}
