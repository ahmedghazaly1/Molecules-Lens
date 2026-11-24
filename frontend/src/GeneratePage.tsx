import { useEffect, useMemo, useState } from "react";
import { fetchMolblock, type BatchQuality, type ImportError, type ModelStage, type Molecule } from "./api";
import { Molecule3D } from "./Molecule3D";

const RESULTS_PER_PAGE = 24;

function MoleculeStructure({ molecule }: { molecule: Molecule }) {
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [result, setResult] = useState<{ molblock: string | null; error?: string } | null>(null);
  const molblock = molecule.molblock ?? result?.molblock;

  useEffect(() => {
    if (view !== "3d" || molblock || result) return;
    const controller = new AbortController();
    fetchMolblock(molecule.smiles, controller.signal).then((value) => {
      if (!controller.signal.aborted) setResult({ molblock: value });
    }).catch((err: unknown) => {
      if (!controller.signal.aborted) {
        setResult({ molblock: null, error: err instanceof Error ? err.message : "Could not load the 3D structure" });
      }
    });
    return () => controller.abort();
  }, [view, molecule.smiles, molblock, result]);

  return <>
    <div className="structure">
      {view === "3d" && molblock ? <Molecule3D molblock={molblock} /> : <img src={`data:image/png;base64,${molecule.image}`} alt="" />}
      <div className="view-toggle">
        <button type="button" className={view === "2d" ? "active" : ""} onClick={() => setView("2d")}>2D</button>
        <button type="button" className={view === "3d" ? "active" : ""} onClick={() => {
          if (result?.error) setResult(null);
          setView("3d");
        }}>3D</button>
      </div>
    </div>
    {view === "3d" && !molblock ? <p className="muted" role="status">
      {result ? result.error ?? "No 3D conformer for this SMILES." : "Preparing 3D structure…"}
    </p> : null}
  </>;
}

const SCORE_NOTES: Record<string, string> = {
  MW: "Molecular weight in g/mol. Lipinski’s rule prefers 500 or less.",
  LogP: "Calculated fat solubility (octanol–water). Higher means more lipophilic. Lipinski prefers 5 or less. This is not a lab measurement.",
  LogS: "Estimated aqueous solubility. Generation accepts LogS values from −4.0 to −1.0.",
  HBD: "Hydrogen-bond donors. Lipinski prefers 5 or fewer.",
  HBA: "Hydrogen-bond acceptors. Lipinski prefers 10 or fewer.",
  SAscore: "How hard the structure is to synthesize. 1 is easy, 10 is hard. The badge calls 6 or below easy.",
  QED: "How drug-like the structure looks, from 0 to 1. Closer to 1 is more drug-like. This is not proof of activity.",
  Reward: "Score against the active reference set. New RL runs combine structural similarity (50%) with QED, synthetic accessibility, and LogP similarity (50%). It is not measured activity.",
  pIC50: "Predicted binding strength from the MoleculeX model. Higher means stronger predicted activity. This is not a lab measurement.",
};

function ScoreLabel({ name }: { name: string }) {
  const [open, setOpen] = useState(false);
  return (
    <dt>
      {name}
      <button
        type="button"
        className="info"
        aria-expanded={open}
        aria-label={`What ${name} means`}
        onClick={() => setOpen((value) => !value)}
      >
        i
      </button>
      {open ? <span className="note">{SCORE_NOTES[name]}</span> : null}
    </dt>
  );
}

function formatPic50(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(2);
}

function formatReward(reward: number | null): string {
  if (reward == null || Number.isNaN(reward)) return "—";
  if (Math.abs(reward) >= 1000) return reward.toExponential(1);
  return reward.toFixed(2);
}

type Props = {
  quality: BatchQuality | null;
  molecules: Molecule[];
  sampler: string;
  importErrors: ImportError[];
  modelStage: ModelStage;
  onModelStageChange: (stage: ModelStage) => void;
  loading: boolean;
  error: string | null;
  selected: Molecule | null;
  detailOpen: boolean;
  onGenerate: (n: number) => void;
  onImport: (text: string, format: "smiles" | "csv") => Promise<Molecule[]>;
  onEditInDraw: () => void;
  onSelect: (id: string) => void;
  onCloseDetail: () => void;
  onOpenLens: () => void;
};

export function GeneratePage({
  quality,
  molecules,
  sampler,
  importErrors,
  modelStage,
  onModelStageChange,
  loading,
  error,
  selected,
  detailOpen,
  onGenerate,
  onImport,
  onEditInDraw,
  onSelect,
  onCloseDetail,
  onOpenLens,
}: Props) {
  const [lipinskiOnly, setLipinskiOnly] = useState(false);
  const [bbbOnly, setBbbOnly] = useState(false);
  const [noPains, setNoPains] = useState(false);
  const [noBrenk, setNoBrenk] = useState(false);
  const [saMax, setSaMax] = useState(10);
  const [qedMin, setQedMin] = useState(0);
  const [batchSize, setBatchSize] = useState(12);
  const [resultView, setResultView] = useState<"grid" | "table">("grid");
  const [resultPage, setResultPage] = useState(1);
  const [smilesInput, setSmilesInput] = useState("");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [importInputError, setImportInputError] = useState<string | null>(null);
  const screening = quality?.screening;
  const brenkAlerts = selected?.brenk_alerts ?? [];
  const brenkCount = brenkAlerts.length;
  const brenkBadge = (
    <span className={brenkCount ? "badge warn" : "badge pass"}>
      Brenk {brenkCount} {brenkCount === 1 ? "alert" : "alerts"}
    </span>
  );

  async function submitImport() {
    setImportInputError(null);
    try {
      if (csvFile) {
        if (csvFile.size > 2_000_000) throw new Error("CSV must be 2 MB or smaller");
        await onImport(await csvFile.text(), "csv");
      } else if (smilesInput.trim()) {
        await onImport(smilesInput.trim(), "smiles");
      } else {
        throw new Error("Paste a SMILES string or choose a CSV file");
      }
    } catch (err) {
      setImportInputError(err instanceof Error ? err.message : "Could not read file");
    }
  }

  const visible = useMemo(
    () =>
      molecules.filter((mol) => {
        if (lipinskiOnly && !mol.lipinski) return false;
        if (bbbOnly && !mol.bbb) return false;
        if (noPains && mol.pains) return false;
        if (noBrenk && mol.brenk_alerts.length) return false;
        if (mol.sa_score > saMax) return false;
        if (Math.round(mol.qed * 100) / 100 < qedMin) return false;
        return true;
      }),
    [molecules, lipinskiOnly, bbbOnly, noPains, noBrenk, saMax, qedMin],
  );
  const pageCount = Math.max(1, Math.ceil(visible.length / RESULTS_PER_PAGE));
  const page = Math.min(resultPage, pageCount);
  const pageStart = (page - 1) * RESULTS_PER_PAGE;
  const pageMolecules = visible.slice(pageStart, pageStart + RESULTS_PER_PAGE);

  useEffect(() => setResultPage(1), [visible]);

  useEffect(() => {
    if (!detailOpen) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onCloseDetail();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detailOpen, onCloseDetail]);

  return (
    <main className="page-content">
      <div className="page-heading">
        <div><p className="eyebrow">01 / DISCOVER</p><h1>Generate molecules</h1><p>Explore candidate structures and narrow the batch with medicinal chemistry filters.</p></div>
        <button className="primary" onClick={() => onGenerate(batchSize)} disabled={loading}>{loading ? "Generating…" : `◈  Generate ${batchSize} molecules`}</button>
      </div>
      <section className="import-panel surface-card" aria-label="Import molecules">
        <div className="section-head"><h2>Import molecules</h2><span className="mini-label">YOUR STRUCTURES</span></div>
        <p className="muted">Paste one SMILES string, or choose a CSV file with a SMILES column. Imported molecules use the same properties and Lens explanations.</p>
        <div className="import-controls">
          <label>SMILES<input value={smilesInput} onChange={(e) => { setSmilesInput(e.target.value); setCsvFile(null); }} placeholder="e.g. CCO" disabled={loading} /></label>
          <label>CSV file<input type="file" accept=".csv,text/csv" onChange={(e) => { setCsvFile(e.target.files?.[0] ?? null); setSmilesInput(""); }} disabled={loading} /></label>
          <button type="button" className="secondary" onClick={submitImport} disabled={loading}>{loading ? "Working…" : "Import and inspect"}</button>
        </div>
        {importInputError ? <p className="error">{importInputError}</p> : null}
        {importErrors.length ? <div className="import-errors"><strong>{importErrors.length} row{importErrors.length === 1 ? "" : "s"} could not be imported</strong><ul>{importErrors.map((item) => <li key={item.row}>Row {item.row}: {item.reason}{item.smiles ? ` (${item.smiles})` : ""}</li>)}</ul></div> : null}
      </section>
      {sampler !== "imported" ? <section className="quality-panel surface-card" aria-label="Generated molecule quality" aria-busy={loading}>
        <div className="section-head"><h2>Generated molecule quality</h2><span className="mini-label">{loading ? "CALCULATING…" : quality ? `${quality.count} CANDIDATES · ENTIRE BATCH` : "NO BATCH YET"}</span></div>
        <div className="quality-grid">
          {[
            { label: "Valid structures", value: quality?.validity, note: quality?.attempted ? `${quality.valid_attempts} of ${quality.attempted} decoded attempts` : "Awaiting generation attempts", help: "Chemically valid, nonempty structures divided by every decoded sample, before deduplication and property filtering. Includes complete sampling rounds." },
            { label: "New molecules", value: quality?.novelty, note: quality ? quality.novelty === null ? "Training dataset unavailable" : "New to target training data" : "Compared with target training data", help: quality ? `${quality.novelty_note} ${quality.reference_count ?? 0} reference molecules. ${quality.datasets.join(', ')}` : "Compared with pretraining and fine-tuning datasets. Base training is excluded." },
            { label: "Structural diversity", value: quality?.diversity, note: "Variation within the candidate batch", help: "Average pairwise dissimilarity using Morgan fingerprints (radius 2, 2048 bits) and Tanimoto similarity. Excludes self-pairs. Requires at least two candidates; repeated structures reduce diversity.", score: true },
            { label: "Chemical checks passed", value: quality?.chemical_pass, note: "Passed RDKit PAINS screening", help: "Fraction of displayed batch candidates with no RDKit PAINS alerts. Lipinski checks are separate. This is PAINS screening, not the full MOSES filter suite or a safety assessment." },
          ].map(metric => <div className="quality-card" key={metric.label}><div className="quality-label">{metric.label}<span className="quality-info" tabIndex={0} role="img" aria-label={metric.help}>ⓘ<span role="tooltip">{metric.help}</span></span></div><strong>{loading ? "…" : metric.value == null ? quality ? "Unavailable" : "—" : metric.score ? `${metric.value.toFixed(2)} / 1` : `${(metric.value * 100).toFixed(1).replace(/\.0$/, '')}%`}</strong><p>{metric.note}</p></div>)}
        </div>
        {screening ? <p className="hint quality-foot">
          {quality.count} of {screening.requested} requested candidates accepted.
          {screening.accepted_per_minute != null && screening.elapsed_seconds != null &&
            ` ${screening.accepted_per_minute.toLocaleString(undefined, { maximumFractionDigits: 1 })} accepted/min over ${(screening.elapsed_seconds / 60).toFixed(2)} min of sampling and screening.`}
          {screening.reference_coverage === "partial" && " Training-set exclusion used a partial reference."}
          {screening.reference_coverage === "unavailable" && " Training-set exclusion requires reference data for this checkpoint."}
        </p> : null}
        <p className="hint quality-foot">Batch scores stay unchanged when you refine results. They do not establish biological activity or safety.</p>
      </section> : null}
      {error ? <p className="error banner">{error}</p> : null}
      <div className="generate-layout">
      <aside className="filters surface-card">
        <div className="section-head"><h2>Generation setup</h2><span className="mini-label">{sampler === "imported" ? "IMPORTED" : sampler === "gentrl" ? "GENTRL" : "MOCK"}</span></div>
        <label className="slider"><span>Model stage</span><select value={modelStage} onChange={(e) => onModelStageChange(e.target.value as ModelStage)} disabled={loading}><option value="before_rl">Before RL</option><option value="after_rl">After RL</option></select></label>
        <label className="slider"><span>Batch size</span><input type="number" min={1} step={1} value={batchSize} onChange={(e) => { const next = Number(e.target.value); if (Number.isFinite(next)) setBatchSize(Math.max(1, Math.round(next))); }} /></label>
        <p className="hint">{sampler === "imported" ? "Showing imported structures. Generate a batch to use the model." : sampler === "gentrl" ? "Using the GENTRL sampler." : "Using mock SMILES while the GENTRL worker is unavailable."}</p>
        <hr className="divider" />
        <div className="section-head"><h2>Refine results</h2><button type="button" className="text-button" onClick={() => {setLipinskiOnly(false); setBbbOnly(false); setNoPains(false); setNoBrenk(false); setSaMax(10); setQedMin(0);}}>Reset</button></div>
        <label className="check">
          <input type="checkbox" checked={lipinskiOnly} onChange={(e) => setLipinskiOnly(e.target.checked)} />
          Lipinski pass
        </label>
        <label className="check">
          <input type="checkbox" checked={bbbOnly} onChange={(e) => setBbbOnly(e.target.checked)} />
          BBB pass
        </label>
        <label className="check">
          <input type="checkbox" checked={noPains} onChange={(e) => setNoPains(e.target.checked)} />
          No PAINS
        </label>
        <label className="check">
          <input type="checkbox" checked={noBrenk} onChange={(e) => setNoBrenk(e.target.checked)} />
          No Brenk alerts
        </label>
        <label className="slider">
          <span>SAscore at or below <strong>{saMax.toFixed(1)}</strong></span>
          <input
            type="range"
            min={1}
            max={10}
            step={0.1}
            value={saMax}
            onChange={(e) => setSaMax(Number(e.target.value))}
          />
          <span className="hint">1 easy · 10 hard. Lower is stricter.</span>
        </label>
        <label className="slider">
          <span>QED at or above <strong>{qedMin.toFixed(2)}</strong></span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={qedMin}
            onChange={(e) => setQedMin(Number(e.target.value))}
          />
          <span className="hint">0–1. Higher is stricter.</span>
        </label>
      </aside>

      <section className="results surface-card">
        <div className="results-head"><div><h2>Candidate library</h2><p className="muted">{molecules.length ? `${molecules.length} ${sampler === "imported" ? "imported" : "generated"} · ${visible.length} match current filters` : "Your molecules will appear here"}</p></div><div className="segmented"><button type="button" className={resultView === "grid" ? "active" : ""} onClick={() => setResultView("grid")}>Grid</button><button type="button" className={resultView === "table" ? "active" : ""} onClick={() => setResultView("table")}>Table</button></div></div>
        {pageCount > 1 ? <nav className="results-pagination" aria-label="Candidate pages">
          <span className="muted" aria-live="polite">{pageStart + 1}–{pageStart + pageMolecules.length} of {visible.length}</span>
          <div className="segmented">
            <button type="button" disabled={page === 1} onClick={() => setResultPage(page - 1)}>Previous</button>
            <select aria-label="Candidate page" value={page} onChange={(event) => setResultPage(Number(event.target.value))}>
              {Array.from({ length: pageCount }, (_, index) => <option key={index} value={index + 1}>Page {index + 1}</option>)}
            </select>
            <button type="button" disabled={page === pageCount} onClick={() => setResultPage(page + 1)}>Next</button>
          </div>
        </nav> : null}
        {loading && !molecules.length && sampler !== "imported" ? (
          <div className="empty-state" role="status"><h3>Working on your request</h3><p>Large batches can take longer to finish.</p></div>
        ) : !molecules.length && !error ? (
          <div className="empty-state"><span>⬡</span><h3>Start a new discovery batch</h3><p>Generate molecules to inspect their structures and predicted properties.</p></div>
        ) : !visible.length ? (
          <div className="empty-state"><h3>No matching molecules</h3><p>Adjust the filters to see more candidates.</p></div>
        ) : resultView === "table" ? (
          <div className="table-scroll"><table className="result-table"><thead><tr><th>Molecule</th><th>SMILES</th><th>QED</th><th>SA</th><th>LogP</th><th>pIC50</th><th>Lipinski</th><th>BBB</th><th>PAINS</th><th>Brenk alerts</th></tr></thead><tbody>{pageMolecules.map((mol) => <tr key={mol.id} onClick={() => onSelect(mol.id)}><td>{mol.id}</td><td className="mono" title={mol.smiles}>{mol.smiles}</td><td>{mol.qed.toFixed(2)}</td><td>{mol.sa_score.toFixed(1)}</td><td>{mol.logp.toFixed(2)}</td><td>{formatPic50(mol.pic50)}</td><td>{mol.lipinski ? "Pass" : "Fail"}</td><td>{mol.bbb ? "Pass" : "Fail"}</td><td>{mol.pains ? "Alert" : "Clear"}</td><td>{mol.brenk_alerts.length}</td></tr>)}</tbody></table></div>
        ) : (
          <div className="grid">
            {pageMolecules.map((mol) => (
              <button
                key={mol.id}
                className={selected?.id === mol.id ? "card selected" : "card"}
                onClick={() => onSelect(mol.id)}
              >
                <div className="mol-card-head"><strong>{mol.id}</strong><span>View details →</span></div>
                <img src={`data:image/png;base64,${mol.image}`} alt={`2D structure for ${mol.id}`} loading="lazy" decoding="async" />
                <p className="card-smiles" title={mol.smiles}>{mol.smiles}</p>
                <div className="badges">
                  <span className="badge pass">QED {mol.qed.toFixed(2)}</span><span className={mol.sa_easy ? "badge" : "badge warn"}>SA {mol.sa_score.toFixed(1)}</span><span className="badge">pIC50 {formatPic50(mol.pic50)}</span>
                </div>
                <div className="card-checks">
                  <span className={mol.lipinski ? "ok" : "bad"}>{mol.lipinski ? "✓ Lipinski" : "× Lipinski"}</span>
                  <span className={mol.bbb ? "ok" : "bad"}>{mol.bbb ? "✓ BBB pass" : "× BBB fail"}</span>
                  <span className={mol.pains ? "bad" : "ok"}>{mol.pains ? "! PAINS alert" : "✓ PAINS clear"}</span>
                  <span className={mol.brenk_alerts.length ? "bad" : "ok"}>Brenk {mol.brenk_alerts.length}</span>
                </div>
              </button>
            ))}
          </div>
        )}
        <p className="results-foot">Select a molecule for its structure, properties, and Lens explanations. Computed and predicted values are not laboratory measurements.</p>
      </section>
      </div>

      {selected && detailOpen ? (
        <div className="modal-backdrop" onClick={onCloseDetail}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h2>Molecule</h2>
              <button type="button" className="close" onClick={onCloseDetail}>
                Close
              </button>
            </div>
            <MoleculeStructure key={selected.id} molecule={selected} />
            <p className="smiles">{selected.smiles}</p>
            {selected.quality ? (
              <section className="molecule-quality" aria-label="Molecule quality checks">
                <h3>Structure checks</h3>
                <div className="badges">
                  <span className="badge pass">✓ Valid structure</span>
                  <span className={selected.quality.novel === null ? "badge" : selected.quality.novel ? "badge pass" : "badge warn"}>
                    {selected.quality.novel === null ? "Training dataset unavailable" : selected.quality.novel ? "✓ New to target training data" : "In target training data"}
                  </span>
                  <span className={selected.quality.chemical_pass ? "badge pass" : "badge warn"}>
                    {selected.quality.chemical_pass ? "✓ PAINS checks passed" : "PAINS alert"}
                  </span>
                  {brenkBadge}
                </div>
                {selected.quality.alerts.map(alert => <p className="hint" key={alert}>{alert}</p>)}
                {brenkAlerts.length ? <p className="hint">Brenk matches {brenkAlerts.join(", ")}</p> : null}
                <p className="hint">Structural diversity describes the whole batch.</p>
              </section>
            ) : null}
            <dl>
              <div>
                <ScoreLabel name="MW" />
                <dd>{selected.mw}</dd>
              </div>
              <div>
                <ScoreLabel name="LogP" />
                <dd>{selected.logp}</dd>
              </div>
              {selected.logs !== undefined ? <div>
                <ScoreLabel name="LogS" />
                <dd>{selected.logs.toFixed(2)}</dd>
              </div> : null}
              <div>
                <ScoreLabel name="HBD" />
                <dd>{selected.hbd}</dd>
              </div>
              <div>
                <ScoreLabel name="HBA" />
                <dd>{selected.hba}</dd>
              </div>
              <div>
                <ScoreLabel name="SAscore" />
                <dd>{selected.sa_score}</dd>
              </div>
              <div>
                <ScoreLabel name="QED" />
                <dd>{selected.qed.toFixed(2)}</dd>
              </div>
              <div>
                <dt>BBB</dt>
                <dd><span className={selected.bbb ? "badge pass" : "badge fail"}>{selected.bbb ? "Pass" : "Fail"}</span></dd>
              </div>
              {sampler !== "imported" ? <div>
                <ScoreLabel name="Reward" />
                <dd>{formatReward(selected.reward)}</dd>
              </div> : null}
              <div>
                <ScoreLabel name="pIC50" />
                <dd>{formatPic50(selected.pic50)}</dd>
              </div>
            </dl>
            <div className="badges">
              <span className={selected.lipinski ? "badge pass" : "badge fail"}>
                {selected.lipinski ? "Lipinski pass" : "Lipinski fail"}
              </span>
              <span className={selected.pains ? "badge fail" : "badge pass"}>
                {selected.pains ? "PAINS alert" : "PAINS clear"}
              </span>
              {brenkBadge}
              <span className={selected.sa_easy ? "badge pass" : "badge warn"}>
                SA {selected.sa_score} {selected.sa_easy ? "(easy)" : "(hard)"}
              </span>
              <span className="badge">QED {selected.qed.toFixed(2)}</span>
              {sampler !== "imported" ? <span className="badge reward">Reward {formatReward(selected.reward)}</span> : null}
              <span className="badge">pIC50 {formatPic50(selected.pic50)}</span>
            </div>
            <div className="molecule-actions"><button type="button" className="secondary" onClick={onEditInDraw}>Edit in Draw ✎</button><button type="button" className="primary" onClick={onOpenLens}>Inspect in Lens →</button></div>
            {selected.pains_alert ? <p className="muted">{selected.pains_alert}</p> : null}
            {!selected.quality && brenkAlerts.length ? <p className="muted">Brenk matches {brenkAlerts.join(", ")}</p> : null}
          </div>
        </div>
      ) : null}
    </main>
  );
}
