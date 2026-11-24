import { useState } from "react";
import { generateBatch, importMolecules, type BatchQuality, type ImportError, type ModelStage, type Molecule } from "./api";
import { GeneratePage } from "./GeneratePage";
import { LensPage } from "./LensPage";
import { TrainPage } from "./TrainPage";
import { DrawPage } from "./DrawPage";

type Workspace = "generate" | "draw" | "train" | "lens";
const labels: Record<Workspace, string> = { generate: "Generate", draw: "Draw", train: "Training", lens: "Lens" };

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>("generate");
  const [molecules, setMolecules] = useState<Molecule[]>([]);
  const [quality, setQuality] = useState<BatchQuality | null>(null);
  const [modelStage, setModelStage] = useState<ModelStage>("after_rl");
  const [sampler, setSampler] = useState("mock");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [importErrors, setImportErrors] = useState<ImportError[]>([]);
  const [drawOriginal, setDrawOriginal] = useState<Molecule | null>(null);
  const [comparison, setComparison] = useState<{ original: Molecule; edited: Molecule } | null>(null);

  async function onGenerate(n: number) {
    setLoading(true);
    setError(null);
    setQuality(null);
    setImportErrors([]);
    try {
      const data = await generateBatch(n, modelStage);
      setMolecules(data.molecules);
      setQuality(data.quality);
      setSampler(data.sampler);
      setSelectedId(null);
      setComparison(null);
      setDetailOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate molecules");
    } finally {
      setLoading(false);
    }
  }

  function showImportedBatch(data: Awaited<ReturnType<typeof importMolecules>>, selectedId: string | null = null) {
    setMolecules(data.molecules);
    setImportErrors(data.errors);
    setQuality(null);
    setSampler("imported");
    setSelectedId(selectedId);
    setComparison(null);
    setDetailOpen(false);
  }

  async function onImport(text: string, format: "smiles" | "csv") {
    setLoading(true);
    setError(null);
    try {
      const data = await importMolecules(text, format);
      showImportedBatch(data);
      return data.molecules;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not import molecules");
      throw err;
    } finally {
      setLoading(false);
    }
  }

  async function onDrawAnalyze(smiles: string) {
    setLoading(true);
    try {
      const data = await importMolecules(smiles, "smiles");
      const edited = data.molecules[0];
      if (!edited) throw new Error("The drawn structure could not be analyzed");
      if (drawOriginal) {
        setComparison({ original: drawOriginal, edited });
      } else {
        showImportedBatch(data, edited.id);
      }
      setWorkspace("lens");
    } finally {
      setLoading(false);
    }
  }

  const selected = molecules.find((m) => m.id === selectedId) ?? null;
  const lensSelected = comparison?.edited ?? selected;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark" aria-hidden="true">⬡</span><span>MoleculeLens</span></div>
        <div className="sidebar-group">
          <p className="sidebar-label">WORKSPACE</p>
          <nav className="side-nav" aria-label="Main navigation">
            {(["generate", "draw", "train", "lens"] as Workspace[]).map((tab) => (
              <button key={tab} type="button" className={workspace === tab ? "active" : ""} onClick={() => { if (tab === "draw" && workspace !== "draw") setDrawOriginal(null); setWorkspace(tab); }} aria-current={workspace === tab ? "page" : undefined}>
                <span className="nav-icon" aria-hidden="true">{tab === "generate" ? "◈" : tab === "draw" ? "✎" : tab === "train" ? "▥" : "◎"}</span>{labels[tab]}
              </button>
            ))}
          </nav>
        </div>
        <p className="sidebar-foot"><strong>Research workspace</strong><br />Molecule discovery, model runs, and interpretation.</p>
      </aside>
      <div className="app-main">
        <header className="app-topbar">
          <span>Workspace <span className="crumb-divider">/</span> <strong>{labels[workspace]}</strong></span>
          {lensSelected ? <button className="current-selection" onClick={() => setWorkspace("lens")} title={lensSelected.smiles}>◎ &nbsp; {comparison ? "Molecule comparison" : `Selected: ${lensSelected.id}`} <span>View in Lens →</span></button> : <span className="topbar-note">No molecule selected</span>}
        </header>
        {workspace === "generate" ? (
          <GeneratePage quality={quality} molecules={molecules} sampler={sampler} importErrors={importErrors} modelStage={modelStage} onModelStageChange={(stage) => { setModelStage(stage); setMolecules([]); setQuality(null); setImportErrors([]); setSelectedId(null); setComparison(null); setDetailOpen(false); setError(null); }} loading={loading} error={error} selected={selected} detailOpen={detailOpen} onGenerate={onGenerate} onImport={onImport} onEditInDraw={() => { if (selected) { setDrawOriginal(selected); setDetailOpen(false); setWorkspace("draw"); } }} onSelect={(id) => { setSelectedId(id); setComparison(null); setDetailOpen(true); }} onCloseDetail={() => setDetailOpen(false)} onOpenLens={() => { setDetailOpen(false); setWorkspace("lens"); }} />
        ) : workspace === "draw" ? <DrawPage initialSmiles={drawOriginal?.smiles ?? ""} loading={loading} onAnalyze={onDrawAnalyze} /> : workspace === "train" ? <TrainPage /> : <LensPage selected={lensSelected} original={comparison?.original ?? null} onOpenGenerate={() => setWorkspace("generate")} />}
      </div>
    </div>
  );
}
