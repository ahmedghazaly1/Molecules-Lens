import { useEffect, useState } from "react";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import type { Ketcher } from "ketcher-core";
import "ketcher-react/dist/index.css";

const provider = new StandaloneStructServiceProvider();

type Props = {
  initialSmiles: string;
  loading: boolean;
  onAnalyze: (smiles: string) => Promise<void>;
};

export function DrawPage({ initialSmiles, loading, onAnalyze }: Props) {
  const [editor, setEditor] = useState<Ketcher | null>(null);
  const [loadedStructure, setLoadedStructure] = useState<{ editor: Ketcher; smiles: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const transferring = Boolean(initialSmiles && (loadedStructure?.editor !== editor || loadedStructure?.smiles !== initialSmiles));

  useEffect(() => {
    if (!editor || !initialSmiles) return;
    let current = true;
    setError(null);
    async function loadStructure(instance: Ketcher) {
      try {
        await instance.setMolecule(initialSmiles);
        if (!instance.editor.struct().atoms.size) throw new Error("The editor could not load this molecule");
        if (current) setLoadedStructure({ editor: instance, smiles: initialSmiles });
      } catch (err) {
        if (current) setError(err instanceof Error ? err.message : "Could not load structure");
      }
    }
    void loadStructure(editor);
    return () => { current = false; };
  }, [initialSmiles, editor]);

  async function analyze() {
    setError(null);
    try {
      const smiles = (await editor?.getSmiles())?.trim();
      if (!smiles) throw new Error("Draw a molecule first");
      await onAnalyze(smiles);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read the drawing");
    }
  }

  return <main className="page-content">
    <div className="page-heading"><div><p className="eyebrow">02 / DESIGN</p><h1>Draw a molecule</h1><p>Sketch a structure or edit a generated molecule, then analyze it in Lens.</p></div><button className="primary" onClick={analyze} disabled={!editor || (transferring && !error) || loading}>{loading ? "Analyzing…" : "Analyze molecule →"}</button></div>
    {error ? <p className="error banner">{error}</p> : null}
    {transferring && !error ? <p className="muted" role="status">Loading the selected molecule…</p> : null}
    <section className="draw-editor surface-card" aria-label="Molecule editor">
      <Editor staticResourcesUrl="/" structServiceProvider={provider} disableMacromoleculesEditor onInit={setEditor} errorHandler={setError} />
    </section>
    <p className="hint draw-note">Edited structures are analyzed as imported molecules. They do not inherit the original model reward.</p>
  </main>;
}
