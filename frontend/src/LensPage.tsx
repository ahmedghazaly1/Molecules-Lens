import { useEffect, useState } from "react";
import { explainMolecule, type Explanation, type Molecule, type PropertyCheck, type RewardShap } from "./api";

type ColorMode = "logp" | "sa" | "pains" | "brenk" | "map";

const COLOR_COPY: Record<ColorMode, string> = {
  logp: "Orange atoms raise LogP (more fat-soluble). Blue atoms lower it (more water-soluble).",
  sa: "Orange atoms sit in fragments the SA scorer treats as harder to make. Blue atoms sit in easier fragments.",
  pains: "Pink atoms are the PAINS match. A clear molecule has no highlight.",
  brenk: "Pink highlights mark the atoms in matched Brenk patterns.",
  map: "Green atoms support the Dice similarity to the nearest reference molecule. Pink atoms work against it. Grey atoms do not change it.",
};

function atomLabel(symbol: string | undefined, index: number): string {
  return `${symbol ?? "atom"} ${index + 1}`;
}

function LogPReading({ atoms }: { atoms: Explanation["logp_atoms"] }) {
  const ranked = [...atoms].filter((atom) => atom.contribution != null && Math.abs(atom.contribution) >= 0.02);
  const raises = ranked
    .filter((atom) => (atom.contribution ?? 0) > 0)
    .sort((a, b) => (b.contribution ?? 0) - (a.contribution ?? 0))
    .slice(0, 3);
  const lowers = ranked
    .filter((atom) => (atom.contribution ?? 0) < 0)
    .sort((a, b) => (a.contribution ?? 0) - (b.contribution ?? 0))
    .slice(0, 3);
  if (!raises.length && !lowers.length) return null;
  return (
    <div className="logp-reading">
      {raises.length ? (
        <p>
          <span className="swatch raise" />
          Raises LogP:{" "}
          {raises.map((atom) => `${atomLabel(atom.symbol, atom.atom)} (${atom.contribution?.toFixed(2)})`).join(", ")}
        </p>
      ) : null}
      {lowers.length ? (
        <p>
          <span className="swatch lower" />
          Lowers LogP:{" "}
          {lowers.map((atom) => `${atomLabel(atom.symbol, atom.atom)} (${atom.contribution?.toFixed(2)})`).join(", ")}
        </p>
      ) : null}
    </div>
  );
}

const REWARD_INPUTS = [
  { key: "qed", label: "QED" },
  { key: "sa", label: "SA" },
  { key: "logp", label: "LogP" },
] as const;

function RewardBars({ reward }: { reward: RewardShap }) {
  const maxAbs = Math.max(...REWARD_INPUTS.map((item) => Math.abs(reward.shap[item.key])), 1e-9);
  return (
    <article className="chart-card">
      <h3>Reward {reward.reward.toFixed(2)}</h3>
      {reward.structure_similarity != null && reward.property_similarity != null ? (
        <p className="muted">Nearest reference structure similarity {reward.structure_similarity.toFixed(2)} (50% of reward); QED, SA, and LogP similarity {reward.property_similarity.toFixed(2)} (50%).</p>
      ) : null}
      <p className="muted">
        Each bar is how that property moved the reward relative to the {reward.reference ?? "reference"} mean while structure similarity stays fixed. Baseline{" "}
        {reward.baseline.toFixed(2)}. Baseline plus the three values equals the reward.
      </p>
      <ul className="shap-list">
        {REWARD_INPUTS.map((item) => {
          const value = reward.shap[item.key];
          const width = `${Math.round((Math.abs(value) / maxAbs) * 50)}%`;
          return (
            <li key={item.key}>
              <span>
                {item.label} = {reward.inputs[item.key].toFixed(2)} (ref {reward.reference_means[item.key].toFixed(2)})
              </span>
              <span className="shap-track">
                <i className={value >= 0 ? "pos" : "neg"} style={value >= 0 ? { left: "50%", width } : { right: "50%", width }} />
              </span>
              <span>{value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2)}</span>
            </li>
          );
        })}
      </ul>
      {reward.reference_rows && reward.reference_rows.dropped > 0 ? (
        <p className="muted">
          Reference mean uses {reward.reference_rows.kept} parsed rows. {reward.reference_rows.dropped} rows were dropped.
        </p>
      ) : null}
    </article>
  );
}

type Props = {
  selected: Molecule | null;
  original: Molecule | null;
  onOpenGenerate: () => void;
};

function ComparisonCard({ title, molecule }: { title: string; molecule: Molecule }) {
  const properties = [
    ["Molecular weight", `${molecule.mw.toFixed(2)} Da`],
    ["LogP", molecule.logp.toFixed(2)],
    ["H-bond donors", molecule.hbd],
    ["H-bond acceptors", molecule.hba],
    ["QED", molecule.qed.toFixed(2)],
    ["SA score", molecule.sa_score.toFixed(2)],
    ["Lipinski", molecule.lipinski ? "Pass" : "Fail"],
    ["BBB", molecule.bbb ? "Pass" : "Fail"],
    ["PAINS", molecule.pains_alert ?? (molecule.pains ? "Alert" : "Clear")],
    ["Brenk alerts", molecule.brenk_alerts.join(", ") || "None"],
    ["Predicted pIC50", molecule.pic50?.toFixed(2) ?? "Unavailable"],
    ["Model reward", molecule.reward?.toFixed(2) ?? "Unavailable"],
  ];

  return (
    <article className="detail comparison-card">
      <h2>{title}</h2>
      <div className="structure comparison-structure">
        <img src={`data:image/png;base64,${molecule.image}`} alt={`${title} 2D structure`} />
      </div>
      <div>
        <h3>Calculated properties</h3>
        <dl className="comparison-properties">
          {properties.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <p className="smiles" aria-label="SMILES">{molecule.smiles}</p>
      </div>
    </article>
  );
}

function PropertyChecks({ checks }: { checks: PropertyCheck[] }) {
  return (
    <ul className="check-list">
      {checks.map((check) => (
        <li key={check.name}>
          <span className={check.ok ? "badge pass" : "badge fail"}>{check.ok ? "Pass" : "Fail"}</span>
          <span>{check.name} {check.value.toLocaleString("en-US", { maximumFractionDigits: 6 })} ({check.limit})</span>
        </li>
      ))}
    </ul>
  );
}

export function LensPage({ selected, original, onOpenGenerate }: Props) {
  return (
    <main className="page-content">
      <div className="page-heading"><div><p className="eyebrow">03 / INTERPRET</p><h1>{original ? "Molecule comparison" : "Molecular Lens"}</h1><p>{original ? "Compare the original and edited structures and their calculated properties." : "See which atoms and properties contribute to a candidate’s predicted profile."}</p></div></div>
      {selected ? (
        original ? (
          <>
            <div className="molecule-comparison">
              <ComparisonCard title="Original molecule" molecule={original} />
              <ComparisonCard title="Edited molecule" molecule={selected} />
            </div>
            <details className="surface-card comparison-analysis">
              <summary>Detailed analysis of the edited molecule</summary>
              <LensAnalysis selected={selected} />
            </details>
          </>
        ) : <LensAnalysis selected={selected} />
      ) : (
        <section className="surface-card empty-state"><span>◎</span><h2>Select a molecule to inspect</h2><p>Generate candidates, choose one, and return here for atom maps and property explanations.</p><button className="primary" onClick={onOpenGenerate}>Go to Generate →</button></section>
      )}
    </main>
  );
}

function LensAnalysis({ selected }: { selected: Molecule }) {
  const [mode, setMode] = useState<ColorMode>("logp");
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const smiles = selected.smiles;
    let cancelled = false;
    if (selected.id.startsWith("import-")) setMode("logp");
    setExplanation(null);
    setLoading(true);
    setError(null);
    explainMolecule(smiles, !selected.id.startsWith("import-"))
      .then((data) => {
        if (!cancelled) setExplanation(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setExplanation(null);
          setError(err instanceof Error ? err.message : "Could not explain this molecule");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const weakest = explanation?.qed_terms.find((term) => term.name === explanation.weakest_qed);

  return (
    <div className="lens-layout">
      <section className="detail">
        <div className="section-head"><div><h2>Candidate {selected.id}</h2><p className="muted">{selected.id.startsWith("import-") ? "Imported molecule" : "Selected from the latest generated batch"}</p></div><span className={selected.lipinski ? "badge pass" : "badge fail"}>{selected.lipinski ? "Lipinski pass" : "Lipinski fail"}</span></div>
        <p className="smiles">{selected.smiles}</p>
        <div className="view-toggle lens-toggle">
          <button type="button" className={mode === "logp" ? "active" : ""} onClick={() => setMode("logp")}>
            LogP
          </button>
          <button type="button" className={mode === "sa" ? "active" : ""} onClick={() => setMode("sa")}>
            SA
          </button>
          <button type="button" className={mode === "pains" ? "active" : ""} onClick={() => setMode("pains")}>
            PAINS
          </button>
          <button type="button" className={mode === "brenk" ? "active" : ""} onClick={() => setMode("brenk")}>
            Brenk
          </button>
          {!selected.id.startsWith("import-") ? <button type="button" className={mode === "map" ? "active" : ""} onClick={() => setMode("map")}>
            Structural similarity map
          </button> : null}
        </div>
        <div className="structure lens-structure">
          {explanation && mode === "map" && explanation.similarity ? (
            <img src={`data:image/png;base64,${explanation.similarity.image}`} alt="" />
          ) : explanation && mode !== "map" ? (
            <img src={`data:image/png;base64,${explanation.images[mode]}`} alt="" />
          ) : (
            <img src={`data:image/png;base64,${selected.image}`} alt="" />
          )}
        </div>
        <p className="lens-description">{mode === "map" && explanation?.similarity ? `${COLOR_COPY.map} Dataset: ${explanation.similarity.reference_dataset}.` : COLOR_COPY[mode]}</p>
        {mode === "logp" && explanation ? <LogPReading atoms={explanation.logp_atoms} /> : null}
        {mode === "map" && explanation?.similarity ? (
          <p className="smiles">
            {explanation.similarity.reference_dataset} · Reference {explanation.similarity.reference_smiles} · Dice {explanation.similarity.dice.toFixed(2)}
          </p>
        ) : null}
        {mode === "map" && explanation && !explanation.similarity ? (
          <p className="error">{explanation.similarity_error ?? "No structural similarity map for this molecule."}</p>
        ) : null}
        {loading ? <p className="muted">Explaining…</p> : null}
        {error ? <p className="error">{error}</p> : null}
      </section>

      {explanation ? (
        <section className="lens-readout">
          <article className="chart-card property-summary"><div className="section-head"><div><h3>Property summary</h3><p className="muted">Computed and predicted values</p></div><strong className="qed-hero">{explanation.qed.toFixed(2)} <small>QED</small></strong></div><div className="run-stats"><div><strong>{explanation.logp.toFixed(2)}</strong><span>LogP</span></div><div><strong>{explanation.sa_score.toFixed(2)}</strong><span>SA score</span></div><div><strong>{selected.pic50?.toFixed(2) ?? "—"}</strong><span>Pred. pIC50</span></div></div><p className="hint">pIC50 is model predicted, not measured activity.</p></article>
          {explanation.reward_shap ? <RewardBars reward={explanation.reward_shap} /> : null}
          {explanation.reward_error ? <p className="error">{explanation.reward_error}</p> : null}
          <article className="chart-card">
            <h3>QED {explanation.qed.toFixed(2)}</h3>
            <p className="muted">
              Each term is mapped to a desirability from 0 to 1. The weakest term is{" "}
              <strong>{weakest?.label}</strong> at {weakest?.desirability.toFixed(2)}.
            </p>
            <ul className="term-list">
              {explanation.qed_terms.map((term) => (
                <li key={term.name} className={term.name === explanation.weakest_qed ? "weak" : ""}>
                  <span>{term.label}</span>
                  <span>{term.value}</span>
                  <span className="bar">
                    <i style={{ width: `${Math.round(term.desirability * 100)}%` }} />
                  </span>
                  <span>{term.desirability.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </article>

          <article className="chart-card">
            <h3>Lipinski</h3>
            <PropertyChecks checks={explanation.lipinski} />
          </article>

          <article className="chart-card">
            <h3>BBB {explanation.bbb.every((check) => check.ok) ? "pass" : "fail"}</h3>
            <PropertyChecks checks={explanation.bbb} />
          </article>

          <article className="chart-card">
            <h3>PAINS</h3>
            <p className={explanation.pains ? "error" : "muted"}>
              {explanation.pains
                ? `${explanation.pains_alert}. ${explanation.pains_atoms.length} atoms are highlighted in the PAINS view.`
                : "No PAINS match."}
            </p>
            <p className="muted">
              LogP {explanation.logp.toFixed(2)} · SA {explanation.sa_score.toFixed(2)}. These are the same RDKit scores as Generate. The colors show which atoms feed them.
            </p>
          </article>
          <article className="chart-card">
            <h3>Brenk alerts ({explanation.brenk_alerts.length})</h3>
            <p className="muted">Structural patterns associated with potential reactivity, instability, toxicity, or poor pharmacokinetics.</p>
            {explanation.brenk_alerts.length ? (
              <ul>
                {explanation.brenk_alerts.map((alert) => <li key={alert}>{alert}</li>)}
              </ul>
            ) : <p className="muted">No Brenk patterns matched.</p>}
          </article>
        </section>
      ) : null}
    </div>
  );
}
