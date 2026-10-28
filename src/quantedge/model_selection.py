"""Pick one model with a rule that's written down before looking at the results.

The danger with trying lots of models is that one of them will look great by luck. So the
rule is fixed in a config file first, applied only to the development period, and then
the winner is frozen before the test period is ever run.

Rule:
  1. Gates. Drop any model that a backtest rejects (p < gate_level) on the dev period.
  2. Score. Average the mean FZ0 loss over a few confidence levels (lower is better).
     FZ0 scores VaR and ES together and rewards a model that's right on calm and wild days.
  3. Winner = lowest score among models that pass every gate. If nobody passes everything,
     take the ones passing the most gates and pick the lowest score among those.
     If two scores are equal (within 1e-6), the simpler model wins.
"""
import numpy as np
import pandas as pd


def apply_selection_rule(table: pd.DataFrame, candidates: list[dict], selection: dict) -> pd.DataFrame:
    """table: one row per (method, confidence) with p-value columns and mean_fz0.

    Returns one row per candidate with gates passed, score and rank (rank 1 = winner).
    """
    level = float(selection["gate_level"])
    rows = []
    for spec in candidates:
        name = spec["name"]
        f = table[table.method == name].set_index("confidence")
        passed, failed = 0, []
        for gate in selection["gates"]:
            c, test = gate["confidence"], gate["test"]
            p = f[test].get(c, np.nan) if test in f else np.nan
            if np.isfinite(p) and p >= level:
                passed += 1
            else:
                failed.append(f"{test}@{c:g} (p={p:.3g})")
        score = float(np.mean([f.mean_fz0.get(c, np.nan) for c in selection["score_confidences"]]))
        rows.append({"method": name, "gates_passed": passed, "gates_total": len(selection["gates"]),
                     "failed_gates": "; ".join(failed) or "none", "score": score,
                     "complexity": spec.get("complexity", 0)})
    out = pd.DataFrame(rows)
    out["score_rounded"] = out.score.round(6)
    out = out.sort_values(["gates_passed", "score_rounded", "complexity"], ascending=[False, True, True])
    out["rank"] = np.arange(1, len(out) + 1)
    return out.drop(columns="score_rounded").reset_index(drop=True)
