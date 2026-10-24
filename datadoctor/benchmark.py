"""Data-quality benchmark.

Take a clean dataset, inject *known* corruption, and measure
  (A) how well each detector finds the corrupted rows  (precision / recall / F1)
  (B) how much valid data each cleaning pipeline damages or deletes

Detectors compared
  1. IQR rules                       textbook Tukey fences per column
  2. Isolation Forest (plain)        on raw numeric columns, top 5% flagged
     Isolation Forest (oracle rate)  same, but told the true share of error rows (an unfair advantage)
  3. LOF (plain)                     on raw numeric columns, top 5% flagged
  4. DataDoctor w/o relationships    rules + context z + Isolation Forest corroboration
  5. DataDoctor (combined)           the above + learned cross-column relationships

Ground truth for a row is "a numeric error was injected into it". Rows holding
*legitimate* extreme values (bulk orders) are negatives - flagging them is a false positive.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly import baseline_ml_rows, baseline_stat_rows, detect
from .plan import apply_plan, build_plan, default_approved
from .profiling import profile
from .synth import ERROR_TYPES, Truth, corrupt, make_clean_sales

METHODS = [
    "IQR rules",
    "Isolation Forest (plain)",
    "Isolation Forest (oracle rate)",
    "LOF (plain)",
    "DataDoctor w/o relationships",
    "DataDoctor (combined)",
]


def _prf(flag: pd.Series, pos: pd.Series):
    tp = int((flag & pos).sum())
    fp = int((flag & ~pos).sum())
    fn = int((~flag & pos).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def row_flags(findings: pd.DataFrame, index, verdicts=("error", "review")) -> pd.Series:
    flag = pd.Series(False, index=index)
    if len(findings):
        rows = findings.loc[findings.verdict.isin(verdicts), "row"].unique()
        flag.loc[rows] = True
    return flag


def evaluate_detection(dirty: pd.DataFrame, truth: Truth, seed: int = 0):
    prof = profile(dirty)
    typed = prof.typed
    full = detect(typed, prof.kinds, seed=seed)
    ablate = detect(typed, prof.kinds, seed=seed, use_relationships=False)
    cols = full.num_cols
    flags = {
        "IQR rules": baseline_stat_rows(typed, cols),
        "Isolation Forest (plain)": baseline_ml_rows(typed, cols, "iforest", 0.05, seed),
        "Isolation Forest (oracle rate)": baseline_ml_rows(typed, cols, "iforest", float((truth.row_error_type != "").mean()), seed),
        "LOF (plain)": baseline_ml_rows(typed, cols, "lof", 0.05, seed),
        "DataDoctor w/o relationships": row_flags(ablate.findings, dirty.index),
        "DataDoctor (combined)": row_flags(full.findings, dirty.index),
    }
    pos = truth.row_error_type != ""
    legit = truth.legit_row
    clean_rows = ~pos & ~legit
    rows, by_type = [], []
    for name in METHODS:
        fl = flags[name]
        p, r, f1 = _prf(fl, pos)
        rows.append({
            "method": name, "precision": p, "recall": r, "f1": f1,
            "legit_rows_flagged_pct": 100 * fl[legit].mean(),
            "clean_rows_flagged_pct": 100 * fl[clean_rows].mean(),
            "rows_flagged": int(fl.sum()),
        })
        for et in ERROR_TYPES:
            m = truth.row_error_type == et
            by_type.append({"method": name, "error_type": et, "recall": float(fl[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(by_type), (prof, full)


# --------------------------------------------------------------------------- cleaning evaluation
def naive_clean(prof) -> pd.DataFrame:
    """What a beginner's one-liner does: drop duplicates, drop missing, drop IQR outliers."""
    t = prof.typed.copy()
    cols = [c for c, k in prof.kinds.items() if k["kind"] == "numeric" and t[c].nunique() > 5]
    t = t.drop_duplicates().dropna()
    return t.loc[~baseline_stat_rows(t, cols)]


def _eq(col: str, cleaned: pd.Series, true: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(true):
        a = pd.to_numeric(cleaned, errors="coerce").astype("float64")
        return pd.Series(np.isclose(a.to_numpy(), true.astype("float64").to_numpy(), rtol=0.01, atol=0.02) & a.notna().to_numpy(), index=cleaned.index)
    if pd.api.types.is_datetime64_any_dtype(true):
        if not pd.api.types.is_datetime64_any_dtype(cleaned):
            return pd.Series(False, index=cleaned.index)
        return (pd.to_datetime(cleaned) == true).fillna(False)
    return (cleaned.astype("string") == true.astype("string")).fillna(False).astype(bool)


def score_cleaning(dirty: pd.DataFrame, truth: Truth, cleaned: pd.DataFrame) -> dict:
    idx = cleaned.index.intersection(dirty.index)
    cleaned = cleaned.loc[idx]
    cols = [c for c in truth.clean.columns if c != "order_id" and c in cleaned.columns]
    t = truth.clean.loc[idx]
    err, miss = truth.err_cells.loc[idx], truth.missing_cells.loc[idx]
    var, dfm = truth.variant_cells.loc[idx], truth.datefmt_cells.loc[idx]
    legit = truth.legit_row.loc[idx]

    eq = pd.DataFrame({c: _eq(c, cleaned[c], t[c]) for c in cols})
    valid = ~(err | miss | var | dfm)[cols]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(t[c])]
    cat_cols = [c for c in cols if c in ("city", "product")]

    all_orders = set(truth.order_id)
    kept = set(truth.order_id.loc[idx])
    valid_orders = set(truth.order_id[truth.row_error_type == ""]) - set(truth.order_id[truth.row_error_type != ""])
    res = {
        "rows_out": int(len(idx)),
        "orders_lost_pct": 100 * (1 - len(kept & all_orders) / len(all_orders)),
        "valid_orders_lost_pct": 100 * len(valid_orders - kept) / max(len(valid_orders), 1),
        "residual_duplicate_rows": int(truth.order_id.loc[idx].duplicated().sum()),
        "valid_cells_damaged_pct": 100 * float((~eq & valid).to_numpy().sum()) / max(int(valid.to_numpy().sum()), 1),
    }
    # damage specifically to legitimate extreme (bulk) orders
    lv = valid.loc[legit, num_cols]
    res["legit_order_cells_damaged_pct"] = 100 * float((~eq.loc[legit, num_cols] & lv).to_numpy().sum()) / max(int(lv.to_numpy().sum()), 1)
    e = err[num_cols]
    total_err = max(int(truth.err_cells[num_cols].to_numpy().sum()), 1)  # all injected errors, incl. rows later deleted
    restored = int((eq[num_cols] & e).to_numpy().sum())
    untouched = 0
    for c in num_cols:
        m = err[c]
        if m.any():
            same = pd.to_numeric(cleaned.loc[m, c], errors="coerce").astype(float).to_numpy() == pd.to_numeric(dirty.loc[idx].loc[m, c], errors="coerce").astype(float).to_numpy()
            untouched += int(same.sum())
    res["error_cells_restored_pct"] = 100 * restored / total_err
    res["error_cells_left_wrong_pct"] = 100 * untouched / total_err
    res["error_cells_other_pct"] = 100 - res["error_cells_restored_pct"] - res["error_cells_left_wrong_pct"]  # blanked+imputed, or row deleted
    v = var[cat_cols]
    res["category_variants_restored_pct"] = 100 * float((eq[cat_cols] & v).to_numpy().sum()) / max(int(v.to_numpy().sum()), 1)
    if "order_date" in cols:
        m = dfm["order_date"]
        res["dates_parsed_correctly_pct"] = 100 * float((eq.loc[m, "order_date"]).mean()) if m.any() else np.nan
    # imputation quality on injected-missing numeric cells
    derivable = [c for c in ("quantity", "unit_price", "total") if c in num_cols]
    other = [c for c in num_cols if c not in derivable]

    def _imp(cs):
        out = []
        for c in cs:
            m = miss[c] & cleaned[c].notna()
            if m.any():
                a = pd.to_numeric(cleaned.loc[m, c], errors="coerce").astype(float)
                tv = t.loc[m, c].astype(float)
                out.append(pd.DataFrame({"rel": (a - tv).abs() / tv.abs().clip(lower=1e-9), "ok": np.isclose(a, tv, rtol=0.01, atol=0.02)}))
        return pd.concat(out) if out else None

    d, o = _imp(derivable), _imp(other)
    res["impute_derivable_within_1pct"] = 100 * float(d.ok.mean()) if d is not None else np.nan
    res["impute_other_median_rel_err_pct"] = 100 * float(o.rel.median()) if o is not None else np.nan
    return res


def evaluate_cleaning(dirty: pd.DataFrame, truth: Truth, prof, anom) -> pd.DataFrame:
    plan = build_plan(prof, anom)
    defaults = default_approved(plan)
    review_ids = {a.id for a in plan if a.kind == "review_outliers"}
    pipelines = {
        "Naive (dropna + drop_duplicates + drop IQR outliers)": naive_clean(prof),
        "DataDoctor (default approvals)": apply_plan(dirty, prof, anom, plan, defaults).cleaned,
        "DataDoctor (+ all review items approved)": apply_plan(dirty, prof, anom, plan, defaults | review_ids).cleaned,
    }
    rows = []
    for name, out in pipelines.items():
        r = score_cleaning(dirty, truth, out)
        r["pipeline"] = name
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- runner
def run_benchmark(n=5000, seeds=(0, 1, 2, 3, 4), rates=(0.01, 0.03, 0.05, 0.10), verbose=True):
    det_all, type_all, clean_all = [], [], []
    for rate in rates:
        for seed in seeds:
            t0 = time.time()
            clean, legit = make_clean_sales(n, seed)
            dirty, truth = corrupt(clean, legit, err_rate=rate, seed=seed)
            det, bytype, (prof, anom) = evaluate_detection(dirty, truth, seed)
            for d in (det, bytype):
                d["err_rate"], d["seed"] = rate, seed
            det_all.append(det)
            type_all.append(bytype)
            cl = evaluate_cleaning(dirty, truth, prof, anom)
            cl["err_rate"], cl["seed"] = rate, seed
            clean_all.append(cl)
            if verbose:
                print(f"rate={rate:.2f} seed={seed} done in {time.time()-t0:.1f}s", flush=True)
    return pd.concat(det_all, ignore_index=True), pd.concat(type_all, ignore_index=True), pd.concat(clean_all, ignore_index=True)


def _ms(df, by, cols, order=None):
    g = df.groupby(by)[cols].agg(["mean", "std"])
    if order is not None:
        g = g.reindex(order, level=by[-1] if isinstance(by, list) else by)
    return g


def to_markdown(det, bytype, clean, n, seeds) -> str:
    def fmt(m, s, pct=False):
        return f"{m:.1f} ± {s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"

    lines = [f"# DataDoctor benchmark results", "",
             f"Synthetic sales data, {n:,} clean rows per run, {len(seeds)} seeds ({', '.join(map(str, seeds))}); mean ± std across seeds.",
             "Each run injects: numeric errors (5 types, equal share), missing cells, exact+cosmetic duplicates, spelling variants and mixed date formats; "
             "~2% of rows are *legitimate* bulk orders (extreme but valid).", ""]
    lines += ["## A. Error-row detection (row is positive if a numeric error was injected)", ""]
    for rate, g in det.groupby("err_rate"):
        lines += [f"**Corruption rate {rate:.0%} of rows**", "",
                  "| Method | Precision | Recall | F1 | Legit bulk rows flagged % | Clean rows flagged % |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            x = g[g.method == m]
            lines.append(f"| {m} | {fmt(x.precision.mean(), x.precision.std())} | {fmt(x.recall.mean(), x.recall.std())} | {fmt(x.f1.mean(), x.f1.std())} | "
                         f"{fmt(x.legit_rows_flagged_pct.mean(), x.legit_rows_flagged_pct.std(), True)} | {fmt(x.clean_rows_flagged_pct.mean(), x.clean_rows_flagged_pct.std(), True)} |")
        lines.append("")
    lines += ["## B. Recall by error type (all corruption rates pooled)", "",
              "| Method | " + " | ".join(ERROR_TYPES) + " |", "|---|" + "---|" * len(ERROR_TYPES)]
    for m in METHODS:
        x = bytype[bytype.method == m].groupby("error_type").recall.mean()
        lines.append(f"| {m} | " + " | ".join(f"{x[e]:.2f}" for e in ERROR_TYPES) + " |")
    lines += ["", "## C. End-to-end cleaning: damage to valid data (all corruption rates pooled)", ""]
    cols = [("valid_orders_lost_pct", "Valid orders deleted %"),
            ("valid_cells_damaged_pct", "Valid cells damaged %"), ("legit_order_cells_damaged_pct", "Legit bulk-order cells damaged %"),
            ("error_cells_restored_pct", "Error cells restored to truth %"), ("error_cells_left_wrong_pct", "Error cells left wrong %"),
            ("category_variants_restored_pct", "Spelling variants fixed %"), ("dates_parsed_correctly_pct", "Mixed-format dates correct %"),
            ("residual_duplicate_rows", "Residual duplicate rows")]
    lines += ["| Pipeline | " + " | ".join(l for _, l in cols) + " |", "|---|" + "---|" * len(cols)]
    for pname, g in clean.groupby("pipeline", sort=False):
        lines.append(f"| {pname} | " + " | ".join(
            ("n/a" if pd.isna(g[c].mean()) else f"{g[c].mean():.1f}") for c, _ in cols) + " |")
    lines += ["", "Imputation quality (injected-missing cells):", "",
              "| Pipeline | Derivable columns (qty/price/total) within 1% of truth % | Other numeric columns: median relative error % |", "|---|---|---|"]
    for pname, g in clean.groupby("pipeline", sort=False):
        a, b = g.impute_derivable_within_1pct.mean(), g.impute_other_median_rel_err_pct.mean()
        lines.append(f"| {pname} | {'n/a' if pd.isna(a) else f'{a:.1f}'} | {'n/a' if pd.isna(b) else f'{b:.1f}'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    det, bytype, clean = run_benchmark(a.n, tuple(a.seeds), tuple(a.rates))
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    det.to_csv(out / "detection_by_rate.csv", index=False)
    bytype.to_csv(out / "detection_by_type.csv", index=False)
    clean.to_csv(out / "cleaning.csv", index=False)
    md = to_markdown(det, bytype, clean, a.n, a.seeds)
    (out / "benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
