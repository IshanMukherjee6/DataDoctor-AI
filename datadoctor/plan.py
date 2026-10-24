"""Stage 3 - explainable cleaning recommendations + approval-gated application.

`build_plan` turns profiling + anomaly findings into a list of Actions, each with a
plain-language explanation, a default (on/off) and alternative strategies.
`apply_plan` executes only the actions the user approved and writes a full audit log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiling import (
    Profile, clean_number_strings, norm_series, parse_dates_flexible, standardised_view,
)
from .anomaly import AnomalyResult

STEP = {"coerce": 10, "dates": 20, "categories": 30, "fuzzy": 35, "dedupe": 40, "fix": 50, "review": 55, "impute": 60}
CHANGE_LOG_CAP = 50_000


@dataclass
class Action:
    id: str
    step: int
    kind: str
    column: str | None
    title: str
    explanation: str
    affected: int
    params: dict = field(default_factory=dict)
    default_on: bool = True
    strategy: str | None = None
    options: list | None = None


# --------------------------------------------------------------------------- building the plan
def _recommend_imputation(col, kind, s: pd.Series, pct_missing, ctx, rels, outlier_share):
    """Return (strategy, options, explanation, default_on)."""
    if kind == "numeric":
        x = s.dropna().to_numpy(dtype=float)
        skew = 0.0
        if len(x) > 8 and x.std() > 0:
            skew = float(np.mean(((x - x.mean()) / x.std()) ** 3))
        opts = ["derive", "group_median", "median", "mean", "drop_rows", "leave"]
        if pct_missing > 50:
            return "leave", opts, f"{pct_missing:.0f}% of '{col}' is missing - imputing would mostly invent data, so it is left as-is (consider dropping the column).", False
        in_rel = [r for r in rels if col in (r.target, r.a, r.b)]
        if in_rel:
            r = in_rel[0]
            fb = "group_median" if ctx else "median"
            return "derive", opts, (f"'{col}' follows the learned rule {r.text()}; missing values are recomputed from the other columns of the same row "
                                    f"(exact), falling back to the {fb.replace('_', ' ')} when the other columns are also missing."), True
        if ctx:
            return "group_median", opts, f"'{col}' differs strongly by '{ctx[0]}' (explains {ctx[1]:.0%} of its spread), so each missing value gets the median of its own '{ctx[0]}' group rather than a global value.", True
        if abs(skew) > 1 or outlier_share > 0.01:
            return "median", opts, f"'{col}' is skewed / has outliers (skew={skew:.1f}, outliers={outlier_share:.1%}); the median is robust to both, the mean is not.", True
        return "mean", opts, f"'{col}' is roughly symmetric with few outliers (skew={skew:.1f}); the mean is an unbiased fill.", True
    if kind in ("categorical", "boolean"):
        vc = s.dropna().astype(str).value_counts(normalize=True)
        opts = ["mode", "constant:Unknown", "drop_rows", "leave"]
        top = float(vc.iloc[0]) if len(vc) else 0
        if pct_missing > 50:
            return "leave", opts, f"{pct_missing:.0f}% missing - left as-is.", False
        if top >= 0.30:
            return "mode", opts, f"The most common value ('{vc.index[0]}') covers {top:.0%} of known rows, so it is a reasonable fill.", True
        return "constant:Unknown", opts, f"No category dominates (top share {top:.0%}); filling with the mode would invent a majority, so a neutral 'Unknown' label is used.", True
    if kind == "date":
        return "leave", ["leave", "drop_rows"], "Dates should not be invented; missing dates are left blank and flagged.", False
    return "leave", ["leave", "drop_rows", "constant:Unknown"], f"'{col}' is an identifier / free text; there is no sound way to infer missing values.", False


def build_plan(prof: Profile, anom: AnomalyResult) -> list[Action]:
    plan: list[Action] = []
    kinds, typed = prof.kinds, prof.typed

    for c, info in prof.numeric_text.items():
        plan.append(Action(
            f"coerce:{c}", STEP["coerce"], "coerce_numeric", c,
            f"Convert '{c}' from text to numbers",
            f"'{c}' is stored as text (currency symbols, thousands separators, brackets). {info['n_failed']} value(s) cannot be read as numbers and will become missing.",
            info["n_text"], default_on=True))

    for c, info in prof.dates.items():
        fam = info["families"]
        warn = ""
        if info["ambiguous"]:
            warn = " WARNING: every DD/MM vs MM/DD value is ambiguous (no day > 12 seen); day-first was assumed."
        plan.append(Action(
            f"dates:{c}", STEP["dates"], "parse_dates", c,
            f"Standardise dates in '{c}' to YYYY-MM-DD",
            f"'{c}' mixes {len(fam)} date format(s): " + "; ".join(f"{k.strip()} (e.g. {v['example']})" for k, v in fam.items())
            + f". {info['unparsed']} value(s) are not valid dates." + warn,
            int(typed[c].notna().sum()), default_on=True))

    for c, info in prof.categories.items():
        if info["mapping"]:
            examples = "; ".join(f"{g['canonical']} ← {', '.join(repr(v) for v, _ in g['variants'] if v != g['canonical'])}" for g in info["groups"][:4])
            plan.append(Action(
                f"categories:{c}", STEP["categories"], "standardise_categories", c,
                f"Standardise spelling in '{c}' ({info['n_distinct_before']} → {info['n_distinct_after']} distinct values)",
                f"Variants that differ only by case, spacing or punctuation were merged into the most frequent spelling. {examples}",
                info["n_variant_cells"], params={"mapping": info["mapping"]}, default_on=True))
        if info["fuzzy_mapping"]:
            pairs = "; ".join(f"'{a}' → '{b}' ({n} rows)" for a, b, n in info["fuzzy_pairs"][:5])
            plan.append(Action(
                f"fuzzy:{c}", STEP["fuzzy"], "fuzzy_categories", c,
                f"Merge likely typos in '{c}'",
                f"Rare spellings very similar to a much more common one: {pairs}. Off by default because a similar-looking value can still be a different, valid category.",
                info["fuzzy_cells"], params={"mapping": info["fuzzy_mapping"]}, default_on=False))

    if prof.dup_after_standardise > 0:
        has_key = prof.key_col is not None
        plan.append(Action(
            "dedupe", STEP["dedupe"], "drop_duplicates", None,
            f"Remove {prof.dup_after_standardise:,} duplicate rows",
            f"{prof.dup_exact:,} rows are exact duplicates; {prof.dup_after_standardise - prof.dup_exact:,} more become identical once spelling/format differences are standardised. "
            + (f"The ID column '{prof.key_col}' makes repeats clearly erroneous, so the first occurrence is kept."
               if has_key else "No ID column exists, so identical rows might be genuine repeat transactions - review before approving."),
            prof.dup_after_standardise, default_on=has_key))

    f = anom.findings
    if len(f):
        errs = f[f.verdict == "error"]
        label = {"impossible": "Impossible values", "sentinel": "Placeholder/sentinel codes", "relationship": "Values that break a learned relationship"}
        for (issue, col), g in errs.groupby(["issue", "column"]):
            n_derive = int((g.fix_kind == "derive").sum())
            fixes = [{"row": r, "value": (None if pd.isna(v) else float(v))} for r, v in zip(g.row, g.fix)]
            reasons = "; ".join(g.reason.head(2).tolist())
            plan.append(Action(
                f"fix:{issue}:{col}", STEP["fix"], "fix_values", col,
                f"{label.get(issue, issue)} in '{col}' ({len(g)})",
                f"{reasons}. {n_derive} can be repaired exactly from the other columns of the same row; the rest are blanked and then imputed.",
                len(g), params={"fixes": fixes}, default_on=True, strategy="apply_suggested",
                options=["apply_suggested", "set_missing", "leave"]))
        rev = f[f.verdict == "review"]
        for col, g in rev.groupby("column"):
            plan.append(Action(
                f"review:{col}", STEP["review"], "review_outliers", col,
                f"Review {len(g)} suspicious value(s) in '{col}'",
                "Statistically extreme and NOT explained by any relationship in the row. They might be real (so nothing changes by default) or typos. "
                "Approve to blank them (then imputed) or cap them at the plausible boundary.",
                len(g), params={"rows": g.row.tolist(), "caps": [None if pd.isna(x) else float(x) for x in g.cap]},
                default_on=False, strategy="set_missing", options=["set_missing", "cap", "leave"]))

    # imputation, one action per column with missing values
    for _, row in prof.missing.iterrows():
        c, nm, kind = row["column"], int(row["n_missing"]), row["kind"]
        if nm == 0:
            continue
        s = typed[c]
        ctx = anom.contexts.get(c)
        f_c = f[f.column == c] if len(f) else f
        out_share = float(((f_c.verdict != "legit") if len(f_c) else pd.Series(dtype=bool)).sum()) / max(prof.n_rows, 1)
        strat, opts, why, on = _recommend_imputation(c, kind, s, row["pct_missing"], ctx, anom.relationships, out_share)
        plan.append(Action(
            f"impute:{c}", STEP["impute"], "impute", c,
            f"Fill {nm:,} missing value(s) in '{c}' ({row['pct_missing']:.1f}%)", why, nm,
            params={"group_col": ctx[0] if ctx else None}, default_on=on, strategy=strat, options=opts))
    return plan


# --------------------------------------------------------------------------- applying the plan
@dataclass
class CleanResult:
    cleaned: pd.DataFrame
    action_log: pd.DataFrame
    changes: pd.DataFrame
    changes_truncated: bool


def _same(a, b):
    both_nan = pd.isna(a) & pd.isna(b)
    try:
        eq = (a == b) | both_nan
    except Exception:
        eq = both_nan
    return eq


def apply_plan(df: pd.DataFrame, prof: Profile, anom: AnomalyResult, plan: list[Action],
               approved: set[str], strategies: dict | None = None) -> CleanResult:
    strategies = strategies or {}
    out = df.copy()
    changes, log = [], []
    state = {"n": 0, "trunc": False}

    def record(action, col, before: pd.Series, after: pd.Series):
        diff = ~_same(before, after)
        k = int(diff.sum())
        if k and state["n"] < CHANGE_LOG_CAP:
            idxs = before.index[diff][: CHANGE_LOG_CAP - state["n"]]
            for i in idxs:
                changes.append({"action": action.id, "row": i, "column": col, "old": before.loc[i], "new": after.loc[i]})
            state["n"] += len(idxs)
            state["trunc"] = state["trunc"] or k > len(idxs)
        return k

    for a in sorted(plan, key=lambda a: (a.step, a.id)):
        if a.id not in approved:
            continue
        strat = strategies.get(a.id, a.strategy)
        n_changed, note = 0, ""
        if a.kind == "coerce_numeric":
            before = out[a.column]
            out[a.column] = clean_number_strings(before)
            n_changed = record(a, a.column, before, out[a.column])
        elif a.kind == "parse_dates":
            before = out[a.column]
            parsed, _ = parse_dates_flexible(before)
            out[a.column] = parsed
            n_changed = int(before.notna().sum())
            for i in before.index[before.notna() & parsed.isna()][:50]:
                changes.append({"action": a.id, "row": i, "column": a.column, "old": before.loc[i], "new": "(unparseable → blank)"})
        elif a.kind in ("standardise_categories", "fuzzy_categories"):
            before = out[a.column]
            m = a.params["mapping"]
            after = before.map(lambda v, m=m: m.get(v, v) if isinstance(v, str) else v)
            out[a.column] = after
            n_changed = record(a, a.column, before, after)
        elif a.kind == "drop_duplicates":
            std = standardised_view(out, prof.kinds, prof.categories)
            dup = std.duplicated(keep="first")
            for i in out.index[dup][: max(0, CHANGE_LOG_CAP - state["n"])]:
                changes.append({"action": a.id, "row": i, "column": "(row)", "old": "duplicate row", "new": "removed"})
            state["n"] += int(min(dup.sum(), max(0, CHANGE_LOG_CAP - state["n"])))
            n_changed = int(dup.sum())
            out = out.loc[~dup]
        elif a.kind == "fix_values" and strat != "leave":
            s = out[a.column].astype("float64")
            before = s.copy()
            for fx in a.params["fixes"]:
                if fx["row"] not in s.index:
                    continue
                if strat == "apply_suggested" and fx["value"] is not None:
                    s.loc[fx["row"]] = fx["value"]
                else:
                    s.loc[fx["row"]] = np.nan
            out[a.column] = s
            n_changed = record(a, a.column, before, s)
        elif a.kind == "review_outliers" and strat != "leave":
            s = out[a.column].astype("float64")
            before = s.copy()
            for r, cap in zip(a.params["rows"], a.params["caps"]):
                if r not in s.index:
                    continue
                s.loc[r] = cap if (strat == "cap" and cap is not None) else np.nan
            out[a.column] = s
            n_changed = record(a, a.column, before, s)
        elif a.kind == "impute" and strat != "leave":
            n_changed, note = _impute(out, a, strat, prof, anom, record)
        log.append({"action": a.id, "title": a.title, "strategy": strat or "", "cells_or_rows_changed": n_changed, "note": note})

    # restore integer-looking columns
    for c, k in prof.kinds.items():
        if k["kind"] == "numeric" and k.get("int_like") and c in out.columns:
            x = out[c]
            if pd.api.types.is_numeric_dtype(x):
                nn = x.dropna()
                if len(nn) and np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0):
                    out[c] = x.round().astype("Int64")
    return CleanResult(out, pd.DataFrame(log), pd.DataFrame(changes), state["trunc"])


def _impute(out: pd.DataFrame, a: Action, strat: str, prof: Profile, anom: AnomalyResult, record):
    c = a.column
    kind = prof.kinds[c]["kind"]
    before = out[c].copy()
    s = out[c]
    note = ""
    if strat == "drop_rows":
        miss = s.isna()
        n = int(miss.sum())
        out.drop(index=out.index[miss], inplace=True)
        return n, "rows removed"
    if kind == "numeric":
        s = s.astype("float64")
        if strat == "derive":
            for r in anom.relationships:
                if c not in (r.target, r.a, r.b):
                    continue
                A, B, C = (out[x].astype("float64") if pd.api.types.is_numeric_dtype(out[x]) else clean_number_strings(out[x]) for x in (r.a, r.b, r.target))
                if c == r.target:
                    val = r.compute(A, B)
                elif c == r.a:
                    val = C / B.replace(0, np.nan) if r.op == "prod" else C - B
                else:
                    val = C / A.replace(0, np.nan) if r.op == "prod" else C - A
                val = val.round(0 if prof.kinds[c].get("int_like") else 2)
                s = s.fillna(val)
            strat = "group_median" if a.params.get("group_col") else "median"
        if strat == "group_median" and a.params.get("group_col") and a.params["group_col"] in out.columns:
            g = norm_series(out[a.params["group_col"]])
            s = s.fillna(s.groupby(g).transform("median"))
            strat = "median"
        if strat == "median":
            s = s.fillna(s.median())
        elif strat == "mean":
            s = s.fillna(s.mean())
        if prof.kinds[c].get("int_like"):
            s = s.where(s.isna(), s.round())
        out[c] = s
    elif kind in ("categorical", "boolean", "text", "id"):
        if strat == "mode":
            m = s.mode(dropna=True)
            if len(m):
                out[c] = s.fillna(m.iloc[0])
        elif strat.startswith("constant:"):
            out[c] = s.fillna(strat.split(":", 1)[1])
    return record(a, c, before, out[c]), note


# --------------------------------------------------------------------------- headline report
def headline(prof: Profile, anom: AnomalyResult) -> dict:
    f = anom.findings
    n_err = int((f.verdict == "error").sum()) if len(f) else 0
    n_rev = int((f.verdict == "review").sum()) if len(f) else 0
    n_legit = int(((f.verdict == "legit") & (f.issue == "extreme_but_consistent")).sum()) if len(f) else 0
    cat_cols = {c: i for c, i in prof.categories.items() if i["mapping"]}
    mixed_dates = {c: len(i["families"]) for c, i in prof.dates.items() if len(i["families"]) > 1}
    return {
        "rows": prof.n_rows, "cols": prof.n_cols,
        "missing_pct": round(prof.missing_pct_cells, 2), "missing_cells": prof.n_missing_cells,
        "duplicate_rows_exact": prof.dup_exact, "duplicate_rows": prof.dup_after_standardise,
        "suspicious_values": n_err + n_rev, "likely_errors": n_err, "needs_review": n_rev,
        "unusual_but_valid_kept": n_legit,
        "inconsistent_category_columns": len(cat_cols),
        "inconsistent_category_variants": int(sum(len(i["mapping"]) for i in cat_cols.values())),
        "mixed_date_columns": mixed_dates,
        "relationships": [r.text() for r in anom.relationships],
        "text_numeric_columns": list(prof.numeric_text),
    }


def headline_text(h: dict) -> str:
    lines = [
        f"{h['missing_pct']}% of cells missing ({h['missing_cells']:,} cells).",
        f"{h['duplicate_rows']:,} duplicate rows identified ({h['duplicate_rows_exact']:,} exact + {h['duplicate_rows'] - h['duplicate_rows_exact']:,} hidden by spelling/format differences).",
        f"{h['suspicious_values']:,} suspicious numerical values flagged ({h['likely_errors']:,} likely errors, {h['needs_review']:,} for review).",
        f"{h['unusual_but_valid_kept']:,} extreme-but-consistent values recognised as legitimate and kept.",
        f"{h['inconsistent_category_variants']} inconsistent category spellings detected across {h['inconsistent_category_columns']} column(s).",
    ]
    if h["mixed_date_columns"]:
        lines.append("Mixed date formats: " + ", ".join(f"{c} ({n} formats)" for c, n in h["mixed_date_columns"].items()) + ".")
    if h["relationships"]:
        lines.append("Learned relationships: " + "; ".join(h["relationships"]) + ".")
    return "\n".join(f"- {l}" for l in lines)


def default_approved(plan: list[Action]) -> set[str]:
    return {a.id for a in plan if a.default_on}


def run_all(df: pd.DataFrame, **detect_kwargs):
    """Convenience: profile -> detect -> plan. Returns (profile, anomaly, plan)."""
    from .profiling import profile
    from .anomaly import detect
    prof = profile(df)
    anom = detect(prof.typed, prof.kinds, **detect_kwargs)
    return prof, anom, build_plan(prof, anom)
