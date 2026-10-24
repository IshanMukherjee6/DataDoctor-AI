"""Stage 2 - anomaly detection that separates *likely errors* from *unusual but legitimate*.

Signals combined per numeric cell:
  1. Hard rules         impossible values (range hints / user constraints), sentinel codes (-999, 9999 ...)
  2. Relationships      learned arithmetic structure, e.g. total = quantity x unit_price
  3. Context-aware z    robust z-score inside the group that explains the column (price by product)
  4. ML corroboration   Isolation Forest (or LOF) multivariate isolation score

Verdicts:
  error   very likely wrong (auto-fix proposed)
  review  suspicious, needs a human decision
  legit   unusual magnitude, but consistent with the rest of the row -> keep
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .profiling import norm_series, range_hint

SENTINELS = {999.0, 9999.0, 99999.0, 999999.0, 9999999.0}


# --------------------------------------------------------------------------- robust statistics
def _robust_scale(x: np.ndarray, med: float) -> float:
    dev = np.abs(x - med)
    mad = np.nanmedian(dev)
    scale = 1.4826 * mad
    if scale <= 0:
        scale = 1.2533 * np.nanmean(dev)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(x)
    return max(float(scale), 1e-9)


def _skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) == 0:
        return 0.0
    return float(np.mean(((x - x.mean()) / x.std()) ** 3))


def best_context(xt: np.ndarray, cat_normed: dict, min_group: int = 20, min_gain: float = 0.30):
    """Pick the categorical column that best explains xt (share of absolute deviation removed)."""
    ok = np.isfinite(xt)
    if ok.sum() < 100:
        return None, 0.0
    gmed = np.nanmedian(xt)
    base = np.abs(xt[ok] - gmed).sum()
    best, best_gain = None, 0.0
    for c, g in cat_normed.items():
        gs = pd.Series(g.to_numpy()[ok])
        if gs.nunique() < 2 or gs.nunique() > 30:
            continue
        s = pd.Series(xt[ok])
        med = s.groupby(gs.to_numpy()).transform("median")
        gain = 1 - np.abs(s - med).sum() / max(base, 1e-12)
        if gain > best_gain:
            best, best_gain = c, float(gain)
    if best_gain >= min_gain:
        return best, best_gain
    return None, best_gain


def contextual_z(xt: np.ndarray, groups: pd.Series | None, min_group: int = 20):
    """Robust z per cell (median/MAD inside group, global fallback). Returns z, center, scale."""
    n = len(xt)
    gmed = np.nanmedian(xt)
    gscale = _robust_scale(xt[np.isfinite(xt)], gmed)
    center = np.full(n, gmed)
    scale = np.full(n, gscale)
    if groups is not None:
        garr = groups.to_numpy()
        for g in pd.unique(garr[~pd.isna(garr)]):
            m = (garr == g) & np.isfinite(xt)
            if m.sum() >= min_group:
                med = np.nanmedian(xt[m])
                sc = _robust_scale(xt[m], med)
                sc = max(sc, 0.01 * abs(med))
                center[garr == g] = med
                scale[garr == g] = sc
    z = (xt - center) / scale
    return z, center, scale


# --------------------------------------------------------------------------- relationships
@dataclass
class Relationship:
    target: str
    a: str
    b: str
    op: str          # 'prod' or 'sum'
    support: float   # share of complete rows consistent with the relation

    def text(self) -> str:
        sym = "×" if self.op == "prod" else "+"
        return f"{self.target} = {self.a} {sym} {self.b}"

    def compute(self, a, b):
        return a * b if self.op == "prod" else a + b


def _consistent(c, ab):
    return np.isclose(c, ab, rtol=0.01, atol=0.02)


def find_relationships(typed: pd.DataFrame, cols: list[str], min_support: float = 0.85, max_cols: int = 14):
    cols = cols[:max_cols]
    X = {c: typed[c].to_numpy(dtype=float) for c in cols}
    found = []
    for target in cols:
        best = None
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if target in (a, b):
                    continue
                for op in ("prod", "sum"):
                    ab = X[a] * X[b] if op == "prod" else X[a] + X[b]
                    m = np.isfinite(X[target]) & np.isfinite(ab)
                    if m.sum() < 50:
                        continue
                    sup = float(_consistent(X[target][m], ab[m]).mean())
                    # guard against trivial identities (b == 1 almost always)
                    if sup >= min_support and (best is None or sup > best.support):
                        if len(np.unique(X[a][m])) > 3 and len(np.unique(X[b][m])) > 3:
                            best = Relationship(target, a, b, op, sup)
        if best is not None:
            found.append(best)
    # keep one relation per unordered triple (total=q*p and q=total/p are the same fact)
    seen, unique = set(), []
    for r in sorted(found, key=lambda r: -r.support):
        key = frozenset((r.target, r.a, r.b))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# --------------------------------------------------------------------------- ML scores
def ml_percentile(X: np.ndarray, method: str = "iforest", seed: int = 0) -> np.ndarray:
    """Anomaly percentile in [0,1] (1 = most anomalous) from an unsupervised model."""
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    if method == "lof":
        iqr = np.subtract(*np.nanpercentile(X, [75, 25], axis=0))
        iqr = np.where(iqr > 0, iqr, 1.0)
        Xs = (X - med) / iqr
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(len(Xs), size=min(len(Xs), 20000), replace=False)
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xs[fit_idx])
        score = -lof.score_samples(Xs)
    else:
        iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X)
        score = -iso.score_samples(X)
    return pd.Series(score).rank(pct=True).to_numpy()


# --------------------------------------------------------------------------- main detector
@dataclass
class AnomalyResult:
    findings: pd.DataFrame
    relationships: list
    contexts: dict
    log_scaled: list
    ml_pct: pd.Series
    num_cols: list
    method: str = "iforest"


FINDING_COLS = ["row", "column", "value", "issue", "verdict", "confidence", "z", "reason", "fix", "fix_kind", "cap"]


def detect(
    typed: pd.DataFrame,
    kinds: dict,
    constraints: dict | None = None,
    ml_method: str = "iforest",
    z_thresh: float = 3.5,
    strong_z: float = 8.0,
    ml_cut: float = 0.95,
    seed: int = 0,
    use_relationships: bool = True,
    use_context: bool = True,
    use_ml: bool = True,
) -> AnomalyResult:
    constraints = constraints or {}
    num_cols = [c for c, k in kinds.items() if k["kind"] == "numeric" and typed[c].nunique() > 5]
    cat_normed = {}
    if use_context:
        for c, k in kinds.items():
            if k["kind"] == "categorical":
                g = norm_series(typed[c])
                if 2 <= g.nunique() <= 30:
                    cat_normed[c] = g
    idx = typed.index
    n = len(typed)
    if not num_cols:
        return AnomalyResult(pd.DataFrame(columns=FINDING_COLS), [], {}, [], pd.Series(0.0, index=idx), [], ml_method)

    X = {c: typed[c].to_numpy(dtype=float) for c in num_cols}

    # ---- per-column robust z (log scale for positive, right-skewed columns; context-aware)
    zinfo, contexts, log_scaled = {}, {}, []
    for c in num_cols:
        x = X[c]
        pos = x[np.isfinite(x)]
        # log scale only for strictly positive, heavy right-tailed columns (robust: q95/median)
        use_log = (
            len(pos) > 0 and (pos > 0).mean() >= 0.95
            and np.nanpercentile(pos, 95) / max(np.nanmedian(pos), 1e-12) >= 3.0
        )
        xt = np.log(np.where(x > 0, x, np.nan)) if use_log else x.copy()
        if use_log:
            log_scaled.append(c)
        ctx, gain = best_context(xt, cat_normed) if cat_normed else (None, 0.0)
        if ctx:
            contexts[c] = (ctx, gain)
        z, center, scale = contextual_z(xt, cat_normed.get(ctx) if ctx else None)
        zinfo[c] = {"z": z, "center": center, "scale": scale, "log": use_log}

    # ---- relationships
    rels = find_relationships(typed, num_cols) if use_relationships else []
    rel_ok = {c: np.zeros(n, dtype=bool) for c in num_cols}
    rel_bad = []  # (rel, violating-row mask)
    for r in rels:
        a, b, c = X[r.a], X[r.b], X[r.target]
        ab = r.compute(a, b)
        complete = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        ok = complete & _consistent(c, ab)
        bad = complete & ~ok
        for col in (r.a, r.b, r.target):
            rel_ok[col] |= ok
        rel_bad.append((r, bad))
        # one *input* missing but target present: the implied value must be plausible in context
        for miss_col, known_col, other in ((r.a, r.b, r.a), (r.b, r.a, r.b)):
            if r.op == "prod":
                with np.errstate(all="ignore"):
                    implied = c / X[known_col]
            else:
                implied = c - X[known_col]
            sel = np.isfinite(c) & np.isfinite(X[known_col]) & ~np.isfinite(X[miss_col]) & np.isfinite(implied)
            if not sel.any():
                continue
            zi = zinfo[miss_col]
            with np.errstate(all="ignore"):
                it = np.log(np.where(implied > 0, implied, np.nan)) if zi["log"] else implied
                z_imp = (it - zi["center"]) / zi["scale"]
            plausible = sel & np.isfinite(z_imp) & (np.abs(z_imp) < z_thresh)
            rel_ok[known_col] |= plausible
            rel_ok[r.target] |= plausible

    # ---- ML score (features: context-aware robust z of each column + relationship log-residuals)
    feats = [np.clip(zinfo[c]["z"], -15, 15) for c in num_cols]
    for r in rels:
        with np.errstate(all="ignore"):
            ab = r.compute(X[r.a], X[r.b])
            ratio = np.log(np.abs(X[r.target]) + 1e-9) - np.log(np.abs(ab) + 1e-9)
        feats.append(np.clip(ratio, -5, 5))
    ml = ml_percentile(np.column_stack(feats), ml_method, seed) if use_ml else np.zeros(n)
    ml_pct = pd.Series(ml, index=idx)

    cells: dict[tuple, dict] = {}

    def add(i, col, issue, verdict, conf, reason, fix=np.nan, fix_kind="set_missing", cap=np.nan):
        key = (idx[i], col)
        if key in cells:
            return
        cells[key] = {
            "row": idx[i], "column": col, "value": X[col][i], "issue": issue, "verdict": verdict,
            "confidence": conf, "z": float(zinfo[col]["z"][i]) if np.isfinite(zinfo[col]["z"][i]) else np.nan,
            "reason": reason, "fix": fix, "fix_kind": fix_kind, "cap": cap,
        }

    # ---- 1. hard rules
    for c in num_cols:
        rng = constraints.get(c) or range_hint(c, typed[c])
        if rng is None:
            continue
        lo, hi = rng[0], rng[1]
        why = rng[2] if len(rng) > 2 else f"outside allowed range [{lo}, {hi}]"
        x = X[c]
        with np.errstate(invalid="ignore"):
            bad = np.zeros(n, dtype=bool)
            if lo is not None:
                bad |= x < lo
            if hi is not None:
                bad |= x > hi
        for i in np.flatnonzero(bad):
            add(i, c, "impossible", "error", 0.97, f"{x[i]:g}: {why}")

    # ---- 2. sentinel codes
    for c in num_cols:
        x = X[c]
        with np.errstate(invalid="ignore"):
            cand = np.isin(np.abs(x), list(SENTINELS)) & (np.abs(zinfo[c]["z"]) >= z_thresh)
        for i in np.flatnonzero(cand):
            add(i, c, "sentinel", "error", 0.95, f"{x[i]:g} looks like a placeholder/sentinel code, and is far from typical values")

    # ---- 3. relationship violations (attribute to the most suspicious member)
    for r, bad in rel_bad:
        for i in np.flatnonzero(bad):
            members = [r.target, r.a, r.b]
            flagged = [m for m in members if (idx[i], m) in cells]
            if flagged:
                suspect = flagged[0]
            else:
                zs = {m: abs(zinfo[m]["z"][i]) if np.isfinite(zinfo[m]["z"][i]) else 0.0 for m in members}
                suspect = max(zs, key=zs.get)
                if zs[suspect] < 2.5:
                    suspect = r.target
            av, bv, cv = X[r.a][i], X[r.b][i], X[r.target][i]
            if suspect == r.target:
                fix = r.compute(av, bv)
            elif suspect == r.a:
                fix = cv / bv if (r.op == "prod" and bv != 0) else (cv - bv if r.op == "sum" else np.nan)
            else:
                fix = cv / av if (r.op == "prod" and av != 0) else (cv - av if r.op == "sum" else np.nan)
            if np.isfinite(fix):
                fix = float(round(fix)) if (kinds[suspect].get("int_like") and np.isfinite(fix)) else float(round(fix, 2))
                rng = constraints.get(suspect) or range_hint(suspect, typed[suspect])
                if rng and ((rng[0] is not None and fix < rng[0]) or (rng[1] is not None and fix > rng[1])):
                    fix = np.nan
            reason = f"breaks learned rule {r.text()} (holds in {r.support:.0%} of rows); '{suspect}' is the most suspicious value"
            if (idx[i], suspect) in cells:  # already a hard error: upgrade its fix with the derived value
                if np.isfinite(fix):
                    cells[(idx[i], suspect)]["fix"] = fix
                    cells[(idx[i], suspect)]["fix_kind"] = "derive"
                    cells[(idx[i], suspect)]["reason"] += f"; recoverable from {r.text()}"
                continue
            add(i, suspect, "relationship", "error", 0.9, reason, fix=fix, fix_kind="derive" if np.isfinite(fix) else "set_missing")

    # ---- 4. statistical outliers, judged in context
    for c in num_cols:
        z = zinfo[c]["z"]
        info = zinfo[c]
        out = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= z_thresh))
        for i in out:
            if (idx[i], c) in cells:
                continue
            bound = info["center"][i] + np.sign(z[i]) * z_thresh * info["scale"][i]
            cap = float(np.exp(bound)) if info["log"] else float(bound)
            where = f" within {contexts[c][0]}" if c in contexts else ""
            base = f"|z|={abs(z[i]):.1f}{where}"
            if rel_ok[c][i]:
                add(i, c, "extreme_but_consistent", "legit", 0.9,
                    f"{base}, but the row satisfies its learned arithmetic relationship, so the value is plausible", fix_kind="keep")
            elif abs(z[i]) >= strong_z:
                add(i, c, "outlier", "review", 0.65, f"{base}: extreme and not explained by any relationship in the row", cap=cap)
            elif ml[i] >= ml_cut:
                add(i, c, "outlier", "review", 0.55, f"{base} and also isolated by the {ml_method} model (top {100*(1-ml[i]):.0f}% most unusual rows)", cap=cap)
            else:
                add(i, c, "mild_outlier", "legit", 0.6, f"{base}: moderately unusual, not corroborated by the multivariate model", fix_kind="keep")

    # ---- 5. multivariate-only anomalies (ML signal with no cell-level explanation)
    touched_rows = {k[0] for k in cells}
    pos = {lab: i for i, lab in enumerate(idx)}
    for i in np.flatnonzero(ml >= 0.99):
        if idx[i] in touched_rows:
            continue
        zs = {c: abs(zinfo[c]["z"][i]) if np.isfinite(zinfo[c]["z"][i]) else 0.0 for c in num_cols}
        top = max(zs, key=zs.get)
        if zs[top] >= 2.5:
            add(i, top, "multivariate", "review", 0.5,
                f"row is in the top 1% most unusual combinations ({ml_method}); '{top}' contributes most (|z|={zs[top]:.1f})",
                cap=float(np.exp(zinfo[top]["center"][i] + np.sign(zinfo[top]["z"][i]) * z_thresh * zinfo[top]["scale"][i])) if zinfo[top]["log"]
                else float(zinfo[top]["center"][i] + np.sign(zinfo[top]["z"][i]) * z_thresh * zinfo[top]["scale"][i]))

    findings = pd.DataFrame(list(cells.values()), columns=FINDING_COLS)
    return AnomalyResult(findings, rels, contexts, log_scaled, ml_pct, num_cols, ml_method)


# --------------------------------------------------------------------------- baselines (for benchmarking)
def baseline_stat_rows(typed: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.Series:
    """Textbook Tukey IQR fences per column. Row flagged if any value is outside."""
    flag = pd.Series(False, index=typed.index)
    for c in cols:
        x = typed[c]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        flag |= (x < q1 - k * iqr) | (x > q3 + k * iqr)
    return flag


def baseline_ml_rows(typed: pd.DataFrame, cols: list[str], method: str = "iforest", contamination: float = 0.05, seed: int = 0) -> pd.Series:
    """Plain unsupervised detector on raw numeric columns, flagging the top `contamination` share."""
    pct = ml_percentile(typed[cols].to_numpy(dtype=float), method, seed)
    return pd.Series(pct >= 1 - contamination, index=typed.index)
