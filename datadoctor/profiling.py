"""Stage 1 - rule-based profiling.

Type inference, missing values, duplicates, category standardisation, mixed date
formats, numeric-stored-as-text and name-based range hints. Pure pandas/numpy.
"""
from __future__ import annotations

import difflib
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

STRICT_ID_TOKENS = {"id", "uuid", "guid", "key", "ref", "sku", "index"}
SOFT_ID_TOKENS = {"code", "no", "number", "num"}
NONNEG_TOKENS = {
    "qty", "quantity", "units", "price", "amount", "total", "sales", "revenue", "cost",
    "count", "stock", "weight", "days", "distance", "salary", "income", "mrp",
}
SIGNED_TOKENS = {
    "profit", "loss", "balance", "change", "delta", "diff", "net", "temperature", "temp",
    "lat", "latitude", "lon", "lng", "longitude", "return", "growth", "variance", "margin",
}
EXTRA_NA = ["?", "-", "--", "none", "None", "missing", "n.a.", "#N/A"]


# --------------------------------------------------------------------------- helpers
def tokens(name) -> set[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return set(re.findall(r"[a-z]+", s.lower()))


def is_texty(s: pd.Series) -> bool:
    return not (is_numeric_dtype(s) or is_datetime64_any_dtype(s) or is_bool_dtype(s))


def norm_key(v) -> str:
    s = str(v).lower().strip()
    return re.sub(r"[\W_]+", " ", s).strip()


def norm_series(s: pd.Series) -> pd.Series:
    """Normalised category key per cell (case/whitespace/punctuation-insensitive)."""
    mapping = {u: norm_key(u) for u in s.dropna().unique()}
    return s.map(mapping)


def range_hint(name, s: pd.Series | None = None):
    """Return (lo, hi, why) or None. Guarded: a 'non-negative' hint is dropped if
    the column data clearly contradicts it (>=10% negative values)."""
    t = tokens(name)
    if t & SIGNED_TOKENS:
        return None
    if "age" in t:
        return (0.0, 120.0, "age must be between 0 and 120")
    if t & {"pct", "percent", "percentage", "discount"}:
        return (0.0, 100.0, "a percentage must be between 0 and 100")
    if t & NONNEG_TOKENS:
        if s is not None:
            x = pd.to_numeric(s, errors="coerce").dropna()
            if len(x) and (x < 0).mean() >= 0.10:
                return None
        return (0.0, None, f"'{name}' cannot be negative")
    return None


def clean_number_strings(s: pd.Series) -> pd.Series:
    """'₹ 1,200.50', '(45)', '12%', 'Rs. 300' -> floats; unparseable -> NaN."""
    t = s.astype("string").str.strip()
    neg = t.str.fullmatch(r"\(.*\)").fillna(False).astype(bool).to_numpy()
    t = t.str.replace(r"(?i)^(rs\.?|inr|usd|eur|gbp)\s*", "", regex=True)
    t = t.str.replace(r"[()₹$€£\s,%]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    arr = pd.Series(out).to_numpy(dtype="float64", na_value=np.nan)
    arr = np.where(neg, -arr, arr)
    return pd.Series(arr, index=s.index, name=s.name)


# --------------------------------------------------------------------------- dates
_SLASH = re.compile(r"^9{1,2}(?P<sep>[/\-.])9{1,2}(?P=sep)9{4}$")


def _shape(t: pd.Series) -> pd.Series:
    return t.str.replace(r"\d", "9", regex=True).str.replace(r"[A-Za-z]+", "a", regex=True)


def parse_dates_flexible(s: pd.Series):
    """Parse a column holding dates in several formats.

    Numeric 'DD/MM/YYYY' vs 'MM/DD/YYYY' is resolved from the data: any value with a
    first part > 12 proves day-first, any second part > 12 proves month-first.
    If nothing proves it, day-first is assumed and `ambiguous` is reported.
    """
    info = {"families": {}, "ambiguous": False, "dayfirst_assumed": None, "unparsed": 0}
    if is_datetime64_any_dtype(s):
        return s, info
    t = s.astype("string").str.strip()
    ok = (t.notna() & (t != "")).to_numpy()
    if not ok.any():
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]"), info
    shape = _shape(t)
    pieces, slash_parts = [], []
    for sh, idx in shape[ok].groupby(shape[ok]).groups.items():
        sub = t.loc[idx]
        if sh.startswith("9999-99-99") or sh.startswith("9999/99/99"):
            fam = "ISO  YYYY-MM-DD"
            pieces.append(pd.to_datetime(sub.str.replace("/", "-", regex=False), format="ISO8601", errors="coerce"))
        elif _SLASH.match(sh):
            sep = _SLASH.match(sh).group("sep")
            fam = f"numeric with '{sep}'  (DD{sep}MM{sep}YYYY or MM{sep}DD{sep}YYYY)"
            slash_parts.append(sub)
        else:
            fam = "textual month  (e.g. 04 Mar 2026)"
            pieces.append(pd.to_datetime(sub, format="mixed", dayfirst=True, errors="coerce"))
        f = info["families"].setdefault(fam, {"count": 0, "example": str(sub.iloc[0])})
        f["count"] += int(len(sub))
    if slash_parts:
        sl = pd.concat(slash_parts)
        parts = sl.str.split(r"[/\-.]", expand=True)
        first = pd.to_numeric(parts[0], errors="coerce").to_numpy()
        second = pd.to_numeric(parts[1], errors="coerce").to_numpy()
        year = pd.to_numeric(parts[2], errors="coerce").to_numpy()
        n_df, n_mf = int((first > 12).sum()), int((second > 12).sum())
        default_dayfirst = n_df >= n_mf
        info["ambiguous"] = (n_df == 0 and n_mf == 0)
        info["dayfirst_assumed"] = bool(default_dayfirst)
        dayfirst = np.where(first > 12, True, np.where(second > 12, False, default_dayfirst))
        month = np.where(dayfirst, second, first)
        day = np.where(dayfirst, first, second)
        pieces.append(pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}, index=sl.index), errors="coerce"))
    parsed = pd.concat(pieces).reindex(s.index).astype("datetime64[ns]")
    info["unparsed"] = int((ok & parsed.isna().to_numpy()).sum())
    return parsed, info


# --------------------------------------------------------------------------- IO
def load_table(src, name: str | None = None, sheet=0) -> pd.DataFrame:
    """Read CSV/XLSX from a path or a file-like object (e.g. Streamlit upload)."""
    name = name or getattr(src, "name", str(src))
    low = name.lower()
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(src, sheet_name=sheet, na_values=EXTRA_NA)
    else:
        raw = src.read() if hasattr(src, "read") else open(src, "rb").read()
        text = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        df = pd.read_csv(io.StringIO(text), na_values=EXTRA_NA)
        if df.shape[1] == 1:  # maybe ; or \t separated
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", na_values=EXTRA_NA)
    for c in df.columns:
        if is_texty(df[c]):
            blank = df[c].astype("string").str.fullmatch(r"\s*").fillna(False).astype(bool)
            if blank.any():
                df[c] = df[c].mask(blank, np.nan)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- kinds
def infer_kind(s: pd.Series, name: str) -> dict:
    nn = s.dropna()
    n = len(nn)
    t = tokens(name)
    if n == 0:
        return {"kind": "empty"}
    if is_bool_dtype(s):
        return {"kind": "boolean"}
    if is_datetime64_any_dtype(s):
        return {"kind": "date", "needs_parse": False}
    uniq_ratio = nn.nunique() / n
    if t & STRICT_ID_TOKENS or ((t & SOFT_ID_TOKENS) and uniq_ratio >= 0.9):
        return {"kind": "id"}
    if is_numeric_dtype(s):
        return {"kind": "numeric", "needs_coercion": False}
    sample = nn.sample(min(n, 1000), random_state=0)
    if clean_number_strings(sample).notna().mean() >= 0.9:
        return {"kind": "numeric", "needs_coercion": True}
    parsed, _ = parse_dates_flexible(sample)
    if parsed.notna().mean() >= 0.9:
        return {"kind": "date", "needs_parse": True}
    nkeys = norm_series(nn).nunique()
    if nkeys <= max(50, 0.05 * n):
        return {"kind": "categorical"}
    if uniq_ratio >= 0.9:
        return {"kind": "id"}
    return {"kind": "text"}


def to_typed(df: pd.DataFrame, kinds: dict) -> pd.DataFrame:
    t = df.copy()
    for c, k in kinds.items():
        if k["kind"] == "numeric" and k.get("needs_coercion"):
            t[c] = clean_number_strings(df[c])
        elif k["kind"] == "date" and k.get("needs_parse"):
            t[c], _ = parse_dates_flexible(df[c])
    for c, k in kinds.items():
        if k["kind"] == "numeric":
            x = t[c].dropna()
            k["int_like"] = bool(len(x) and np.all(np.mod(x.to_numpy(dtype=float), 1) == 0))
    return t


# --------------------------------------------------------------------------- categories
def analyse_categories(s: pd.Series, fuzzy: bool = True, max_fuzzy_groups: int = 400) -> dict:
    vc = s.dropna().astype(str).value_counts()
    groups: dict[str, list] = {}
    for val, cnt in vc.items():
        groups.setdefault(norm_key(val), []).append((val, int(cnt)))
    mapping, group_info = {}, []
    for key, items in groups.items():
        if len(items) > 1:
            canon = max(items, key=lambda x: (x[1], x[0] == x[0].strip(), x[0] != x[0].lower(), x[0] != x[0].upper()))[0]
            for v, _ in items:
                if v != canon:
                    mapping[v] = canon
            group_info.append({"canonical": canon, "variants": [(v, c) for v, c in items], "rows": sum(c for _, c in items)})
        else:
            canon_by_key = items[0][0]
    n_variant_cells = int(sum(vc[v] for v in mapping))

    # fuzzy typo candidates: rare keys close to a much more frequent key
    fuzzy_map, fuzzy_pairs = {}, []
    if fuzzy and len(groups) <= max_fuzzy_groups:
        key_count = {k: sum(c for _, c in items) for k, items in groups.items()}
        key_canon = {}
        for k, items in groups.items():
            key_canon[k] = max(items, key=lambda x: (x[1], x[0] == x[0].strip()))[0]
        big = sorted(key_count, key=key_count.get, reverse=True)
        for k in big:
            if len(k) < 5:
                continue
            for b in big:
                if b == k or key_count[b] < 20 * key_count[k] or len(b) < 5:
                    continue
                if difflib.SequenceMatcher(None, k, b).ratio() >= 0.9:
                    for v, _ in groups[k]:
                        fuzzy_map[v] = key_canon[b]
                    fuzzy_pairs.append((key_canon[k], key_canon[b], key_count[k]))
                    break
    return {
        "mapping": mapping,
        "groups": group_info,
        "n_variant_cells": n_variant_cells,
        "n_distinct_before": int(len(vc)),
        "n_distinct_after": int(len(groups)),
        "fuzzy_mapping": fuzzy_map,
        "fuzzy_pairs": fuzzy_pairs,
        "fuzzy_cells": int(sum(vc.get(v, 0) for v in fuzzy_map)),
    }


# --------------------------------------------------------------------------- profile
@dataclass
class Profile:
    n_rows: int
    n_cols: int
    kinds: dict
    typed: pd.DataFrame
    missing: pd.DataFrame
    n_missing_cells: int
    missing_pct_cells: float
    dup_exact: int
    dup_after_standardise: int
    key_col: str | None
    key_conflicts: int
    categories: dict
    dates: dict
    numeric_text: dict
    notes: list = field(default_factory=list)


def standardised_view(typed: pd.DataFrame, kinds: dict, categories: dict) -> pd.DataFrame:
    std = typed.copy()
    for c, info in categories.items():
        if info["mapping"]:
            std[c] = std[c].map(lambda v, m=info["mapping"]: m.get(v, v) if isinstance(v, str) else v)
    for c, k in kinds.items():
        if k["kind"] in ("categorical", "text") and is_texty(std[c]):
            std[c] = std[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return std


def profile(df: pd.DataFrame, fuzzy: bool = True) -> Profile:
    n, m = df.shape
    kinds = {c: infer_kind(df[c], c) for c in df.columns}
    typed = to_typed(df, kinds)

    miss = pd.DataFrame(
        {
            "column": df.columns,
            "kind": [kinds[c]["kind"] for c in df.columns],
            "n_missing": [int(typed[c].isna().sum()) for c in df.columns],
        }
    )
    miss["pct_missing"] = miss["n_missing"] / max(n, 1) * 100
    n_missing = int(miss["n_missing"].sum())

    categories = {
        c: analyse_categories(typed[c], fuzzy=fuzzy)
        for c, k in kinds.items()
        if k["kind"] == "categorical"
    }
    dates, numeric_text, notes = {}, {}, []
    for c, k in kinds.items():
        if k["kind"] == "date" and k.get("needs_parse"):
            _, info = parse_dates_flexible(df[c])
            dates[c] = info
        if k["kind"] == "numeric" and k.get("needs_coercion"):
            raw_nn = df[c].notna().sum()
            numeric_text[c] = {"n_text": int(raw_nn), "n_failed": int(raw_nn - typed[c].notna().sum())}

    std = standardised_view(typed, kinds, categories)
    dup_exact = int(df.duplicated().sum())
    dup_std = int(std.duplicated().sum())

    key_col, key_conflicts = None, 0
    for c, k in kinds.items():
        if k["kind"] == "id" and typed[c].notna().mean() > 0.5 and typed[c].nunique() / max(typed[c].notna().sum(), 1) >= 0.5:
            key_col = c
            break
    if key_col is None:
        notes.append("No ID/key column detected: identical rows may be legitimate repeat records.")
    else:
        dupkey = std[key_col].notna() & std.duplicated(subset=[key_col], keep=False)
        full_dup = std.duplicated(keep=False)
        key_conflicts = int((dupkey & ~full_dup).sum())

    return Profile(
        n_rows=n, n_cols=m, kinds=kinds, typed=typed, missing=miss,
        n_missing_cells=n_missing, missing_pct_cells=n_missing / max(n * m, 1) * 100,
        dup_exact=dup_exact, dup_after_standardise=dup_std,
        key_col=key_col, key_conflicts=key_conflicts,
        categories=categories, dates=dates, numeric_text=numeric_text, notes=notes,
    )
