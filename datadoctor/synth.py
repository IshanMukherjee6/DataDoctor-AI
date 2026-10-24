"""Synthetic sales data + a controlled corruption injector.

`make_clean_sales` builds a clean dataset with realistic structure:
  * total = quantity * unit_price            (a cross-column relationship)
  * unit_price depends on the product        (context)
  * ~2% of orders are *legitimate* bulk orders with very large quantities
    (statistically extreme, but perfectly valid - the case naive cleaners destroy)

`corrupt` then injects known errors and returns ground truth so that
detection and cleaning can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRODUCTS = {
    "Laptop": 55000,
    "Smartphone": 22000,
    "Tablet": 18000,
    "Monitor": 12000,
    "Headphones": 3500,
    "Keyboard": 1500,
}
CITIES = ["Jamshedpur", "Ranchi", "Kolkata", "Patna", "Bhubaneswar", "Dhanbad"]
CAT_COLS = ["city", "product"]
NUM_COLS = ["quantity", "unit_price", "total", "customer_age", "shipping_days"]

ERROR_TYPES = [
    "negative_quantity",
    "magnitude_shift",
    "sentinel_code",
    "impossible_age",
    "total_tampered",
]


def make_clean_sales(n: int = 5000, seed: int = 0, bulk_rate: float = 0.02):
    """Return (clean_df, legit_extreme_mask)."""
    rng = np.random.default_rng(seed)
    order_id = [f"ORD-{i:06d}" for i in range(1, n + 1)]
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    city = rng.choice(CITIES, n, p=[0.30, 0.20, 0.20, 0.10, 0.10, 0.10])
    product = rng.choice(list(PRODUCTS), n, p=[0.12, 0.20, 0.12, 0.14, 0.22, 0.20])
    base = np.array([PRODUCTS[p] for p in product], dtype=float)
    unit_price = np.round(base * rng.lognormal(0, 0.08, n), 2)
    quantity = 1 + rng.poisson(3, n)
    bulk = rng.random(n) < bulk_rate
    quantity = np.where(bulk, rng.integers(60, 200, n), quantity).astype(int)
    total = np.round(quantity * unit_price, 2)
    customer_age = np.clip(np.round(rng.normal(38, 11, n)), 18, 80).astype(int)
    shipping_days = np.clip(rng.poisson(4, n) + 1, 1, 14).astype(int)
    df = pd.DataFrame(
        {
            "order_id": order_id,
            "order_date": pd.to_datetime(dates),
            "city": city,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": total,
            "customer_age": customer_age,
            "shipping_days": shipping_days,
        }
    )
    return df, pd.Series(bulk, index=df.index, name="legit_extreme")


@dataclass
class Truth:
    clean: pd.DataFrame          # true values, aligned to dirty index (copies share truth)
    err_cells: pd.DataFrame      # bool mask: injected numeric error
    missing_cells: pd.DataFrame  # bool mask: injected missing
    variant_cells: pd.DataFrame  # bool mask: injected category spelling variant
    datefmt_cells: pd.DataFrame  # bool mask: date written in a non-ISO format
    row_error_type: pd.Series    # '' or one of ERROR_TYPES
    legit_row: pd.Series         # statistically extreme but valid order
    dup_copy: pd.Series          # row is an injected duplicate of another row
    order_id: pd.Series


def _variant(v: str, rng) -> str:
    choice = rng.integers(0, 5)
    if choice == 0:
        return v.lower()
    if choice == 1:
        return v.upper()
    if choice == 2:
        return f" {v}"
    if choice == 3:
        return f"{v} "
    return v.lower() + "."


def corrupt(
    clean: pd.DataFrame,
    legit_mask: pd.Series,
    err_rate: float = 0.03,
    miss_rate: float = 0.04,
    dup_rate: float = 0.03,
    var_rate: float = 0.15,
    date_mixed: float = 0.40,
    seed: int = 0,
    error_types: list[str] | None = None,
):
    """Inject controlled corruption. Returns (dirty_df, Truth)."""
    rng = np.random.default_rng(seed + 10_000)
    error_types = error_types or ERROR_TYPES
    df = clean.copy()
    n = len(df)
    cols = list(df.columns)

    err = pd.DataFrame(False, index=df.index, columns=cols)
    row_type = pd.Series("", index=df.index, dtype="object")

    # ---- numeric errors (never on legit bulk orders, so classes stay disjoint)
    eligible = np.flatnonzero(~legit_mask.to_numpy())
    n_err = int(round(err_rate * n))
    err_rows = rng.choice(eligible, size=min(n_err, len(eligible)), replace=False)
    for i, r in enumerate(err_rows):
        etype = error_types[i % len(error_types)]
        lab = df.index[r]
        if etype == "negative_quantity":
            df.loc[lab, "quantity"] = -int(df.loc[lab, "quantity"])
            err.loc[lab, "quantity"] = True
        elif etype == "magnitude_shift":
            col = rng.choice(["quantity", "unit_price", "total", "shipping_days"])
            factor = rng.choice([10, 100])
            df.loc[lab, col] = df.loc[lab, col] * factor
            err.loc[lab, col] = True
        elif etype == "sentinel_code":
            col = rng.choice(["quantity", "unit_price", "customer_age", "shipping_days"])
            df.loc[lab, col] = rng.choice([-999, 9999, 99999])
            err.loc[lab, col] = True
        elif etype == "impossible_age":
            df.loc[lab, "customer_age"] = rng.choice([150, 212, 340, -3])
            err.loc[lab, "customer_age"] = True
        elif etype == "total_tampered":
            df.loc[lab, "total"] = round(float(df.loc[lab, "total"]) * rng.uniform(1.4, 3.0), 2)
            err.loc[lab, "total"] = True
        row_type.loc[lab] = etype

    # ---- missing values (random cells on rows that carry no numeric error)
    miss = pd.DataFrame(False, index=df.index, columns=cols)
    ok_rows = np.flatnonzero(row_type.to_numpy() == "")
    n_miss = int(round(miss_rate * n * 3))  # ~miss_rate of all candidate cells
    miss_cols = [c for c in cols if c != "order_id"]
    for _ in range(n_miss):
        r = rng.choice(ok_rows)
        c = rng.choice(miss_cols)
        miss.loc[df.index[r], c] = True
    # keep a column that remains float-capable: convert numeric to float before NaN
    for c in miss_cols:
        if miss[c].any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype("float64")
            df.loc[miss[c], c] = np.nan

    # ---- duplicates: exact copies of rows that are otherwise clean
    pool = np.flatnonzero((row_type.to_numpy() == "") & ~miss.any(axis=1).to_numpy())
    n_dup = int(round(dup_rate * n))
    dup_src = rng.choice(pool, size=min(n_dup, len(pool)), replace=False)

    truth_clean = clean.copy()
    legit = legit_mask.copy()
    base = {
        "df": df,
        "clean": truth_clean,
        "err": err,
        "miss": miss,
        "row_type": row_type,
        "legit": legit,
    }
    copies = {k: (v.iloc[dup_src].copy() if hasattr(v, "iloc") else v) for k, v in base.items()}
    dup_flag_orig = pd.Series(False, index=df.index)
    dup_flag_copy = pd.Series(True, index=copies["df"].index)

    all_df = pd.concat([df, copies["df"]], ignore_index=True)
    all_clean = pd.concat([truth_clean, copies["clean"]], ignore_index=True)
    all_err = pd.concat([err, copies["err"]], ignore_index=True)
    all_miss = pd.concat([miss, copies["miss"]], ignore_index=True)
    all_type = pd.concat([row_type, copies["row_type"]], ignore_index=True)
    all_legit = pd.concat([legit, copies["legit"]], ignore_index=True)
    all_dup = pd.concat([dup_flag_orig, dup_flag_copy], ignore_index=True)

    perm = rng.permutation(len(all_df))
    all_df, all_clean, all_err, all_miss, all_type, all_legit, all_dup = [
        x.iloc[perm].reset_index(drop=True)
        for x in (all_df, all_clean, all_err, all_miss, all_type, all_legit, all_dup)
    ]
    m = len(all_df)

    # ---- category spelling variants (applied AFTER duplication so copies can differ cosmetically)
    var = pd.DataFrame(False, index=all_df.index, columns=cols)
    for c in CAT_COLS:
        vals = all_df[c].astype(object).copy()
        hit = (rng.random(m) < var_rate) & vals.notna().to_numpy()
        for i in np.flatnonzero(hit):
            vals.iloc[i] = _variant(vals.iloc[i], rng)
        all_df[c] = vals.astype("string").astype(object)
        var[c] = hit

    # ---- mixed date formats
    datefmt = pd.DataFrame(False, index=all_df.index, columns=cols)
    d = pd.to_datetime(all_df["order_date"])
    out = []
    formats = ["%d/%m/%Y", "%d %b %Y", "%b %d, %Y"]
    for i in range(m):
        if pd.isna(d.iloc[i]):
            out.append(np.nan)
        elif rng.random() < date_mixed:
            out.append(d.iloc[i].strftime(formats[rng.integers(0, len(formats))]))
            datefmt.iloc[i, cols.index("order_date")] = True
        else:
            out.append(d.iloc[i].strftime("%Y-%m-%d"))
    all_df["order_date"] = pd.Series(out, index=all_df.index, dtype="object")

    truth = Truth(
        clean=all_clean,
        err_cells=all_err,
        missing_cells=all_miss,
        variant_cells=var,
        datefmt_cells=datefmt,
        row_error_type=all_type,
        legit_row=all_legit,
        dup_copy=all_dup,
        order_id=all_clean["order_id"],
    )
    return all_df, truth
