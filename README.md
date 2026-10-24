# DataDoctor AI

Upload a messy CSV/Excel file → get a data-quality report, **explained** cleaning recommendations you approve,
and a cleaned download. Unusual values are *judged*, not blindly deleted.

```bash
pip install -r requirements.txt
streamlit run app.py                       # interactive dashboard (sidebar has a sample dataset)
python -m datadoctor sample_data/messy_sales.csv   # CLI, applies the recommended defaults
python -m datadoctor.benchmark             # full benchmark -> results/
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| 1 Rules | `profiling.py` | type inference (numbers/dates stored as text), missing values, duplicates *after* standardisation, case/space/punctuation category variants, mixed date formats (DD/MM vs MM/DD resolved from the data), range hints (`age`, `price`≥0 …) |
| 2 Anomalies | `anomaly.py` | hard rules + sentinel codes (-999, 9999) · **learned relationships** (`total = quantity × unit_price`) · **context-aware robust z** (price judged within its product) · **Isolation Forest / LOF** corroboration on z-features. Verdicts: `error` / `review` / `legit` |
| 3 Recommendations | `plan.py` | per-column strategy with a written reason (derive from a relationship, group median, median vs mean by skew/outliers, mode vs "Unknown"). Nothing is applied until approved; every change is logged |
| 4 Dashboard | `app.py` | KPI cards, missing heatmap, outlier scatter, approval UI, before/after, logs, CSV/Excel download, benchmark tab |
| Benchmark | `benchmark.py`, `synth.py` | controlled corruption of clean data; precision/recall/F1 and damage to valid data |

Key idea: a bulk order of 150 units is a statistical outlier, but `total = quantity × unit_price` still holds, so it is **kept**.
A quantity of 50 whose total was computed from 5 breaks the rule, so it is an **error** and is repaired exactly (`quantity = total / unit_price`).

## Benchmark (5 seeds × 4 corruption rates, 5,000 rows each; full tables in `results/benchmark.md`)

Pooled over corruption rates 1–10%, row-level detection of injected numeric errors
(legitimate bulk orders are negatives):

| Method | F1 @3% | F1 @10% | Legit bulk rows flagged @3% |
|---|---|---|---|
| IQR rules | 0.21 | 0.47 | 99.6% |
| Isolation Forest (5% contamination) | 0.26 | 0.39 | 97.4% |
| Isolation Forest (told the true error rate) | 0.21 | 0.45 | 92.0% |
| LOF | 0.48 | 0.59 | 13.4% |
| DataDoctor without relationship learning | 0.60 | 0.78 | 99.8% |
| **DataDoctor combined** | **0.96** | **1.00** | **4.9%** |

End-to-end cleaning: a naive `dropna + drop_duplicates + drop IQR outliers` deletes **26.7% of valid orders** and leaves
~58 duplicates per run; DataDoctor's default approvals delete **0%**, damage **0%** of valid cells, fix 100% of spelling
variants and mixed-format dates, and restore 67% of injected error cells exactly (the rest are blanked and imputed).

## Honest caveats

- The data is **synthetic and built with structure the method exploits** (an exact arithmetic relationship, a product-dependent price).
  The ablation row shows the gain comes mostly from relationship learning; on data with no such structure expect results closer to
  "DataDoctor without relationships". Test on real labelled corruptions before trusting the headline numbers.
- Thresholds (z = 3.5, extreme = 8, log-scaling rule, implied-value plausibility check) were tuned on dev seeds 100–101; reported seeds are 0–4.
- Isolation Forest's weak showing is partly structural: legitimate bulk orders are the *most isolated* points, so raw IF flags them first.
- Imputation for non-derivable columns (age, shipping days) has ~25% median relative error - medians are safe, not accurate.
- Legit-row flags that remain are mostly bulk orders with a missing partner cell (cannot be verified) - routed to human review by design.
- Not implemented (the pitch's "later" items): supervised Random Forest on labelled errors, K-means distance scoring.
