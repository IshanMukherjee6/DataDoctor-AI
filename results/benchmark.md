# DataDoctor benchmark results

Synthetic sales data, 5,000 clean rows per run, 5 seeds (0, 1, 2, 3, 4); mean ± std across seeds.
Each run injects: numeric errors (5 types, equal share), missing cells, exact+cosmetic duplicates, spelling variants and mixed date formats; ~2% of rows are *legitimate* bulk orders (extreme but valid).

## A. Error-row detection (row is positive if a numeric error was injected)

**Corruption rate 1% of rows**

| Method | Precision | Recall | F1 | Legit bulk rows flagged % | Clean rows flagged % |
|---|---|---|---|---|---|
| IQR rules | 0.043 ± 0.004 | 0.772 ± 0.039 | 0.082 ± 0.007 | 99.8 ± 0.4 | 15.1 ± 0.7 |
| Isolation Forest (plain) | 0.062 ± 0.008 | 0.320 ± 0.042 | 0.104 ± 0.014 | 97.6 ± 1.4 | 2.8 ± 0.3 |
| Isolation Forest (oracle rate) | 0.008 ± 0.011 | 0.008 ± 0.011 | 0.008 ± 0.011 | 48.2 ± 7.7 | 0.0 ± 0.0 |
| LOF (plain) | 0.163 ± 0.006 | 0.840 ± 0.032 | 0.273 ± 0.010 | 14.1 ± 4.4 | 4.0 ± 0.1 |
| DataDoctor w/o relationships | 0.230 ± 0.020 | 0.776 ± 0.017 | 0.355 ± 0.023 | 99.6 ± 0.5 | 0.5 ± 0.1 |
| DataDoctor (combined) | 0.746 ± 0.047 | 1.000 ± 0.000 | 0.854 ± 0.031 | 3.0 ± 1.8 | 0.3 ± 0.1 |

**Corruption rate 3% of rows**

| Method | Precision | Recall | F1 | Legit bulk rows flagged % | Clean rows flagged % |
|---|---|---|---|---|---|
| IQR rules | 0.124 ± 0.007 | 0.785 ± 0.019 | 0.213 ± 0.011 | 99.6 ± 0.5 | 14.9 ± 0.8 |
| Isolation Forest (plain) | 0.209 ± 0.037 | 0.359 ± 0.064 | 0.264 ± 0.047 | 97.4 ± 1.3 | 2.0 ± 0.3 |
| Isolation Forest (oracle rate) | 0.213 ± 0.042 | 0.215 ± 0.043 | 0.214 ± 0.042 | 92.0 ± 2.7 | 0.4 ± 0.2 |
| LOF (plain) | 0.378 ± 0.029 | 0.651 ± 0.050 | 0.478 ± 0.037 | 13.4 ± 3.9 | 3.0 ± 0.1 |
| DataDoctor w/o relationships | 0.486 ± 0.030 | 0.777 ± 0.018 | 0.598 ± 0.028 | 99.8 ± 0.4 | 0.3 ± 0.1 |
| DataDoctor (combined) | 0.921 ± 0.031 | 0.997 ± 0.004 | 0.957 ± 0.016 | 4.9 ± 2.1 | 0.2 ± 0.1 |

**Corruption rate 5% of rows**

| Method | Precision | Recall | F1 | Legit bulk rows flagged % | Clean rows flagged % |
|---|---|---|---|---|---|
| IQR rules | 0.195 ± 0.007 | 0.797 ± 0.025 | 0.314 ± 0.010 | 99.6 ± 0.5 | 14.9 ± 0.6 |
| Isolation Forest (plain) | 0.378 ± 0.018 | 0.390 ± 0.019 | 0.384 ± 0.019 | 94.8 ± 3.1 | 1.2 ± 0.3 |
| Isolation Forest (oracle rate) | 0.381 ± 0.017 | 0.382 ± 0.017 | 0.382 ± 0.017 | 94.6 ± 2.9 | 1.1 ± 0.3 |
| LOF (plain) | 0.562 ± 0.035 | 0.580 ± 0.036 | 0.571 ± 0.036 | 11.1 ± 4.9 | 2.1 ± 0.1 |
| DataDoctor w/o relationships | 0.631 ± 0.027 | 0.779 ± 0.002 | 0.697 ± 0.017 | 99.7 ± 0.7 | 0.1 ± 0.1 |
| DataDoctor (combined) | 0.979 ± 0.006 | 1.000 ± 0.000 | 0.989 ± 0.003 | 3.5 ± 1.5 | 0.0 ± 0.0 |

**Corruption rate 10% of rows**

| Method | Precision | Recall | F1 | Legit bulk rows flagged % | Clean rows flagged % |
|---|---|---|---|---|---|
| IQR rules | 0.336 ± 0.014 | 0.790 ± 0.008 | 0.472 ± 0.015 | 100.0 ± 0.0 | 14.8 ± 0.7 |
| Isolation Forest (plain) | 0.570 ± 0.033 | 0.294 ± 0.017 | 0.388 ± 0.023 | 85.1 ± 8.0 | 0.4 ± 0.2 |
| Isolation Forest (oracle rate) | 0.450 ± 0.032 | 0.451 ± 0.032 | 0.451 ± 0.032 | 96.9 ± 2.9 | 3.8 ± 0.5 |
| LOF (plain) | 0.871 ± 0.022 | 0.449 ± 0.011 | 0.593 ± 0.015 | 6.0 ± 3.1 | 0.6 ± 0.2 |
| DataDoctor w/o relationships | 0.782 ± 0.028 | 0.768 ± 0.010 | 0.775 ± 0.016 | 99.3 ± 0.7 | 0.0 ± 0.0 |
| DataDoctor (combined) | 0.995 ± 0.004 | 1.000 ± 0.001 | 0.997 ± 0.002 | 2.3 ± 1.8 | 0.0 ± 0.0 |

## B. Recall by error type (all corruption rates pooled)

| Method | negative_quantity | magnitude_shift | sentinel_code | impossible_age | total_tampered |
|---|---|---|---|---|---|
| IQR rules | 1.00 | 0.89 | 0.82 | 1.00 | 0.22 |
| Isolation Forest (plain) | 0.06 | 0.56 | 0.56 | 0.46 | 0.06 |
| Isolation Forest (oracle rate) | 0.05 | 0.41 | 0.45 | 0.37 | 0.05 |
| LOF (plain) | 0.48 | 0.90 | 0.87 | 0.59 | 0.31 |
| DataDoctor w/o relationships | 1.00 | 0.88 | 0.99 | 1.00 | 0.01 |
| DataDoctor (combined) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## C. End-to-end cleaning: damage to valid data (all corruption rates pooled)

| Pipeline | Valid orders deleted % | Valid cells damaged % | Legit bulk-order cells damaged % | Error cells restored to truth % | Error cells left wrong % | Spelling variants fixed % | Mixed-format dates correct % | Residual duplicate rows |
|---|---|---|---|---|---|---|---|---|
| Naive (dropna + drop_duplicates + drop IQR outliers) | 26.7 | 0.0 | 0.0 | 0.0 | 21.4 | 0.0 | 100.0 | 58.5 |
| DataDoctor (default approvals) | 0.0 | 0.0 | 0.0 | 67.2 | 5.1 | 100.0 | 100.0 | 0.0 |
| DataDoctor (+ all review items approved) | 0.0 | 0.0 | 0.7 | 68.0 | 0.6 | 100.0 | 100.0 | 0.0 |

Imputation quality (injected-missing cells):

| Pipeline | Derivable columns (qty/price/total) within 1% of truth % | Other numeric columns: median relative error % |
|---|---|---|
| Naive (dropna + drop_duplicates + drop IQR outliers) | n/a | n/a |
| DataDoctor (default approvals) | 97.2 | 24.5 |
| DataDoctor (+ all review items approved) | 96.2 | 24.5 |