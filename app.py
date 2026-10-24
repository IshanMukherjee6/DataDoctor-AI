"""DataDoctor AI - Streamlit dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datadoctor.anomaly import detect
from datadoctor.plan import apply_plan, build_plan, default_approved, headline, headline_text
from datadoctor.profiling import load_table, profile
from datadoctor.synth import corrupt, make_clean_sales

st.set_page_config(page_title="DataDoctor AI", page_icon="🩺", layout="wide")

# validated categorical slots (blue, orange, aqua) + neutral for normal rows
C_NORMAL, C_ERROR, C_REVIEW, C_LEGIT = "#b8b7b0", "#eb6834", "#2a78d6", "#1baf7a"
VERDICT_STYLE = {
    "error": (C_ERROR, "x", "Likely error"),
    "review": (C_REVIEW, "diamond", "Needs review"),
    "legit": (C_LEGIT, "circle", "Unusual but valid (kept)"),
}


@st.cache_data(show_spinner=False)
def sample_data(n=20000):
    clean, legit = make_clean_sales(n, seed=7)
    dirty, _ = corrupt(clean, legit, err_rate=0.03, seed=7)
    return dirty


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 DataDoctor AI")
up = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls", "xlsm"])
use_sample = st.sidebar.button("…or load the messy sample sales data")
if use_sample:
    st.session_state["source"] = "sample"
if up is not None:
    st.session_state["source"] = "upload"

st.sidebar.markdown("**Anomaly detection**")
ml_method = st.sidebar.selectbox("ML corroboration model", ["iforest", "lof"], format_func=lambda x: {"iforest": "Isolation Forest", "lof": "Local Outlier Factor"}[x])
z_thresh = st.sidebar.slider("Outlier threshold (robust z)", 2.5, 6.0, 3.5, 0.5)
strong_z = st.sidebar.slider("'Extreme' threshold (robust z)", 5.0, 15.0, 8.0, 0.5)
use_rel = st.sidebar.checkbox("Learn cross-column relationships", True, help="e.g. total = quantity × unit_price. Lets extreme-but-consistent rows be kept.")
cons_txt = st.sidebar.text_area("Optional constraints (column:min:max per line)", placeholder="customer_age:18:100\nrating:1:5")

df = None
src = st.session_state.get("source")
if src == "upload" and up is not None:
    sheet = 0
    if up.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
        names = pd.ExcelFile(up).sheet_names
        sheet = st.sidebar.selectbox("Sheet", names)
        up.seek(0)
    df = load_table(up, up.name, sheet=sheet)
    sig_name = f"{up.name}:{sheet}"
elif src == "sample":
    df = sample_data()
    sig_name = "sample"

if df is None:
    st.title("DataDoctor AI")
    st.markdown("Upload a CSV/Excel file to get a data-quality report, explained cleaning recommendations you approve, "
                "and a cleaned file to download. The tool separates **likely errors** from **unusual-but-valid** values instead of deleting everything odd.")
    st.info("Use the sidebar to upload a file, or load the built-in messy sales sample.")
    st.stop()

constraints = {}
for line in cons_txt.splitlines():
    parts = [p.strip() for p in line.split(":")]
    if len(parts) == 3 and parts[0] in df.columns:
        try:
            constraints[parts[0]] = (float(parts[1]) if parts[1] else None, float(parts[2]) if parts[2] else None, f"outside user constraint [{parts[1]}, {parts[2]}]")
        except ValueError:
            pass

sig = f"{sig_name}|{df.shape}|{ml_method}|{z_thresh}|{strong_z}|{use_rel}|{json.dumps(constraints, default=str)}"
if st.session_state.get("sig") != sig:
    with st.spinner("Profiling and detecting anomalies…"):
        prof = profile(df)
        anom = detect(prof.typed, prof.kinds, constraints=constraints, ml_method=ml_method, z_thresh=z_thresh,
                      strong_z=strong_z, use_relationships=use_rel)
        plan = build_plan(prof, anom)
    st.session_state.update(sig=sig, prof=prof, anom=anom, plan=plan, result=None)
prof, anom, plan = st.session_state["prof"], st.session_state["anom"], st.session_state["plan"]
f = anom.findings
H = headline(prof, anom)
h = abs(hash(sig)) % 10**8

st.title("DataDoctor AI")
st.caption(f"{prof.n_rows:,} rows × {prof.n_cols} columns · detector: {ml_method}")

tabs = st.tabs(["Overview", "Missing values", "Anomalies", "Cleaning plan", "Before / after & download", "Benchmark"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Missing cells", f"{H['missing_pct']}%", f"{H['missing_cells']:,} cells", delta_color="off")
    c[1].metric("Duplicate rows", f"{H['duplicate_rows']:,}", f"{H['duplicate_rows_exact']:,} exact", delta_color="off")
    c[2].metric("Suspicious values", f"{H['suspicious_values']:,}", f"{H['likely_errors']} errors · {H['needs_review']} review", delta_color="off")
    c[3].metric("Inconsistent spellings", f"{H['inconsistent_category_variants']}", f"{H['inconsistent_category_columns']} column(s)", delta_color="off")
    c[4].metric("Valid extremes kept", f"{H['unusual_but_valid_kept']:,}", "not flagged", delta_color="off")
    st.markdown("#### Findings")
    st.markdown(headline_text(H))
    for n in prof.notes:
        st.warning(n)
    for col, info in prof.dates.items():
        if info["ambiguous"]:
            st.warning(f"Dates in '{col}': no value proves DD/MM vs MM/DD; day-first was assumed.")
    st.markdown("#### Column types")
    kt = pd.DataFrame({"column": list(prof.kinds), "detected type": [k["kind"] for k in prof.kinds.values()],
                       "missing %": prof.missing.set_index("column").pct_missing.round(2).reindex(list(prof.kinds)).values})
    st.dataframe(kt, hide_index=True, width="stretch")
    if anom.contexts:
        st.markdown("**Context-aware checks:** " + "; ".join(f"'{c}' judged within '{v[0]}' groups ({v[1]:.0%} of spread explained)" for c, v in anom.contexts.items()))

# ------------------------------------------------------------------ missing
with tabs[1]:
    m = prof.missing.sort_values("pct_missing", ascending=False)
    fig = go.Figure(go.Bar(x=m.pct_missing, y=m.column, orientation="h", marker_color=C_REVIEW))
    fig.update_layout(title="Missing values by column (%)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(fig, 320), width="stretch")
    nb = int(min(120, max(prof.n_rows // 20, 1)))
    blocks = np.array_split(np.arange(prof.n_rows), nb)
    mat = np.array([prof.typed.iloc[b].isna().mean().to_numpy() * 100 for b in blocks])
    fig = go.Figure(go.Heatmap(z=mat, x=list(prof.typed.columns), y=[f"{b[0]:,}" for b in blocks], colorscale=[[0, "#f4f3ef"], [1, C_REVIEW]],
                               colorbar=dict(title="% missing"), hovertemplate="rows from %{y}<br>%{x}: %{z:.1f}% missing<extra></extra>"))
    fig.update_layout(title="Missing-value heatmap (blocks of consecutive rows)", yaxis=dict(autorange="reversed", title="first row of block"))
    st.plotly_chart(style(fig, 460), width="stretch")

# ------------------------------------------------------------------ anomalies
with tabs[2]:
    if not len(f):
        st.success("No numeric anomalies found.")
    else:
        st.caption("Verdicts: **error** = impossible / broken rule, auto-fix proposed · **review** = extreme and unexplained, your call · "
                   "**legit** = unusual magnitude but consistent with its row, kept.")
        cols = anom.num_cols
        a, b = st.columns(2)
        xc = a.selectbox("X axis", cols, index=0)
        yc = b.selectbox("Y axis", cols, index=min(1, len(cols) - 1))
        rank = {"error": 3, "review": 2, "legit": 1}
        worst = f.assign(r=f.verdict.map(rank)).sort_values("r", ascending=False).drop_duplicates("row").set_index("row").verdict
        t = prof.typed[[xc, yc]].copy()
        t["verdict"] = worst.reindex(t.index)
        normal = t[t.verdict.isna()]
        if len(normal) > 5000:
            normal = normal.sample(5000, random_state=0)
        fig = go.Figure(go.Scattergl(x=normal[xc], y=normal[yc], mode="markers", name="Normal", marker=dict(color=C_NORMAL, size=5, opacity=0.6)))
        for v, (col, sym, lab) in VERDICT_STYLE.items():
            s = t[t.verdict == v]
            if len(s):
                fig.add_trace(go.Scattergl(x=s[xc], y=s[yc], mode="markers", name=f"{lab} ({len(s)})", text=s.index.astype(str),
                                           marker=dict(color=col, size=9, symbol=sym, line=dict(width=1, color="white"))))
        fig.update_layout(title=f"{yc} vs {xc}", xaxis_title=xc, yaxis_title=yc)
        if st.checkbox("Log axes", value=xc in anom.log_scaled and yc in anom.log_scaled):
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
        st.plotly_chart(style(fig, 440), width="stretch")
        st.markdown("#### All findings")
        fa, fb = st.columns(2)
        vsel = fa.multiselect("Verdict", ["error", "review", "legit"], default=["error", "review"])
        csel = fb.multiselect("Column", sorted(f.column.unique()), default=[])
        view = f[f.verdict.isin(vsel)]
        if csel:
            view = view[view.column.isin(csel)]
        st.dataframe(view[["row", "column", "value", "issue", "verdict", "confidence", "reason", "fix"]].round(3), hide_index=True, width="stretch", height=360)

# ------------------------------------------------------------------ plan
with tabs[3]:
    st.markdown("Every change is a proposal. Tick what you approve, optionally change the strategy, then apply.")
    approved, strategies = set(), {}
    groups = {"coerce_numeric": "Types", "parse_dates": "Types", "standardise_categories": "Categories", "fuzzy_categories": "Categories",
              "drop_duplicates": "Duplicates", "fix_values": "Errors", "review_outliers": "Outliers to review", "impute": "Missing values"}
    last = None
    for a in plan:
        g = groups[a.kind]
        if g != last:
            st.markdown(f"##### {g}")
            last = g
        c1, c2 = st.columns([5, 2])
        with c1:
            on = st.checkbox(f"{a.title}  ·  {a.affected:,} affected", value=a.default_on, key=f"ap_{h}_{a.id}")
            st.caption(a.explanation)
        with c2:
            if a.options:
                strategies[a.id] = st.selectbox("Strategy", a.options, index=a.options.index(a.strategy) if a.strategy in a.options else 0,
                                                key=f"st_{h}_{a.id}", label_visibility="collapsed")
        if on:
            approved.add(a.id)
    if not plan:
        st.success("Nothing to clean.")
    if st.button("Apply approved actions", type="primary"):
        st.session_state["result"] = apply_plan(df, prof, anom, plan, approved, strategies)
        st.success("Done - see the next tab.")

# ------------------------------------------------------------------ before / after
with tabs[4]:
    res = st.session_state.get("result")
    if res is None:
        st.info("Approve and apply actions in the 'Cleaning plan' tab first.")
    else:
        cl = res.cleaned
        after_missing = int(cl.isna().sum().sum())
        c = st.columns(4)
        c[0].metric("Rows", f"{len(cl):,}", f"{len(cl) - len(df):,}")
        c[1].metric("Missing cells", f"{after_missing:,}", f"{after_missing - prof.n_missing_cells:,}", delta_color="inverse")
        c[2].metric("Duplicate rows", f"{int(cl.duplicated().sum()):,}", f"{int(cl.duplicated().sum()) - prof.dup_exact:,}", delta_color="inverse")
        c[3].metric("Cells changed", f"{len(res.changes):,}" + ("+" if res.changes_truncated else ""))
        for cc, info in prof.categories.items():
            if cc in cl.columns:
                st.caption(f"'{cc}': {df[cc].nunique()} → {cl[cc].nunique()} distinct values")
        ncols = [c_ for c_ in anom.num_cols if c_ in cl.columns]
        if ncols:
            col = st.selectbox("Compare distribution of", ncols)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=prof.typed[col], name="Before", marker_color=C_NORMAL, opacity=0.75, nbinsx=60))
            fig.add_trace(go.Histogram(x=pd.to_numeric(cl[col], errors="coerce"), name="After", marker_color=C_REVIEW, opacity=0.6, nbinsx=60))
            fig.update_layout(barmode="overlay", title=f"{col}: before vs after")
            if col in anom.log_scaled:
                fig.update_xaxes(type="log")
            st.plotly_chart(style(fig, 340), width="stretch")
        st.markdown("#### Cleaning log")
        st.dataframe(res.action_log, hide_index=True, width="stretch")
        with st.expander("Cell-level changes" + (" (first 50,000)" if res.changes_truncated else "")):
            st.dataframe(res.changes.head(2000).astype(str), hide_index=True, width="stretch")
        st.markdown("#### Download")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Cleaned CSV", cl.to_csv(index=False).encode(), "cleaned.csv", "text/csv")
        xb = io.BytesIO()
        with pd.ExcelWriter(xb, engine="openpyxl") as w:
            cl.to_excel(w, index=False, sheet_name="cleaned")
            res.action_log.to_excel(w, index=False, sheet_name="log")
        d2.download_button("Cleaned Excel (+log)", xb.getvalue(), "cleaned.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d3.download_button("Change log CSV", res.changes.astype(str).to_csv(index=False).encode(), "changes.csv", "text/csv")
        d4.download_button("Report (Markdown)", ("# DataDoctor report\n\n" + headline_text(H)).encode(), "report.md", "text/markdown")

# ------------------------------------------------------------------ benchmark
with tabs[5]:
    st.markdown("Controlled-corruption benchmark: clean synthetic data + known injected errors → precision / recall / F1 "
                "and **damage to valid data**. Run `python -m datadoctor.benchmark` for the full study.")
    rp = Path("results/detection_by_rate.csv")
    if st.button("Run a quick benchmark now (≈30 s)"):
        from datadoctor.benchmark import run_benchmark
        with st.spinner("Running…"):
            det, bt, cl_ = run_benchmark(n=3000, seeds=(0, 1), rates=(0.03,), verbose=False)
        st.session_state["quick"] = (det, cl_)
    det = None
    if "quick" in st.session_state:
        det, cl_ = st.session_state["quick"]
        st.caption("Quick run (3,000 rows, 2 seeds, 3% corruption)")
    elif rp.exists():
        det = pd.read_csv(rp)
        cl_ = pd.read_csv("results/cleaning.csv") if Path("results/cleaning.csv").exists() else None
        st.caption("Saved full results (results/)")
    if det is not None:
        agg = det.groupby("method", sort=False)[["precision", "recall", "f1", "legit_rows_flagged_pct", "clean_rows_flagged_pct"]].mean().round(3)
        st.dataframe(agg, width="stretch")
        fig = go.Figure()
        for i, (met, color) in enumerate([("precision", C_REVIEW), ("recall", C_ERROR), ("f1", C_LEGIT)]):
            fig.add_trace(go.Bar(x=agg.index, y=agg[met], name=met.upper() if met == "f1" else met.title(), marker_color=color))
        fig.update_layout(barmode="group", title="Detection quality by method (mean over runs)", yaxis_range=[0, 1])
        st.plotly_chart(style(fig, 380), width="stretch")
        if cl_ is not None:
            keep = ["valid_orders_lost_pct", "valid_cells_damaged_pct", "legit_order_cells_damaged_pct", "error_cells_restored_pct", "error_cells_left_wrong_pct"]
            st.markdown("**Cleaning pipelines (mean %)**")
            st.dataframe(cl_.groupby("pipeline", sort=False)[keep].mean().round(2), width="stretch")
    else:
        st.info("No saved results yet. Run the benchmark from the command line or use the button above.")
