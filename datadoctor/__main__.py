"""CLI:  python -m datadoctor input.csv [--out cleaned.csv] [--report report.md]
Applies the default (recommended) approvals. Use the Streamlit app for interactive approval."""
import argparse
import time

from .plan import apply_plan, build_plan, default_approved, headline, headline_text
from .anomaly import detect
from .profiling import load_table, profile


def main():
    ap = argparse.ArgumentParser(prog="datadoctor")
    ap.add_argument("input")
    ap.add_argument("--out", default="cleaned.csv")
    ap.add_argument("--report", default="report.md")
    a = ap.parse_args()
    t0 = time.time()
    df = load_table(a.input)
    prof = profile(df)
    anom = detect(prof.typed, prof.kinds)
    plan = build_plan(prof, anom)
    res = apply_plan(df, prof, anom, plan, default_approved(plan))
    res.cleaned.to_csv(a.out, index=False)
    actions = "\n".join(f"- {r.title}: {r.cells_or_rows_changed} changed" for r in res.action_log.itertuples())
    text = "# DataDoctor report\n\n" + headline_text(headline(prof, anom)) + "\n\n## Actions applied\n\n" + actions
    open(a.report, "w").write(text)
    print(text)
    print(f"\nWrote {a.out} and {a.report} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
