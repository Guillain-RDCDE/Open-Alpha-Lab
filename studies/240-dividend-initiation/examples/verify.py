"""Reproduce the real run (docs/results.md) — Study 240 (Dividend-Initiation).

Loads the cached yfinance panel (50-name universe, annual total return + first-dividend
identification), runs the event-study analysis (initiators vs control non-initiators),
and prints the headline numbers that feed docs/results.md.

    python examples/verify.py           # cache-only (no network)
    python examples/verify.py --fetch   # re-fetch from yfinance and repopulate cache
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_initiation import data, strategy as st  # noqa: E402


def main(fetch: bool = False) -> None:
    print("Study 240 - Dividend-Initiation - real panel\n")

    panel = data.load_real_panel(fetch=fetch)
    if panel["initiation_mask"].empty:
        print("No cached data found.  Run with --fetch to populate the cache.")
        return

    initiation_mask = panel["initiation_mask"]
    control_mask = panel["control_mask"]
    fwd_ret = panel["fwd_ret"]
    ew_ret = panel["ew_ret"]
    n_tickers = len(panel["tickers"])
    init_years = panel["initiation_years"]

    # --- initiator arm ---
    initiator_ret = st.arm_returns(fwd_ret, initiation_mask, min_names=1)

    # --- control arm ---
    control_ret = st.arm_returns(fwd_ret, control_mask, min_names=3)

    # --- EW baseline ---
    ew_aligned = ew_ret.reindex(initiator_ret.index)

    # --- spread ---
    spread_vs_control = st.hedge_returns(initiator_ret, control_ret)
    spread_vs_ew = st.hedge_returns(initiator_ret, ew_aligned)

    # --- summaries ---
    si = st.summary(initiator_ret)
    sc = st.summary(control_ret)
    se_ew = st.summary(ew_aligned)
    sh_ctrl = st.summary(spread_vs_control)
    sh_ew = st.summary(spread_vs_ew)

    fp = data.fingerprint(panel)

    # event counts
    counts = st.event_counts(initiation_mask)

    print(f"Universe: {n_tickers} tickers (survivorship-biased - named on Signal axis)")
    print(f"Initiation events identified: {len(init_years)}")
    years = initiation_mask.index
    print(f"Years: {years.min()} - {years.max()}  |  Panel fingerprint: {fp}\n")

    print(f"{'Arm':<28} {'n_yrs':>6} {'mean':>8} {'vol':>8} {'Sharpe':>8} {'t':>7} {'hit%':>7}")
    print("-" * 78)
    for lbl, s in [("Initiators (first div yr)", si), ("Control (non-payers)", sc),
                   ("EW basket", se_ew)]:
        n = s["n"]
        m = f"{s['mean']:+.2%}" if s["mean"] == s["mean"] else "  n/a"
        v = f"{s['vol']:.2%}" if s["vol"] == s["vol"] else "  n/a"
        sr = f"{s['sharpe']:+.2f}" if s["sharpe"] == s["sharpe"] else "  n/a"
        t = f"{s['tstat']:+.2f}" if s["tstat"] == s["tstat"] else "  n/a"
        h = f"{s['hit_rate']:.0%}" if s["hit_rate"] == s["hit_rate"] else "  n/a"
        print(f"{lbl:<28} {n:>6} {m:>8} {v:>8} {sr:>8} {t:>7} {h:>7}")

    print()
    if sh_ctrl["n"] >= 2:
        print(f"Initiator spread vs control:  mean {sh_ctrl['mean']:+.2%}/yr  t = {sh_ctrl['tstat']:+.2f}  (n={sh_ctrl['n']} years)")
    else:
        print("Initiator spread vs control: insufficient overlap for spread test")
    print(f"Initiator spread vs EW:       mean {sh_ew['mean']:+.2%}/yr  t = {sh_ew['tstat']:+.2f}  (n={sh_ew['n']} years)")
    print()
    print(f"Events per year: min={counts.min()}, median={counts.median():.0f}, max={counts.max()}")

    print("\nVerdict:")
    t_ctrl = sh_ctrl["tstat"] if sh_ctrl["n"] >= 2 else float("nan")
    t_ew = sh_ew["tstat"]
    if (t_ctrl == t_ctrl and abs(t_ctrl) >= 2.0 and sh_ctrl["mean"] > 0):
        print("  Signal: REAL - initiators significantly outperform non-payers")
    elif (t_ew == t_ew and abs(t_ew) >= 1.5) or (t_ctrl == t_ctrl and abs(t_ctrl) >= 1.5):
        print("  Signal: WEAK - some tendency but below the inference bar")
    else:
        print("  Signal: NONE - initiator spread is indistinguishable from noise")
    print("  Tradability: MIRAGE - few events per year, no look-ahead universe, large-cap only")


if __name__ == "__main__":
    main("--fetch" in sys.argv)
