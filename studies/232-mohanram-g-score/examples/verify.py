"""Real-panel verification -- Study 232 (Mohanram G-score). Regenerates docs/results.md numbers.

Reads the desk's shared EDGAR caches (NetIncomeLoss, Assets, CFO, Revenues, yrret),
computes the 8-component G-score, runs the top-quintile / bottom-quintile long-short
with a one-year reporting lag, and reports vs the equal-weight universe. Network is
never touched; the EDGAR cache is maintained centrally.

    python studies/232-mohanram-g-score/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mohanram_g_score import data, strategy as st  # noqa: E402


def main() -> None:
    signal, fwd_ret = data.fetch_panel()

    if signal.empty:
        print("EDGAR cache not found. Run from repo root with the shared _cache/ populated.")
        return

    print("Study 232 -- Mohanram G-score -- real EDGAR panel\n")
    fp_sig = data.fingerprint(signal)
    fp_fwd = data.fingerprint(fwd_ret)
    print(f"G-score panel:          {signal.shape}, fingerprint={fp_sig}")
    print(f"Forward returns panel:  {fwd_ret.shape}, fingerprint={fp_fwd}")
    print(f"Years covered: {signal.index.min()} - {signal.index.max()}")
    print(f"As-of: 2026-06-16 | Source: EDGAR XBRL + Yahoo monthly\n")

    h = st.quintile_hedge(signal, fwd_ret, q=0.20)
    s_hedge = st.summary(h["hedge"])
    s_high = st.summary(h["high"])
    s_low = st.summary(h["low"])
    s_mkt = st.summary(h["market"])

    print("=== Top-quintile G-score vs Bottom-quintile G-score (long-short hedge) ===")
    print(f"  N years:         {s_hedge['n']}")
    print(f"  Mean hedge:      {s_hedge['mean']*100:+.2f}%/yr")
    print(f"  Vol:             {s_hedge['vol']*100:.2f}%/yr")
    print(f"  Sharpe (annual): {s_hedge['sharpe']:+.3f}")
    print(f"  HAC t-stat:      {s_hedge['tstat']:+.3f}")
    print(f"  Hit rate:        {s_hedge['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:    {s_hedge['max_drawdown']*100:.1f}%")

    print("\n=== Legs vs equal-weight market ===")
    print(f"  High G mean:     {s_high['mean']*100:+.2f}%/yr  (market: {s_mkt['mean']*100:+.2f}%/yr)")
    print(f"  Low G mean:      {s_low['mean']*100:+.2f}%/yr")
    print(f"  High vs market:  {(s_high['mean'] - s_mkt['mean'])*100:+.2f}%/yr")
    print(f"  Low vs market:   {(s_low['mean'] - s_mkt['mean'])*100:+.2f}%/yr")

    print("\n=== Year-by-year hedge returns ===")
    for yr in h.index:
        row = h.loc[yr]
        print(f"  {yr}: hedge={row['hedge']*100:+6.1f}%  high={row['high']*100:+6.1f}%  "
              f"low={row['low']*100:+6.1f}%  mkt={row['market']*100:+6.1f}%  n={int(row['n'])}")

    print()
    print(f"quantlab.repro.fingerprint: sig={fp_sig} fwd={fp_fwd}")
    print("\nNOTE Survivorship bias: universe = current S&P 500 projected backwards.")
    print("     Positive results are upper-bound estimates; negative results are real.")
    print("NOTE G6/G7 substitution: revenue-growth and asset-turnover-growth replace")
    print("     R&D-intensity and advertising-intensity (not in desk EDGAR cache).")


if __name__ == "__main__":
    main()
