"""Real-panel verification — Study 242 (Quality-Minus-Junk). Regenerates docs/results.md numbers.

Reads the desk's shared EDGAR caches, builds the QMJ composite (profitability + growth +
safety pillars), runs the top-quintile (quality) / bottom-quintile (junk) long-short with
a one-year reporting lag, and reports vs the equal-weight universe.
Network is never touched; the EDGAR cache is maintained centrally.

    python studies/242-quality-minus-junk/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quality_minus_junk import data, strategy as st  # noqa: E402


def main() -> None:
    score, fwd_ret = data.fetch_panel()

    if score.empty:
        print("EDGAR cache not found. Run from repo root with the shared _cache/ populated.")
        return

    print("Study 242 — Quality-Minus-Junk — real EDGAR panel\n")
    fp_sig = data.fingerprint(score)
    fp_fwd = data.fingerprint(fwd_ret)
    print(f"QMJ composite score: {score.shape}, fingerprint={fp_sig}")
    print(f"Forward returns panel: {fwd_ret.shape}, fingerprint={fp_fwd}")
    print(f"Years covered: {score.index.min()} – {score.index.max()}")
    print(f"As-of: 2026-06-16 | Source: EDGAR XBRL + Yahoo monthly\n")

    h = st.quintile_hedge(score, fwd_ret, q=0.20)
    s_hedge = st.summary(h["hedge"])
    s_high = st.summary(h["high"])
    s_low = st.summary(h["low"])
    s_mkt = st.summary(h["market"])

    print("=== Quality (top quintile QMJ) vs Junk (bottom quintile) — long-short hedge ===")
    print(f"  N years:         {s_hedge['n']}")
    print(f"  Mean hedge:      {s_hedge['mean']*100:+.2f}%/yr")
    print(f"  Vol:             {s_hedge['vol']*100:.2f}%/yr")
    print(f"  Sharpe (annual): {s_hedge['sharpe']:+.3f}")
    print(f"  HAC t-stat:      {s_hedge['tstat']:+.3f}")
    print(f"  Hit rate:        {s_hedge['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:    {s_hedge['max_drawdown']*100:.1f}%")

    print("\n=== Legs vs equal-weight market ===")
    print(f"  Quality mean:    {s_high['mean']*100:+.2f}%/yr  (market: {s_mkt['mean']*100:+.2f}%/yr)")
    print(f"  Junk mean:       {s_low['mean']*100:+.2f}%/yr")
    print(f"  Quality vs mkt:  {(s_high['mean'] - s_mkt['mean'])*100:+.2f}%/yr")
    print(f"  Junk vs mkt:     {(s_low['mean'] - s_mkt['mean'])*100:+.2f}%/yr")

    print("\n=== Year-by-year hedge returns ===")
    for yr in h.index:
        row = h.loc[yr]
        print(f"  {yr}: hedge={row['hedge']*100:+6.1f}%  quality={row['high']*100:+6.1f}%  "
              f"junk={row['low']*100:+6.1f}%  mkt={row['market']*100:+6.1f}%  n={int(row['n'])}")

    print("\nNOTE Survivorship bias: universe = current S&P 500 projected backwards.")
    print("     Positive results are upper-bound estimates, not generalisable population effects.")
    print("\nNOTE QMJ pillars used: Profitability (GP/A+ROA+NetMargin+CFO/A) + Growth + Safety (Eq/Assets).")
    print("     Payout pillar excluded (no dividend/share-count in shared cache).")


if __name__ == "__main__":
    main()
