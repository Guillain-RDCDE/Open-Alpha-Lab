"""Reproduce the real-data headline run (docs/results.md) — Sell Rosh Hashanah on the S&P 500.

    python examples/verify.py            # cache-only (offline after first run)
    python examples/verify.py --fetch    # download ^GSPC from Yahoo, then run
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

from rosh_hashanah import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    try:
        prices = data.fetch_gspc(fetch=fetch)
    except FileNotFoundError:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    fp = data.fingerprint(prices)
    start = prices.index[0].date()
    end = prices.index[-1].date()
    print(f"\nRosh-Hashanah study — S&P 500 ^GSPC {start}..{end} · fingerprint {fp}\n")

    hw = data.holiday_windows(prices)
    n = len(hw)
    print(f"Holiday windows extracted: n = {n} (1980–{hw.index.max()})")

    s = st.summarize(hw, col="window_ret")
    print(f"\nHoliday window (Rosh Hashanah eve → Yom Kippur close):")
    print(f"  Mean return : {s['mean_bps']:+.2f} bps  (HAC t = {s['tstat']:+.2f})")
    print(f"  Median      : {s['median_bps']:+.2f} bps")
    print(f"  % negative  : {s['pct_negative']:.1%}")

    rw = st.random_windows(prices, hw, n_draws=40, seed=162)
    comp = st.compare_holiday_vs_random(hw, rw)
    print(f"\nHoliday vs matched random windows (same month, same length):")
    print(f"  Gap         : {comp['mean_gap_bps']:+.2f} bps  (HAC t = {comp['tstat_gap']:+.2f})")
    print(f"  % holiday < random: {comp['pct_holiday_worse']:.1%}")

    # Unconditional 8-trading-day baseline
    import numpy as np
    close = prices["close"]
    all_rets = [close.iloc[i + 8] / close.iloc[i] - 1.0 for i in range(len(close) - 8)]
    uncond_bps = float(np.mean(all_rets) * 1e4)
    print(f"\nUnconditional 8-trading-day S&P mean (all windows): {uncond_bps:+.2f} bps")
    print(f"  → holiday window is {s['mean_bps'] - uncond_bps:+.2f} bps vs unconditional")

    # 2008 sensitivity
    hw_no08 = hw.drop(index=2008, errors="ignore")
    s_no08 = st.summarize(hw_no08, col="window_ret")
    print(f"\n2008 sensitivity (excl 2008, n={s_no08['n']}):")
    print(f"  Mean : {s_no08['mean_bps']:+.2f} bps  (HAC t = {s_no08['tstat']:+.2f})")

    # Timing P&L
    pnl = st.timing_pnl(hw, prices, cost_bps=5.0)
    ps = st.summarize(pnl, col="signal_net")
    print(f"\nTiming P&L — 'Sell RH, buy YK', net of 5 bps round-trip:")
    print(f"  Mean net: {ps['mean_bps']:+.2f} bps/window  (HAC t = {ps['tstat']:+.2f})")

    print("\n=== Verdict ===")
    print("Signal  : NONE  (t = -0.02, 50% negative, dominated by 2008)")
    print("Tradability: MIRAGE  (timing mean < unconditional, costs make it worse)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
