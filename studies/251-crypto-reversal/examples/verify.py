"""Reproducible headline run for Study 251 (Crypto-Reversal).

    python examples/verify.py            # uses the cached panel
    python examples/verify.py --fetch    # refresh the panel from yfinance first

Prints the data stamp, the gross/net long-short arms, the Fama-MacBeth REV slope,
the cost and lookback sweeps, the sub-era breakdown, and the synthetic controls —
the numbers mirrored in docs/results.md and notebooks/build_notebooks.py (the R dict).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_reversal import data, strategy as st  # noqa: E402


def main(fetch: bool = False) -> None:
    panel = data.fetch_panel(fetch=fetch)
    print(f"panel: {panel.shape[1]} tokens x {panel.shape[0]} days "
          f"[{panel.index.min().date()} -> {panel.index.max().date()}]")
    print(f"fingerprint: {data.fingerprint(panel)}")
    print(f"tokens: {', '.join(panel.columns)}\n")

    sig = st.trailing_signal(panel, lookback=21, skip=1)
    rev = st.long_short_returns(panel, sig, leg="reversal", cost_bps=0.0)
    mom = st.long_short_returns(panel, sig, leg="momentum", cost_bps=0.0)
    shuf = st.long_short_returns(panel, st.shuffle_signal(sig, seed=42), leg="reversal", cost_bps=0.0)
    fm = st.fama_macbeth(panel, sig)
    turn = st.avg_turnover(sig, leg="reversal")

    print("=== Gross long-short arms (21-day signal, daily reform, 0 bps) ===")
    for name, r in [("reversal", rev), ("momentum", mom), ("shuffle(null)", shuf)]:
        s = st.summary(r)
        print(f"  {name:14s} ann={s['ann_return']:+.1%}  sharpe={s['sharpe']:+.3f}  "
              f"vol={s['vol_ann']:.0%}  maxDD={s['max_drawdown']:+.1%}  t={s['tstat']:+.2f}")
    print(f"  Fama-MacBeth: slope={fm['avg_slope_bps']:+.2f} bps/day  t_NW={fm['tstat_nw']:+.2f}  "
          f"reversal premium={fm['reversal_premium_ann']:+.0%}/yr")
    print(f"  one-way turnover: {turn:.1%}/day\n")

    print("=== Reversal at realistic crypto costs (one-way bps) ===")
    for c in [0, 10, 30, 50, 100, 200]:
        s = st.summary(st.long_short_returns(panel, sig, leg="reversal", cost_bps=c))
        print(f"  {c:4d} bps  ann={s['ann_return']:+.1%}  sharpe={s['sharpe']:+.3f}  t={s['tstat']:+.2f}")
    print()

    print("=== Reversal across lookback horizons (gross) ===")
    for lb in [7, 14, 21, 42, 63]:
        s = st.summary(st.long_short_returns(panel, st.trailing_signal(panel, lookback=lb), leg="reversal"))
        print(f"  {lb:3d}d  ann={s['ann_return']:+.1%}  sharpe={s['sharpe']:+.3f}  t={s['tstat']:+.2f}")
    print()

    print("=== Sub-era reversal (gross / net 50bps) ===")
    eras = [("2019-2020", "2019-01-01", "2020-12-31"),
            ("2021-2022", "2021-01-01", "2022-12-31"),
            ("2023-2026", "2023-01-01", "2026-06-15")]
    for name, a, b in eras:
        sub = panel.loc[a:b]
        s_sig = st.trailing_signal(sub, lookback=21, skip=1)
        g = st.summary(st.long_short_returns(sub, s_sig, leg="reversal", cost_bps=0.0))
        nret = st.summary(st.long_short_returns(sub, s_sig, leg="reversal", cost_bps=50.0))
        print(f"  {name}: gross sharpe={g['sharpe']:+.3f} (t={g['tstat']:+.2f})  "
              f"net50 sharpe={nret['sharpe']:+.3f} ann={nret['ann_return']:+.1%}")
    print()

    print("=== Synthetic controls ===")
    for strength, lbl in [(1.0, "planted reversal"), (0.0, "null (no signal)")]:
        sp, _ = data.synthetic_panel(signal_strength=strength, seed=213)
        res = st.compare_arms(sp, cost_bps=0.0)
        print(f"  {lbl:18s} reversal sharpe={res['reversal']['sharpe']:+.3f} "
              f"(t={res['reversal']['tstat']:+.2f}) | momentum sharpe={res['momentum']['sharpe']:+.3f} "
              f"| FM t={res['fama_macbeth']['tstat_nw']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
