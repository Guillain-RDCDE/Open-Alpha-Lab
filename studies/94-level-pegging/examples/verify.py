"""Headline run for Study 94 (Level-Pegging) on the real total-return tapes.

Runs the equal-weight (RSP) vs cap-weight (SPY) S&P 500 race, total return, prints the
numbers the README front-card and the notebooks quote, and stamps the as-of date + content
fingerprint. Re-run to reproduce; match the fingerprint to confirm you hold the same tapes.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from level_pegging import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
EW_FEE_BPS_YR = 20.0  # RSP gross expense ratio ~0.20%/yr (+ higher rebalancing turnover)
CW_FEE_BPS_YR = 9.0  # SPY gross expense ratio ~0.0945%/yr
SPLIT = "2015-01-01"  # start of the mega-cap leadership era (justified, not snooped)


def _row(name, d):
    print(f"  {name:<20} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
          f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  final {d['final']:6.2f}")


def main() -> None:
    frame = data.load_real("RSP", "SPY", mode="total_return").loc[:AS_OF]
    fp = data.fingerprint(frame)
    print(f"[data] RSP (EW) + SPY (CW) total-return: {len(frame):,} rows  "
          f"{frame.index[0].date()} -> {frame.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    gross = strategy.race(frame, 0.0, 0.0)
    net = strategy.race(frame, EW_FEE_BPS_YR, CW_FEE_BPS_YR)

    print("\n--- Full sample, GROSS (total return, no fees) ---")
    _row("RSP equal-weight", gross["ew"])
    _row("SPY cap-weight", gross["cw"])
    print(f"  EW - CW CAGR gap = {gross['ew_minus_cw_cagr']*100:+.2f} pts/yr")

    print(f"\n--- Full sample, NET (RSP -{EW_FEE_BPS_YR:.0f} bps/yr, SPY -{CW_FEE_BPS_YR:.0f} bps/yr) ---")
    _row("RSP equal-weight", net["ew"])
    _row("SPY cap-weight", net["cw"])
    print(f"  EW - CW CAGR gap = {net['ew_minus_cw_cagr']*100:+.2f} pts/yr")

    diff = net["diff"].to_numpy()
    t_diff = strategy.hac_tstat(diff)
    lo, hi = strategy.block_bootstrap_mean_ci(diff)
    print(f"\n  HAC t on EW-CW daily return (net) = {t_diff:+.2f}")
    print(f"  block-boot 95% CI mean daily diff = [{lo*1e4:+.3f}, {hi*1e4:+.3f}] bps/day")

    ab = strategy.alpha_beta(frame)
    print(f"\n  EW on CW regression: alpha = {ab['alpha_ann']*100:+.2f}%/yr "
          f"(HAC t {ab['t_alpha']:+.2f}), beta = {ab['beta']:.3f}, R2 = {ab['r2']:.3f}")

    print(f"\n--- Regime contrast, split at {SPLIT} (net) ---")
    sc = strategy.split_contrast(net["diff"], SPLIT)
    print(f"  early ({frame.index[0].date()}..{SPLIT}) EW-CW = {sc['early_ann']*100:+.2f}%/yr "
          f"(HAC t {sc['early_t']:+.2f})")
    print(f"  late  ({SPLIT}..{frame.index[-1].date()}) EW-CW = {sc['late_ann']*100:+.2f}%/yr "
          f"(HAC t {sc['late_t']:+.2f})")
    print(f"  difference-of-differences = {sc['obs_gap_ann']*100:+.2f} pts/yr  "
          f"block-boot p = {sc['p_boot']:.4f}  ({sc['n_boot']} resamples, block {sc['block']})")

    rel = (1 + net["ew"]["net"]).cumprod() / (1 + net["cw"]["net"]).cumprod()
    print(f"\n  RSP/SPY relative wealth: peak {rel.max():.3f} on {rel.idxmax().date()}  "
          f"-> end {rel.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
