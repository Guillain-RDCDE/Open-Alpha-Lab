"""Headline run for Study 102 (Free-Rebalance) on the real multi-asset total-return tape.

Compares a fixed-weight (50/50 SPY/TLT) portfolio REBALANCED (annual & quarterly) vs the
same initial weights left to DRIFT (buy-and-hold), net of rebalancing costs, on the
honest common window (TLT starts 2002-07). Reports the realised rebalancing bonus
(rebalanced CAGR - drift CAGR), the Booth-Fama diversification-return identity, a
block-bootstrap CI on the daily bonus, and the risk it controls. Stamps as-of + a content
fingerprint per asset.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from free_rebalance import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
TICKERS = ("SPY", "TLT")
COST_BPS = 10.0


def row(name, d):
    print(f"  {name:<22} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
          f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  rebal {d['n_rebal']}")


def main() -> None:
    frame = data.load_real(TICKERS, mode="total_return").loc[:AS_OF]
    print(f"[data] basket {'/'.join(TICKERS)} total-return: {len(frame):,} rows  "
          f"{frame.index[0].date()} -> {frame.index[-1].date()}  as-of {AS_OF}")
    for tk in TICKERS:
        fp = data.fingerprint(frame[[tk]])
        print(f"       {tk:<5} fingerprint={fp}  "
              f"({frame[tk].index[0].date()} -> {frame[tk].index[-1].date()})")
    print(f"       basket fingerprint={data.fingerprint(frame)}")

    drift = strategy.run_portfolio(frame, freq="drift", cost_bps=COST_BPS)
    annual = strategy.run_portfolio(frame, freq="annual", cost_bps=COST_BPS)
    quarterly = strategy.run_portfolio(frame, freq="quarterly", cost_bps=COST_BPS)

    print(f"\n--- 50/50 {'/'.join(TICKERS)} (total return), cost = {COST_BPS:.0f} bps one-way/rebalance ---")
    row("Rebalanced (annual)", annual)
    row("Rebalanced (quarterly)", quarterly)
    row("Drift (buy & hold)", drift)

    bonus_a = annual["cagr"] - drift["cagr"]
    bonus_q = quarterly["cagr"] - drift["cagr"]
    print(f"\n  Rebalancing bonus (annual)    = {bonus_a*100:+.2f} pts/yr CAGR")
    print(f"  Rebalancing bonus (quarterly) = {bonus_q*100:+.2f} pts/yr CAGR")
    print(f"  Sharpe: annual {annual['sharpe']:.3f} | quarterly {quarterly['sharpe']:.3f} | drift {drift['sharpe']:.3f}")
    print(f"  Vol:    annual {annual['vol']*100:.1f}% | quarterly {quarterly['vol']*100:.1f}% | drift {drift['vol']*100:.1f}%")
    print(f"  MaxDD:  annual {annual['max_dd']*100:.1f}% | quarterly {quarterly['max_dd']*100:.1f}% | drift {drift['max_dd']*100:.1f}%")

    # Booth-Fama / Willenbrock diversification-return identity (vs weighted-avg baseline).
    dr = strategy.diversification_return(frame)
    print(f"\n  Booth-Fama diversification return (vs weighted-avg of assets):")
    print(f"    DR = 0.5*(wavg_var - var_p) = {dr['dr_annual']*100:+.2f} pts/yr  "
          f"(var reduction {dr['var_reduction']*100:.2f} pts)")
    print(f"    NOTE: this textbook 'bonus' is always >= 0 and is measured vs the weighted")
    print(f"    average of constituents, NOT the drift portfolio -- the realised bonus above is the honest one.")

    # Block-bootstrap CI on the daily bonus series (annual rebalanced - drift).
    diff = (annual["net"] - drift["net"]).to_numpy()
    t = strategy.hac_tstat(diff)
    lo, hi = strategy.block_bootstrap_ci(diff, block=63, n_boot=2000)
    print(f"\n  Daily bonus (annual reb - drift): mean {diff.mean()*1e4:+.3f} bps/day  HAC t = {t:+.2f}")
    print(f"    block-bootstrap 95% CI on mean daily bonus = [{lo*1e4:+.3f}, {hi*1e4:+.3f}] bps/day")

    # Sub-period sign of the bonus (does it flip by regime?).
    print(f"\n  Bonus by sub-period (annual rebalanced - drift, CAGR pts/yr):")
    yrs = frame.index.year
    halves = [(yrs <= 2013, "2002-2013"), (yrs >= 2014, "2014-2026")]
    for mask, lab in halves:
        sub = frame.loc[mask]
        if len(sub) < 300:
            continue
        ra = strategy.run_portfolio(sub, freq="annual", cost_bps=COST_BPS)
        rd = strategy.run_portfolio(sub, freq="drift", cost_bps=COST_BPS)
        print(f"    {lab}: {(ra['cagr']-rd['cagr'])*100:+.2f} pts/yr")


if __name__ == "__main__":
    main()
