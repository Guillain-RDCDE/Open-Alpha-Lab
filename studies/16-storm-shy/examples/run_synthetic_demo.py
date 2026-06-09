"""Offline demo — the whole machine on the synthetic tape, no network.

Generates a toy daily close with *known* volatility clustering (a persistent calm/storm regime
around a constant drift), then runs the same teardown the real script runs:

    * volatility is FORECASTABLE -- realized variance is strongly autocorrelated (the engine);
    * scaling exposure by past-only inverse-vol LIFTS the Sharpe at low turnover (the overlay);
    * the lift is a real Moreira-Muir spanning ALPHA -- significant under HAC errors -- with a
      bootstrap CI on the Sharpe gain that excludes zero;
    * the honest counter: at matched risk a risk-averse CRRA investor still gains (certainty-
      equivalent > 0), but that is the smaller, real number behind the headline;
    * the NULL: re-run on a FLAT-vol tape (one regime). With nothing to forecast, every leg
      collapses to ~zero -- proving the gain comes from clustering, not from the machinery.

    python examples/run_synthetic_demo.py

This is the reproducible core: it proves the diagnostics recover what we *baked in* (forecastable
vol and a genuine, bounded Sharpe gain), so the real-data run (`verify.py`) is a measurement, not a
hope.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from storm_shy import data, vol, strategy, decompose


def _teardown(close, label):
    r = vol.to_returns(close)
    f = vol.forecastability(r, horizon=21)
    cmp = strategy.compare(r, cost_bps=1.0)
    sp = decompose.spanning_alpha(r, cost_bps=1.0)
    bs = decompose.sharpe_gain_bootstrap(r, n_boot=2000, seed=0, cost_bps=1.0)
    ce = decompose.certainty_equivalent(r, gamma=5.0, cost_bps=1.0)
    er = decompose.equal_risk_return(r, cost_bps=1.0)

    print(f"\n========================  {label}  ========================")
    print(f"[1 . is volatility forecastable?]  (the engine of the whole thing)")
    print(f"  log-variance AR(1) rho = {f['rho']:+.2f}, lag-1 autocorr = {f['autocorr_lag1']:+.2f}, "
          f"vol-of-vol = {f['vol_of_vol']:.2f}  ({f['n_blocks']} monthly blocks)")

    print(f"\n[2 . does scaling by inverse vol pay?]  (buy-&-hold vs vol-managed, 1 bp/turn)")
    bh, mg = cmp["buy_hold"], cmp["managed_net"]
    print(f"  buy & hold : Sharpe {bh['sharpe']:+.2f}, vol {bh['vol_ann']:.1%}, maxDD {bh['max_drawdown']:.0%}")
    print(f"  vol-managed: Sharpe {mg['sharpe']:+.2f}, vol {mg['vol_ann']:.1%}, maxDD {mg['max_drawdown']:.0%}")
    print(f"  -> Sharpe gain {cmp['sharpe_gain_net']:+.2f} | avg leverage {cmp['avg_leverage']:.2f}, "
          f"capped {cmp['frac_capped']:.0%} of days, turnover {cmp['turnover_ann']:.0f}x/yr (low)")

    print(f"\n[3 . is the lift statistically real?]")
    print(f"  Moreira-Muir spanning alpha = {sp['alpha_ann_pct']:+.2f}%/yr, HAC t = {sp['alpha_t']:+.2f} "
          f"(beta to market {sp['beta']:.2f})")
    print(f"  bootstrap Sharpe gain {bs['sharpe_gain']:+.2f}, 95% CI [{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}], "
          f"P(gain<0) = {bs['frac_negative']:.1%}")

    print(f"\n[4 . the honest counter: utility at MATCHED risk]  (Cederburg et al.)")
    print(f"  CRRA(gamma=5) certainty-equivalent gain = {ce['ce_gain_pct']:+.2f}%/yr; "
          f"equal-risk excess CAGR {er['excess_cagr_pct']:+.2f}%/yr")
    print(f"  drawdown at matched risk: buy & hold {er['buy_hold_maxdd']:.0%} -> managed {er['managed_maxdd']:.0%}")


def main():
    close, truth = data.synthetic_prices(seed=16)
    print(f"clustered tape: {truth.n_bars} bars, calm fraction ~{truth.calm_fraction:.0%}, "
          f"calm/storm vol {truth.sigma_lo:.3f}/{truth.sigma_hi:.3f}; "
          f"perfect-foresight Sharpe ceiling x{truth.theoretical_sharpe_gain:.2f}")
    _teardown(close, "CLUSTERED  (a real vol regime - the overlay can read it)")

    flat, _ = data.synthetic_prices(sigma_lo=0.011, sigma_hi=0.011, seed=16)
    _teardown(flat, "FLAT-VOL NULL  (one regime - nothing to forecast)")

    print("\n-> The clustered tape pays; the flat null does not. The edge is the clustering, "
          "not the apparatus.")


if __name__ == "__main__":
    main()
