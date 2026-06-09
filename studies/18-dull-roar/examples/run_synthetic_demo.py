"""Offline demo — prove the machinery on a synthetic universe with a *known* low-vol anomaly.

No network, deterministic. Runs the whole teardown twice: on an **anomaly** panel (a baked flat
security-market line, so the calm decile carries real alpha) and on the **null** (textbook CAPM, alpha
zero everywhere). The point is the contrast — every diagnostic that fires on the anomaly tape must go
quiet on the null, which is what proves the code measures the *effect* and not itself.

    python examples/run_synthetic_demo.py

The real-market verdict (current S&P 500, where the celebrated anomaly turns out to have decayed) is a
separate, fingerprinted run in `examples/verify.py` -> `docs/results.md`.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from dull_roar import data, sort, strategy, decompose, extension


def run(label, sml_slope):
    panel, market, truth = data.synthetic_panel(sml_slope=sml_slope, seed=18)
    sml = sort.security_market_line(panel)
    cmp = strategy.compare(panel, market=market, cost_bps=1.0)
    bab = decompose.beta_neutral_bab(panel, market=market, cost_bps=1.0)
    tilt = decompose.beta_tilt_test(panel, market=market, cost_bps=1.0)
    legs = decompose.leg_attribution(panel, market=market, cost_bps=1.0)
    sd = extension.shorting_decomposition(panel, market=market)

    print(f"\n=== {label} (baked low-minus-high alpha spread {truth.annual_sml_alpha_spread:+.1%}/yr) ===")
    print(f"  SML slope (Sharpe~vol)     {sml['sharpe_vol_slope']:+.2f}   "
          f"low/high bucket Sharpe {sml['low_bucket_sharpe']:+.2f}/{sml['high_bucket_sharpe']:+.2f}")
    print(f"  low-vol long-only Sharpe   {cmp['low_only']['sharpe']:+.2f}  vs market "
          f"{cmp['market']['sharpe']:+.2f}  (gain {cmp['low_minus_market_sharpe']:+.2f})")
    print(f"  beta-neutral BAB alpha     {bab['alpha_ann_pct']:+.1f}%/yr  HAC t={bab['alpha_t']:+.1f}  "
          f"beta {bab['beta']:+.2f}  Sharpe {bab['sharpe']:+.2f}")
    print(f"  is it just low beta?       long-only beta {tilt['low_beta']:.2f}, alpha "
          f"{tilt['alpha_ann_pct']:+.1f}% (t={tilt['alpha_t']:+.1f}), excess CAGR@beta1 "
          f"{tilt['excess_cagr_at_beta1_pct']:+.1f}%")
    print(f"  where it lives             low-leg alpha {legs['low_leg_alpha_pct']:+.1f}% / high-leg "
          f"{legs['high_leg_alpha_pct']:+.1f}%  (share in short {legs['share_in_short_leg']:.0%})")
    print(f"  no-shorting share kept     {sd['share_defensive']:+.0%} of the beta-neutral alpha")


def main():
    print("Dull-Roar — synthetic proof of the machinery (anomaly vs the fair-CAPM null).")
    run("ANOMALY  (flat SML baked in)", sml_slope=0.00018)
    run("NULL     (textbook CAPM)", sml_slope=0.0)
    print("\nOn the anomaly tape every leg fires; on the null every leg goes quiet. "
          "The real S&P 500 verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
