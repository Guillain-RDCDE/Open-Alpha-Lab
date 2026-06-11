"""Offline demo — prove the machinery on a synthetic G10 with a known carry premium and crashes.

No network, deterministic. On the premium tape, high-rate currencies out-earn and the carry book earns a
real, significant premium — with a sharply negative skew (the steamroller). On the full-UIRP null,
nothing. And vol-management lifts the Sharpe but does not dodge the (jump-like) crash.

One stated limit of the control: each synthetic currency's rate is **constant**, so the ranking never
changes and the book's turnover is exactly 0.0×/yr — the "cheap to run" claim is *exercised* only on the
real tape (turnover ≈ 0.5×/yr there; see docs/results.md).

    python examples/run_synthetic_demo.py

The real G10 verdict is in docs/results.md (offline from the shared desk cache; see examples/verify.py).
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from steamroller import data, carry, strategy, decompose, extension


def run(label, cs):
    xr, rates, truth = data.synthetic_carry(carry_strength=cs, seed=27)
    pb = carry.carry_premium_by_bucket(xr, rates)
    cmp = strategy.compare(xr, rates, cost_bps=10.0)
    pt = decompose.premium_tstat(xr, rates, cost_bps=10.0)
    cr = decompose.crash_profile(xr, rates, cost_bps=10.0)
    print(f"\n=== {label} (carry_strength={cs}, premium={truth.has_premium}) ===")
    print(f"  high-minus-low rate bucket spread {pb['hml_ann_pct']:+.1f}%/yr")
    print(f"  carry Sharpe {cmp['sharpe']:+.2f}, premium {pt['mean_ann_pct']:+.1f}%/yr (HAC t {pt['t_stat']:+.1f})")
    print(f"  steamroller: skew {cr['skew']:+.2f}, worst month {cr['worst_month_pct']:+.1f}%, max drawdown {cr['max_drawdown_pct']:.0f}%")


def main():
    print("Steamroller — a synthetic G10 carry premium with risk-off crashes, vs a full-UIRP null.")
    run("CARRY PREMIUM", cs=0.9)
    run("UIRP null", cs=0.0)
    xr, rates, _ = data.synthetic_carry(carry_strength=0.9, seed=27)
    cc = extension.crash_comparison(xr, rates, cost_bps=10.0)
    print(f"\nrisk-managed (shared {cc['n_months']}-month window, post burn-in): "
          f"plain Sharpe {cc['plain']['sharpe']:+.2f} (drawdown {cc['plain']['max_drawdown_pct']:.0f}%) "
          f"-> vol-managed {cc['managed']['sharpe']:+.2f} (drawdown {cc['managed']['max_drawdown_pct']:.0f}%) "
          f"-- the Sharpe lifts, the tail does not. (Plain Sharpe differs from the full-sample headline "
          f"because the managed book only exists after its 12-month vol burn-in.)")
    print("The real G10 verdict (premium, t-stat, 2008/2020 crashes, turnover) is in ../docs/results.md.")


if __name__ == "__main__":
    main()
