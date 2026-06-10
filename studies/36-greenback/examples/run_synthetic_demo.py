"""Offline demo — the carry / dollar-carry / momentum / combo machinery on the synthetic FX control.

No network, deterministic. On the carry tape, high-rate currencies out-earn (the carry premium) with a
sharp negative skew (the steamroller); a trailing-trend momentum book is profitable too; and — the point
of the study — the **carry⊕momentum combo** Sharpe beats either standalone because the two legs barely
correlate, with momentum dulling the carry crash. On the full-UIRP carry null, the carry premium vanishes.

    python examples/run_synthetic_demo.py

The real FX-carry verdict needs short rates from FRED, which time out in this sandbox — so the real run is
PENDING one networked fetch (see examples/verify.py and ../docs/results.md).
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

from greenback import costs, data, extension, strategy


def _line(nm, r):
    s = strategy.summary(r)
    print(f"  {nm:22} Sharpe {s['sharpe']:+5.2f}  ret {s['ann_return']*100:+5.1f}%  vol {s['vol_ann']*100:4.1f}%  "
          f"skew {s['skew']:+5.2f}  maxDD {s['max_drawdown']*100:4.0f}%")


def main() -> None:
    xr, rd, truth = data.synthetic_fx(carry_strength=0.9, trend_strength=0.35, seed=36)
    x0, r0, _ = data.synthetic_fx(carry_strength=0.0, trend_strength=0.35, seed=36)
    print(f"Synthetic control: {truth.n_ccy} currencies x {truth.n_months} months, "
          f"carry_strength {truth.carry_strength} (null=0), trend_strength {truth.trend_strength}\n")

    print("The three sleeves and the combo (net @10 bp):")
    carry = strategy.carry_returns(xr, rd, cost_bps=10.0)
    mom = strategy.momentum_returns(xr, cost_bps=10.0)
    _line("carry (vol-scaled)", carry)
    _line("dollar-carry tilt", strategy.dollar_carry_returns(xr, rd, cost_bps=10.0))
    _line("momentum", mom)
    _line("CARRY+MOMENTUM combo", strategy.combine(carry, mom))

    print(f"\nCarry premium (high-minus-low rate bucket): {strategy.carry_premium_by_bucket(xr, rd)['hml_ann_pct']:+.1f}%/yr")
    print(f"Full-UIRP null bucket spread:               {strategy.carry_premium_by_bucket(x0, r0)['hml_ann_pct']:+.1f}%/yr (≈0)")

    print("\nBeat-7 diversification (carry⊕momentum):")
    d = extension.diversification(xr, rd, cost_bps=10.0)
    print(f"  carry Sharpe {d['carry_sharpe']:+.2f}  momentum {d['momentum_sharpe']:+.2f}  "
          f"combo {d['combo_sharpe']:+.2f}  (combo beats legs: {d['combo_beats_legs']})")
    print(f"  carry↔momentum correlation {d['leg_correlation']:+.2f}  →  they pay at different times")

    cc = extension.carry_crash(xr, rd, cost_bps=10.0)
    print(f"\nThe carry crash (the steamroller): carry skew {cc['carry_skew']:+.2f}, worst month "
          f"{cc['carry_worst_month_pct']:+.1f}%, max DD {cc['carry_max_dd_pct']:.0f}%.")
    print(f"  Momentum cushions it: combo max DD {cc['combo_max_dd_pct']:.0f}% "
          f"(vs carry {cc['carry_max_dd_pct']:.0f}%); in carry's worst 5 months the combo lost "
          f"{cc['combo_in_carry_worst5_pct']:+.1f}% vs carry {cc['carry_worst5_mean_pct']:+.1f}%.")

    print("\nCost sweep on the combo:")
    print(costs.cost_sweep(xr, rd, which="combo").round(3).to_string())

    print("\nThe real FX-carry verdict is PENDING one networked FRED fetch — see examples/verify.py --fetch.")


if __name__ == "__main__":
    main()
