"""Real-data run — the dollar-carry / carry⊕momentum combo on the G10 tape, OFFLINE from cache.

    python examples/verify.py            # reads _cache/g10_short_rates.parquet + _cache/g10_fx.parquet

Builds the carry, dollar-carry, momentum and combo books on the real tape (OECD 3-month short rates +
yfinance FX, a USD-funded monthly book), prints the carry premium and its negative-skew crash, the
momentum sleeve, and the carry⊕momentum combo's Sharpe vs each leg + the leg correlation — fingerprinted
and **as-of pinned to the rates' 2024-01 end** (OECD MEI was discontinued there). Pass nothing: it is
fully offline. Re-run and match the fingerprint to hold the exact data behind the published numbers.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from greenback import costs, data, extension, strategy  # noqa: E402

COST_BPS = 10.0


def main() -> None:
    panel = data.load_real_panel()
    if panel is None:
        print("[skip] missing _cache/g10_short_rates.parquet or _cache/g10_fx.parquet.")
        print("       The offline synthetic core (run_synthetic_demo.py) is the validated machinery proof.")
        return
    xr, rate_diffs = panel
    as_of = data.DATA_AS_OF
    try:
        from quantlab.repro import fingerprint
        from quantlab.stats import sharpe_ci_bootstrap
        fp = fingerprint(xr.round(6))
        carry_ci = sharpe_ci_bootstrap(
            strategy.carry_returns(xr, rate_diffs, cost_bps=COST_BPS), periods_per_year=12, seed=36)
    except Exception:
        fp, carry_ci = "n/a", None

    span = f"{xr.index.min().date()} -> {xr.index.max().date()} ({len(xr)} months)"
    print(f"G10 real tape (USD-funded, monthly): {span}; currencies {list(xr.columns)}")
    print(f"as-of {as_of} · fingerprint {fp}\n")

    carry = strategy.carry_returns(xr, rate_diffs, cost_bps=COST_BPS)
    mom = strategy.momentum_returns(xr, cost_bps=COST_BPS)
    combo = strategy.combine(carry, mom)
    dollar = strategy.dollar_carry_returns(xr, rate_diffs, cost_bps=COST_BPS)
    print("The three sleeves and the combo (net @10 bp, vol-scaled to 8%):")
    for nm, r in [("carry", carry), ("dollar-carry", dollar), ("momentum", mom), ("COMBO", combo)]:
        s = strategy.summary(r)
        print(f"  {nm:14} Sharpe {s['sharpe']:+5.2f}  ret {s['ann_return']*100:+5.1f}%  "
              f"vol {s['vol_ann']*100:4.1f}%  skew {s['skew']:+5.2f}  maxDD {s['max_drawdown']*100:4.0f}%")

    pb = strategy.carry_premium_by_bucket(xr, rate_diffs)
    print(f"\nCarry premium (high-minus-low rate bucket): {pb['hml_ann_pct']:+.1f}%/yr")
    if carry_ci is not None:
        print(f"Carry Sharpe bootstrap 95% CI [{carry_ci['ci_low']:+.2f}, {carry_ci['ci_high']:+.2f}]; "
              f"{carry_ci['frac_negative']*100:.0f}% of resamples negative")

    d = extension.diversification(xr, rate_diffs, cost_bps=COST_BPS)
    print(f"\nDiversification (carry⊕momentum): carry {d['carry_sharpe']:+.2f}, momentum "
          f"{d['momentum_sharpe']:+.2f}, combo {d['combo_sharpe']:+.2f}; leg corr {d['leg_correlation']:+.2f}; "
          f"combo beats best leg: {d['combo_beats_legs']}")

    cc = extension.carry_crash(xr, rate_diffs, cost_bps=COST_BPS)
    print(f"\nThe carry crash (the steamroller): carry skew {cc['carry_skew']:+.2f}, worst month "
          f"{cc['carry_worst_month_pct']:+.1f}%, max DD {cc['carry_max_dd_pct']:.0f}%.")
    print(f"  Combo cushions it: skew {cc['combo_skew']:+.2f}, worst month {cc['combo_worst_month_pct']:+.1f}%, "
          f"max DD {cc['combo_max_dd_pct']:.0f}%; in carry's worst 5 months the combo lost "
          f"{cc['combo_in_carry_worst5_pct']:+.1f}% vs carry {cc['carry_worst5_mean_pct']:+.1f}%.")

    worst = carry.nsmallest(3)
    print("  Worst carry months: " + ", ".join(f"{d_.date()} {v*100:+.1f}%" for d_, v in worst.items()))

    to_c = strategy.turnover_ann(strategy.carry_weights(rate_diffs))
    to_m = strategy.turnover_ann(strategy.momentum_weights(xr))
    be_c = costs.breakeven_cost_bps(xr, rate_diffs, which="carry")
    print(f"\nTurnover: carry {to_c:.2f}x/yr, momentum {to_m:.2f}x/yr; carry break-even ≈ {be_c:.0f} bp.")
    print("Cost sweep on the combo (net Sharpe):")
    print(costs.cost_sweep(xr, rate_diffs, which="combo").round(3).to_string())


if __name__ == "__main__":
    main()
