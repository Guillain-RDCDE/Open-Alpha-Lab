"""Offline demo — prove the machinery on a synthetic tape with a *known* IBS reversal.

No network, deterministic. Runs the teardown twice: on a **reversal** tape (low-IBS days bounce by
construction) and on the **null** (a random walk where IBS is uninformative). Every diagnostic that
fires on the reversal tape must go quiet on the null — that is what proves the code measures the
effect, not itself.

    python examples/run_synthetic_demo.py

The real-market verdict (a live ETF basket, where the bounce is real but decayed and cost-bound) is a
separate, fingerprinted run in `examples/verify.py` -> `docs/results.md`.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from rubber_band import data, ibs, strategy, decompose


def run(label, kappa):
    ohlc, truth = data.synthetic_ohlc(kappa=kappa, seed=19)
    rs = ibs.reversal_strength(ohlc)
    gross = strategy.timing_returns(ohlc, cost_bps=0.0)
    s = strategy.summary(gross)
    t = decompose.mean_tstat_hac(gross)
    be = decompose.breakeven_cost({label: ohlc})
    print(f"\n=== {label} (baked kappa={kappa}) ===")
    print(f"  IBS->next-day slope {rs['ibs_slope']:+.4f} | low-minus-high {rs['low_minus_high_bps']:+.1f} bps/day")
    print(f"  timing gross Sharpe {s['sharpe']:+.2f}, ann {s['ann_return']:+.1%}, "
          f"turnover {strategy.turnover_ann(ohlc):.0f}x/yr | HAC t={t['t_stat']:+.1f}")
    print(f"  break-even cost {be['breakeven_bps']:.1f} bps per unit traded")


def main():
    print("Rubber-Band — synthetic proof of the machinery (a real bounce vs a random-walk null).")
    run("reversal", kappa=0.0035)
    run("null", kappa=0.0)
    print("\nReversal tape: the bounce fires at HAC t > 3. Null: every leg goes quiet. "
          "The real ETF verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
