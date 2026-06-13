"""Offline positive/negative control for Study 98 (High-Noon) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``ath_reversal = 0`` (a driftful random walk): all-time highs are just where an
  uptrend has been, so they cannot forecast *worse* forward returns — the at-ATH minus
  not-at-ATH difference must NOT be significantly negative.
- ``ath_reversal > 0`` (a planted mean-reversion-at-highs drag): an ATH genuinely
  forecasts a reversal — the difference must now be negative with a HAC \\|t\\| >= 2, and
  the avoid-the-highs timer must beat buy-and-hold. The positive control that proves the
  harness CAN detect "ATH is bearish" when it is true.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from high_noon import data, strategy  # noqa: E402

H = 3 * strategy.MONTH  # 3-month forward horizon


def _run(ath_reversal: float) -> dict:
    frame, truth = data.synthetic_daily(n_days=8000, ath_reversal=ath_reversal, seed=98)
    close = frame["close"]
    st = strategy.conditional_forward_stats(close, H, near_pct=0.01)
    pos = strategy.avoid_highs_position(close, near_pct=0.01, lag=1)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    return {"truth": truth, "st": st, "timer": timer, "bh": bh}


def main() -> None:
    for ath_reversal, label in [(0.0, "NULL  (random walk, highs benign)"),
                                (1.0, "PLANTED reversal-at-highs")]:
        r = _run(ath_reversal)
        st, t, b = r["st"], r["timer"], r["bh"]
        print(f"\n=== {label} | frac_at_ath={r['truth']['frac_near_high']:.2f} ===")
        print(f"  3-mo fwd  at-ATH mean {st['at']['mean']*100:+6.2f}%   "
              f"not-ATH mean {st['not_at']['mean']*100:+6.2f}%   "
              f"diff {st['diff_mean']*100:+6.2f}%  HAC t {st['diff_hac_t']:+.2f}")
        print(f"  avoid-highs timer: CAGR {t['cagr']*100:6.2f}%   "
              f"B&H: CAGR {b['cagr']*100:6.2f}%")
        bearish = st["diff_hac_t"] <= -2.0
        helps = t["cagr"] > b["cagr"]
        print(f"  -> ATH bearish? {'YES' if bearish else 'no'}   "
              f"avoid-highs helps? {'YES' if helps else 'no'}   "
              f"(expected: {'YES / YES' if ath_reversal > 0 else 'no / no'})")


if __name__ == "__main__":
    main()
