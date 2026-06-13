"""Headline run for Study 92 (Easy-Money) on the real tapes.

Documents (1) the relentless DECAY of the long-vol ETP VIXY (its CAGR, total wipeout,
worst day) — the contango carry is real and large — and (2) the SHORT-VOL carry that
tries to harvest it, net of trade cost + daily borrow, with the TAIL statistics the
headline Sharpe hides (skew, excess kurtosis, worst day, max drawdown), plus the
leverage that survives the worst day and what the drip is worth at that size.

Prints the numbers the README front-card and the notebooks quote, and stamps the
as-of date + content fingerprints. Re-run to reproduce; match the fingerprints.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from easy_money import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 5.0
BORROW_BPS = 300.0


def main() -> None:
    vixy = data.load_real("VIXY", mode="total_return").loc[:AS_OF]
    vix = data.load_real("^VIX", start="1990-01-01", mode="total_return").loc[:AS_OF]
    fp_vixy = data.fingerprint(vixy)
    fp_vix = data.fingerprint(vix)
    close = vixy["close"]

    print(f"[data] VIXY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp_vixy}")
    print(f"[data] ^VIX spot index : {len(vix):,} rows  "
          f"{vix.index[0].date()} -> {vix.index[-1].date()}  fingerprint={fp_vix}")

    # (1) The decay of the long-vol ETP — is the carry real?
    d = strategy.decay_stats(close)
    print("\n--- (1) The long-vol ETP (VIXY) decay -- the carry IS real ---")
    print(f"  total return over window : {d['total_return']*100:,.1f}%  "
          f"(ends at {d['final_frac_of_start']*100:.4f}% of its start)")
    print(f"  CAGR                     : {d['cagr']*100:.2f}%/yr")
    print(f"  annualised vol           : {d['vol']*100:.1f}%")
    print(f"  worst day / best day     : {d['worst_day']*100:+.1f}% / {d['best_day']*100:+.1f}%")

    # (2) The short-vol carry net of costs + borrow, at 1x.
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=COST_BPS,
                                     borrow_bps_annual=BORROW_BPS)
    t = strategy.hac_tstat(bt["net"].to_numpy())
    print(f"\n--- (2) Short-vol carry (short VIXY 1x), {COST_BPS:.0f} bps/turn + "
          f"{BORROW_BPS:.0f} bps/yr borrow ---")
    print(f"  CAGR        : {bt['cagr']*100:+.2f}%/yr")
    print(f"  vol         : {bt['vol']*100:.1f}%")
    print(f"  Sharpe      : {bt['sharpe']:.3f}   (HAC t on daily ret = {t:+.2f})")
    print(f"  max drawdown: {bt['max_dd']*100:.1f}%")
    print(f"  skew        : {bt['skew']:+.2f}   excess kurtosis: {bt['excess_kurt']:+.1f}")
    print(f"  worst day   : {bt['worst_day']*100:+.1f}%   best day: {bt['best_day']*100:+.1f}%")
    print(f"  final equity: {bt['final']:.3f}x start")

    # (3) Sizing that survives the worst day.
    s = strategy.sized_to_survive(close, max_loss_on_worst_day=0.25,
                                  cost_bps=COST_BPS, borrow_bps_annual=BORROW_BPS)
    sb = s["backtest"]
    print("\n--- (3) Sized so the worst historical day costs <= 25% of capital ---")
    print(f"  survivable leverage      : {s['leverage']:.3f}x  "
          f"(worst up-day of VIXY = {s['worst_up_move']*100:+.1f}%)")
    print(f"  CAGR at that size        : {sb['cagr']*100:+.2f}%/yr")
    print(f"  Sharpe at that size      : {sb['sharpe']:.3f}")
    print(f"  max drawdown at that size: {sb['max_dd']*100:.1f}%")


if __name__ == "__main__":
    main()
