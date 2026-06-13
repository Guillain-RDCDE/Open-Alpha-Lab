"""Headline run for Study 96 (New-Year-Pop) on the real tapes.

Computes the January small-minus-large contrast, the Wilson hit-rate, the pre/post-2000
decay test, and the seasonal-timer tradability read on:

- the LONG sample: ^RUT vs ^GSPC, **price-only**, since 1990;
- the TRADABLE pair: IWM vs SPY, **total-return**, since 2000.

Prints the numbers the README front-card and notebooks quote, with as-of date and content
fingerprints. Re-run to reproduce; match the fingerprints to confirm you hold the same
tapes.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from new_year_pop import data, strategy  # noqa: E402

AS_OF = "2026-06-13"
COST_BPS = 10.0
SPLIT_YEAR = 2000


def _report(label: str, frame, price_only: bool) -> None:
    # Drop any in-progress final month: stamp only complete calendar months.
    frame = frame.loc[:AS_OF]
    fp = data.fingerprint(frame)
    kind = "price-only" if price_only else "total-return"
    print(f"\n=== {label} ({kind}) ===")
    print(f"[data] {len(frame)} months  {frame.index[0].date()} -> {frame.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    c = strategy.contrast(frame)
    print(f"  Jan mean spread (small-large)   = {c['jan_mean']*100:+.2f}%/mo")
    print(f"  Rest-of-year mean spread        = {c['rest_mean']*100:+.2f}%/mo")
    print(f"  Jan-minus-rest contrast         = {c['diff']*100:+.2f} pts  (HAC t = {c['t_contrast']:+.2f})")
    print(f"  Jan small>large hit-rate        = {c['jan_wins']}/{c['n_jan']} "
          f"({c['jan_hit_rate']*100:.0f}%)  Wilson95 [{c['wilson_lo']*100:.0f}%, {c['wilson_hi']*100:.0f}%]")

    sub = strategy.subperiod_january(frame, split_year=SPLIT_YEAR)
    print(f"  Jan spread pre-{SPLIT_YEAR}  ({sub['early_n']:2d} yrs) = {sub['early_mean']*100:+.2f}%/mo")
    print(f"  Jan spread {SPLIT_YEAR}+    ({sub['late_n']:2d} yrs) = {sub['late_mean']*100:+.2f}%/mo")
    print(f"  Decay (early-late) difference   = {sub['diff']*100:+.2f} pts  (Welch t = {sub['t_diff']:+.2f})")

    tim = strategy.seasonal_timer(frame, cost_bps=COST_BPS)
    t, sbh, lbh = tim["timer"], tim["small_bh"], tim["large_bh"]
    print(f"  Seasonal timer  CAGR {t['cagr']*100:6.2f}%  Sharpe {t['sharpe']:.3f}  maxDD {t['max_dd']*100:6.1f}%")
    print(f"  Small buy&hold  CAGR {sbh['cagr']*100:6.2f}%  Sharpe {sbh['sharpe']:.3f}  maxDD {sbh['max_dd']*100:6.1f}%")
    print(f"  Large buy&hold  CAGR {lbh['cagr']*100:6.2f}%  Sharpe {lbh['sharpe']:.3f}  maxDD {lbh['max_dd']*100:6.1f}%")


def main() -> None:
    long = data.load_real_long("^RUT", "^GSPC")
    _report("LONG ^RUT vs ^GSPC since 1990", long, price_only=True)

    trad = data.load_real_tradable("IWM", "SPY")
    _report("TRADABLE IWM vs SPY since 2000", trad, price_only=False)


if __name__ == "__main__":
    main()
