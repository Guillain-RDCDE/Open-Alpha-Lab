"""Real-tape verification — Study 105 (Coppock-Curve). Regenerates docs/results.md numbers.

Fetches (or reads from cache) monthly bars for ^GSPC and SPY, computes the Coppock Curve,
identifies buy signals (deduped turn-up-from-below-zero), measures 6- and 12-month forward
returns vs a random-timing control and buy-and-hold. Network is touched only with --fetch.

    python studies/105-coppock-curve/examples/verify.py            # cache-only
    python studies/105-coppock-curve/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coppock_curve import data, strategy as st  # noqa: E402

TICKERS = ["^GSPC", "SPY"]
COST_BPS = 5.0


def run_one(ticker: str, fetch: bool) -> None:
    bars = data.fetch_monthly(ticker, fetch=fetch)
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    n_sigs = int(sigs.sum())
    fp = data.fingerprint(bars)
    print(f"\n{'='*60}")
    print(f"{ticker}: {bars.index[0].date()} → {bars.index[-1].date()}  n_months={len(bars)}  fp={fp}")
    print(f"Coppock signals (deduped): {n_sigs}")
    if n_sigs > 0:
        for d in sigs[sigs].index:
            print(f"  {d.date()}  cc={cc[d]:+.4f}")

    for horizon in [6, 12]:
        cop_led = st.forward_returns(bars["close"], sigs, horizon=horizon, cost_bps=COST_BPS)
        rand_sigs = st.random_signals(n_sigs, bars["close"], seed=42)
        rand_led = st.forward_returns(bars["close"], rand_sigs, horizon=horizon, cost_bps=COST_BPS)
        bah = st.buy_and_hold_returns(bars["close"], horizon=horizon)

        cop = st.summarize(cop_led, "ret_gross")
        rand = st.summarize(rand_led, "ret_gross")
        bah_mean = float(bah.mean())

        print(f"\n  --- {horizon}-month forward returns (gross) ---")
        print(
            f"  Coppock: n={cop['n_signals']:2d}  hit={cop['hit_rate']:.3f}  "
            f"mean={cop['mean_log_ret']*100:+.1f}%  HAC t={cop['tstat']:+.2f}  "
            f"warn={cop['low_n_warning']}"
        )
        print(
            f"  Random:  n={rand['n_signals']:2d}  hit={rand['hit_rate']:.3f}  "
            f"mean={rand['mean_log_ret']*100:+.1f}%  HAC t={rand['tstat']:+.2f}"
        )
        print(f"  BaH:     n={len(bah):4d}  hit={(bah>0).mean():.3f}  mean={bah_mean*100:+.1f}%")
        print(f"  Cop − BaH mean: {(cop['mean_log_ret'] - bah_mean)*100:+.1f} pct-pts")
        print(f"  Cop − Rand mean: {(cop['mean_log_ret'] - rand['mean_log_ret'])*100:+.1f} pct-pts")
        print(f"  ⚠ n={cop['n_signals']} signals — inherently low-power inference; t-stat unreliable.")


def main(fetch: bool) -> None:
    if fetch:
        print("Warming cache...")
        for t in TICKERS:
            b = data.fetch_monthly(t, fetch=True)
            print(f"  {t}: {b.index[0].date()} → {b.index[-1].date()}  fp={data.fingerprint(b)}")
        print()

    for t in TICKERS:
        run_one(t, fetch=False)

    print(
        "\n⚠  Coppock Curve generates ~10-20 signals over 70 years — effective n is very small."
        "\n   Even a HAC |t|>2 at n=19 is fragile; the inference bar is in practice much higher."
        "\n   See docs/results.md for the full verdict discussion."
    )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
