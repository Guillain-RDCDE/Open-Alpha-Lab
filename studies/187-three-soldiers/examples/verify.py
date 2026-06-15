"""Real-tape verification — Study 187 (Three-Soldiers). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the 15-ticker basket, detects Three White Soldiers and
Three Black Crows, measures forward returns vs a random-day baseline, and reports
pattern counts, excess returns, and multiple-comparisons-corrected t-stats.
Network is touched only with --fetch.

    python studies/187-three-soldiers/examples/verify.py            # cache-only
    python studies/187-three-soldiers/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from three_soldiers import data, strategy as st  # noqa: E402

TICKERS = data.DEFAULT_TICKERS
HORIZONS = (1, 5, 10)
BONFERRONI_N = len(st.PATTERN_NAMES) * len(HORIZONS)  # 6


def main(fetch: bool) -> None:
    if fetch:
        print("=== refreshing caches ===")
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:6s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== data stamps (cache-only) ===")
    for t in TICKERS:
        try:
            b = data.fetch_daily(t, fetch=False)
            print(f"{t:6s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b):5d} fp={data.fingerprint(b)}")
        except FileNotFoundError:
            print(f"{t:6s} NOT CACHED — run with --fetch")

    print()
    print("=== pattern counts and excess returns (pooled, cost=1bp) ===")
    pooled = st.run_pooled_study(TICKERS, fetch=False, horizons=HORIZONS, cost_bps=1.0, seed=0)

    for name in st.PATTERN_NAMES:
        if name not in pooled:
            print(f"\n{name}: NO TRIGGERS found")
            continue
        df = pooled[name]
        print(f"\n{'-'*70}")
        print(f"Pattern: {name.replace('_', ' ').title()} (claimed dir={st.PATTERN_DIR[name]:+d})")
        print(f"Total triggers (pooled): {len(df)}")
        for h in HORIZONS:
            s = st.summarize_pattern(df, horizon=h)
            if not s:
                continue
            t_exc = s.get("tstat_excess", float("nan"))
            surv = "*SURVIVES Bonferroni*" if abs(t_exc) >= 2.65 else ""
            print(
                f"  horizon {h:2d}d  n={s['n']:5d}  signal={s['mean_bps']:+7.2f}bps  "
                f"excess={s['excess_bps']:+7.2f}bps  t_excess={t_exc:+6.2f}  "
                f"t_signal={s['tstat_signal']:+6.2f}  win={s['win_rate']:.1%}  {surv}"
            )

    print()
    print(f"Multiple-comparisons note: Bonferroni n={BONFERRONI_N} -> "
          f"|t|_adj = 2.65 for alpha=5%.")
    print("Signal=REAL requires |t_excess| >= 2 raw and survival of Bonferroni correction.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
