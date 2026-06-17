"""Headline run for Study 220 (Small Dogs of the Dow) on the real point-in-time tape.

Computes annual returns for three strategies — the full Dogs (top-10 yield), the Small Dogs
(5 lowest-priced of the 10 Dogs), and the Foolish Four (Dogs #2-#5 by price ascending) —
against the DIA total-return benchmark. Charges 20 bps one-way on turnover. Stamps the
as-of date + content fingerprint.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from small_dogs import data, strategy  # noqa: E402

AS_OF = "2026-06-13"
COST_BPS = 20.0
FIRST_YEAR, LAST_YEAR = 2000, 2025


def main() -> None:
    cache = os.path.abspath(os.path.join(HERE, "..", "_cache"))
    panel = data.load_panel(end=AS_OF, cache_dir=cache)
    fp = data.fingerprint(panel)
    px = panel["prices"]
    print(
        f"[data] Dow panel: {px.shape[1]} tickers x {len(px):,} days  "
        f"bench {panel['bench'].index[0].date()} -> {panel['bench'].index[-1].date()}  "
        f"as-of {AS_OF}  fingerprint={fp}"
    )

    ann = strategy.annual_returns_from_panel(panel, FIRST_YEAR, LAST_YEAR, cost_bps=COST_BPS)
    n = len(ann)

    def stats(col, bench="dow"):
        s = strategy.summarize(ann, col=col, bench=bench)
        exc = (ann[col] - ann[bench]).to_numpy()
        t = strategy.hac_tstat(exc)
        lo, hi, p = strategy.block_bootstrap_ci(exc)
        ab = strategy.alpha_beta(ann, strat_col=col, bench_col=bench)
        return s, t, lo, hi, p, ab

    print(f"\n--- Three strategies vs Dow (DIA TR), {n} years, net {COST_BPS:.0f} bps/turnover ---")
    for col, label in [("dogs", "Dogs-10"), ("small_dogs", "Small-Dogs-5"), ("foolish4", "Foolish-Four")]:
        s, t, lo, hi, p, ab = stats(col, bench="dow")
        print(f"\n  {label}:")
        print(f"    CAGR {s['strat_cagr']*100:.2f}%  vs Dow {s['bench_cagr']*100:.2f}%  "
              f"gap {s['cagr_gap']*100:+.2f} pts/yr  grew $1->${s['total_strat']:.2f}")
        print(f"    Mean excess  {s['mean_excess']*100:+.2f}%/yr  "
              f"win {s['win_rate']*100:.0f}% ({int(round(s['win_rate']*n))}/{n} yrs)")
        print(f"    HAC t {t:+.2f}   boot CI [{lo*100:+.2f}%, {hi*100:+.2f}%]  p={p:.2f}")
        print(f"    alpha {ab['alpha']*100:+.2f}%/yr (t {ab['alpha_t']:+.2f})  "
              f"beta {ab['beta']:.2f}  R2 {ab['r2']:.2f}")

    print("\n--- Small Dogs vs full Dogs (incremental test) ---")
    s, t, lo, hi, p, ab = stats(col="small_dogs", bench="dogs")
    print(f"  Small-Dogs minus Dogs: mean {s['mean_excess']*100:+.2f}%/yr  "
          f"HAC t {t:+.2f}  CI [{lo*100:+.2f}%, {hi*100:+.2f}%]  p={p:.2f}")

    print("\n--- Foolish Four vs full Dogs (incremental test) ---")
    s, t, lo, hi, p, ab = stats(col="foolish4", bench="dogs")
    print(f"  Foolish-4 minus Dogs: mean {s['mean_excess']*100:+.2f}%/yr  "
          f"HAC t {t:+.2f}  CI [{lo*100:+.2f}%, {hi*100:+.2f}%]  p={p:.2f}")


if __name__ == "__main__":
    main()
