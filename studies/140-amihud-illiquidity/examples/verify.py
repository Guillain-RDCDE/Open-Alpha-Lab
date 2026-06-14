"""Real-tape verification — Study 140 (Amihud-Illiquidity). Regenerates docs/results.md numbers.

Reads (or builds) the annual ILLIQ panel from Yahoo Finance daily data for S&P 500 names,
runs the quintile sort, computes the hedge return time series, and reports HAC t-stats and
the random-quintile control. Network is touched only with --fetch.

    python studies/140-amihud-illiquidity/examples/verify.py            # cache-only
    python studies/140-amihud-illiquidity/examples/verify.py --fetch    # rebuild from Yahoo

SURVIVORSHIP BIAS WARNING: the ticker universe is the *current* S&P 500 projected backwards
from the EDGAR cache. All results are upper bounds on the true live premium. The Amihud
premium famously lives in micro-caps; on this large-cap panel any positive finding is
primarily a survivorship-bias artefact, not a tradable edge.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from amihud_illiquidity import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    print("Study 140 — Amihud-Illiquidity — real tape verification")
    print("=" * 60)
    print()

    illiq, fwd_ret = data.fetch_panel(fetch=fetch)
    if illiq.empty:
        print("No cached panel found. Run with --fetch to download data.")
        return

    print(f"Panel: {len(illiq.index)} sort years ({illiq.index.min()}–{illiq.index.max()}), "
          f"up to {illiq.notna().sum(axis=1).max()} tickers per year.")
    print(f"ILLIQ fingerprint : {data.fingerprint(illiq)}")
    print(f"FwdRet fingerprint: {data.fingerprint(fwd_ret)}")
    print()

    # Quintile sort
    res = st.quintile_sort(illiq, fwd_ret)
    print("=== Quintile returns (annual, calendar year y+1 after sort year y) ===")
    print(res[["q1", "q2", "q3", "q4", "q5", "hedge", "market", "n"]].round(4).to_string())
    print()

    print("=== Headline statistics ===")
    for col in ["q1", "q2", "q3", "q4", "q5", "hedge", "market"]:
        s = st.summarize(res[col], label=col)
        print(f"{col:7s}: mean={s['mean']*100:+6.2f}%  vol={s['vol']*100:.2f}%  "
              f"sharpe={s['sharpe']:.3f}  HAC_t={s['tstat']:+6.3f}  hit={s['hit_rate']:.2f}")
    print()

    # Q5 vs market
    q5_mkt = res["q5"] - res["market"]
    sq5m = st.summarize(q5_mkt)
    q1_mkt = res["q1"] - res["market"]
    sq1m = st.summarize(q1_mkt)
    print(f"Q5 vs market: mean={sq5m['mean']*100:+6.2f}%  HAC_t={sq5m['tstat']:+.3f}")
    print(f"Q1 vs market: mean={sq1m['mean']*100:+6.2f}%  HAC_t={sq1m['tstat']:+.3f}")
    print()

    # Monotonicity
    means = [res[f"q{q}"].mean() for q in range(1, 6)]
    monotone = all(means[i] <= means[i + 1] for i in range(4))
    print(f"Monotone (Q1 < Q2 < Q3 < Q4 < Q5): {monotone}  "
          f"values: {[f'{m*100:.1f}%' for m in means]}")
    print()

    # Random control
    print("=== Random-quintile control (500 draws per year) ===")
    rand = st.random_hedge_returns(illiq, fwd_ret, n_draws=500, seed=140)
    p_val = float((rand >= res["hedge"].mean()).mean())
    print(f"Random hedge: mean={rand.mean()*100:+.2f}%  std={rand.std()*100:.2f}%")
    print(f"Empirical p-value (random >= real): {p_val:.4f}")
    print()

    # Compound returns
    print("=== Compound returns over full period ===")
    for col in ["q1", "q5", "market"]:
        cret = (1 + res[col]).prod() - 1
        n_yr = len(res)
        ann = (1 + cret) ** (1 / n_yr) - 1
        print(f"{col:7s}: compound={cret*100:.1f}%  annualized={ann*100:.2f}%/yr")
    print()

    print("SURVIVORSHIP-BIAS NOTE:")
    print("  The universe is the current S&P 500 projected backwards — firms that")
    print("  grew enough to survive and enter the index. The Q5 (illiquid) bucket")
    print("  within a large-cap universe is dominated by smaller/faster-growing")
    print("  S&P 500 members: their survival is guaranteed by the panel construction.")
    print("  This result is a survivorship-bias upper bound, not a live tradable edge.")
    print("  The true premium on an ex-ante large-cap universe would be absent or much weaker.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
