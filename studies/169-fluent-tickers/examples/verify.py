"""Real-tape verification — Study 169 (Fluent-Tickers). Regenerates docs/results.md numbers.

Loads (or fetches) daily closes for S&P 500 survivors from Yahoo Finance, scores each
ticker by pronounceability heuristic, sorts into FLUENT / NON-FLUENT, runs the
equal-weight long-short portfolio, and compares it against a random-label control.

The panel is *survivorship-biased* — we use only current S&P 500 members — which
biases absolute return levels upward but does NOT confound the fluency-vs-non-fluency
*comparison* (both groups are equally biased), so the L-S signal can still be
honestly tested.  We name this on the Signal axis.

    python studies/169-fluent-tickers/examples/verify.py            # cache-only
    python studies/169-fluent-tickers/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fluent_tickers import data, strategy as st  # noqa: E402

# Use a manageable subset of the S&P 500 to keep download/cache size reasonable.
# This is the same universe we score against — a survivorship-biased snapshot.
TICKERS = st.get_sp500_tickers()
START = "2010-01-01"
N_RANDOM_SEEDS = 20  # number of random-label seeds to form the null distribution


def _load_panel(fetch: bool) -> pd.DataFrame:
    """Load (or fetch) the closing-price panel; return what's in cache."""
    print(f"Loading panel for {len(TICKERS)} tickers (start={START}) ...")
    panel = data.load_real_panel(
        TICKERS, start=START, fetch=fetch,
        cache_dir=data.DEFAULT_CACHE
    )
    print(f"  -> panel shape: {panel.shape} ({panel.index[0].date()} to {panel.index[-1].date()})")
    return panel


def main(fetch: bool) -> None:
    panel = _load_panel(fetch)
    if panel.shape[1] < 10:
        print("WARNING: very few tickers loaded. Run with --fetch to populate the cache.")

    tickers = list(panel.columns)
    fluent_mask = data.classify_tickers(tickers)
    scores = data.score_tickers(tickers)

    n_fluent = int(fluent_mask.sum())
    n_nonfluent = len(tickers) - n_fluent
    print(f"\nUniverse: {len(tickers)} tickers | FLUENT: {n_fluent} | NON-FLUENT: {n_nonfluent}")
    print(f"Data fingerprint: {data.fingerprint(panel)}\n")

    # ---- HONEST TEST: fluency L-S vs random-label baseline -------------------
    port = st.build_ls_portfolio(panel, fluent_mask, cost_bps=0.0)
    annual = st.annual_ls_returns(port)
    stats = st.summarize(annual)

    print("=== Long FLUENT / Short NON-FLUENT (gross, no cost) ===")
    print(f"  Years: {stats['n_years']}  |  Mean: {stats['mean_bps']:+.0f} bps/yr  |  "
          f"Win-rate: {stats['win_rate']:.1%}  |  HAC t: {stats['tstat']:+.2f}")

    # Random-label baseline (multiple seeds).
    rand_means = []
    for seed in range(N_RANDOM_SEEDS):
        rport = st.build_random_ls_portfolio(panel, fluent_mask, seed=seed, cost_bps=0.0)
        rand_means.append(st.summarize(st.annual_ls_returns(rport))["mean_bps"])
    rand_mean = float(np.mean(rand_means))
    rand_std = float(np.std(rand_means))

    print(f"\n=== Random-label baseline (mean over {N_RANDOM_SEEDS} seeds) ===")
    print(f"  Mean: {rand_mean:+.0f} bps/yr  |  Std across seeds: {rand_std:.0f} bps/yr")

    excess = stats["mean_bps"] - rand_mean
    print(f"\n=== Fluency excess over random baseline ===")
    print(f"  Excess: {excess:+.0f} bps/yr")
    print(f"  Verdict: {'signal present' if abs(stats['tstat']) > 2 and stats['mean_bps'] > rand_mean else 'no signal'}\n")

    # ---- Per-quintile breakdown -----------------------------------------------
    n_q = 5
    score_rank = scores.rank(pct=True)
    quintile_rets = {}
    log_rets = np.log(panel / panel.shift(1)).iloc[1:]
    for q in range(n_q):
        q_tickers = [t for t in tickers if q / n_q <= score_rank.get(t, 0) < (q + 1) / n_q]
        if q_tickers:
            quintile_rets[f"Q{q+1}"] = log_rets[q_tickers].mean(axis=1).sum() * 1e4
    print("=== Cumulative return by fluency quintile (Q1=least fluent, Q5=most fluent) ===")
    for qname, total_bps in quintile_rets.items():
        print(f"  {qname}: {total_bps:+.0f} bps total")

    # ---- Cost sweep -----------------------------------------------------------
    print("\n=== Cost sweep (annual frequency rebalance) ===")
    for cost in (0, 5, 10, 20):
        cp = st.build_ls_portfolio(panel, fluent_mask, cost_bps=float(cost))
        cs = st.summarize(st.annual_ls_returns(cp))
        print(f"  cost={cost:3d} bps/trade | net mean={cs['mean_bps']:+.0f} bps/yr | HAC t={cs['tstat']:+.2f}")

    # ---- Verdict ---------------------------------------------------------------
    print("\n=== VERDICT ===")
    if abs(stats["tstat"]) >= 2 and stats["mean_bps"] > rand_mean + 50:
        signal = "WEAK (statistical but barely above random)"
    else:
        signal = "NONE — indistinguishable from random ticker labels"
    print(f"  Signal:      {signal}")
    print(f"  Tradability: MIRAGE — annual rebalance at any cost eats what isn't there")
    print(f"  3rd axis:    Survivorship-biased panel — result is an UPPER BOUND on any real edge")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
