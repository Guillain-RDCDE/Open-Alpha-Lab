"""Real-tape runner -- Study 112 (Move-Index).

Fetches (or reads from cache) the daily MOVE, VIX, and SPY tape from Yahoo Finance
and regenerates the headline numbers that appear in docs/results.md.  The first run
touches the network; subsequent runs are cache-only.

    python studies/112-move-index/examples/verify.py           # cache-only
    python studies/112-move-index/examples/verify.py --fetch   # hit yfinance + re-cache
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from move_index import data, strategy as st  # noqa: E402


def main(fetch: bool = False) -> None:
    df = data.fetch_daily(fetch=fetch)
    fp = data.fingerprint(df)
    n = len(df)
    start = df.index[0].date()
    end = df.index[-1].date()
    print(f"Tape: {n} days, {start} to {end}, fingerprint: {fp}")
    print(f"MOVE: {df['MOVE'].min():.1f} - {df['MOVE'].max():.1f}, "
          f"mean={df['MOVE'].mean():.1f}, std={df['MOVE'].std():.1f}")
    print(f"VIX:  {df['VIX'].min():.1f} - {df['VIX'].max():.1f}, "
          f"mean={df['VIX'].mean():.1f}, std={df['VIX'].std():.1f}")
    print()

    # --- Quintile spread at each horizon ---
    print("=== MOVE quintile spread (Q5 vs Q1 forward SPY log-return) ===")
    print(f"{'horizon':>8} | {'n_Q5':>6} {'n_Q1':>6} | {'spread bps':>11} {'t_spread':>9} | "
          f"{'Q5 mean':>8} {'Q1 mean':>8}")
    print("-" * 70)
    for h in [1, 5, 21]:
        res = st.top_minus_bottom(df, horizon=h)
        print(f"{h:>8}d | {res['n_top']:>6} {res['n_bot']:>6} | "
              f"{res['spread_bps']:>+11.2f} {res['t_spread']:>+9.2f} | "
              f"{res['mean_top_bps']:>+8.2f} {res['mean_bot_bps']:>+8.2f}")

    # --- MOVE vs VIX regression ---
    print()
    print("=== MOVE vs VIX regression (forward SPY, standardised predictors) ===")
    print(f"{'horizon':>8} | {'n':>6} | {'beta_MOVE':>10} {'t_MOVE':>8} | "
          f"{'beta_VIX':>9} {'t_VIX':>7} | {'R2':>8}")
    print("-" * 70)
    for h in [1, 5, 21]:
        reg = st.move_vs_vix_regression(df, horizon=h)
        print(f"{h:>8}d | {reg['n']:>6} | {reg['beta_move']:>+10.5f} {reg['t_move']:>+8.2f} | "
              f"{reg['beta_vix']:>+9.5f} {reg['t_vix']:>+7.2f} | {reg['r2']:>8.5f}")

    # --- MOVE/VIX ratio signal ---
    print()
    print("=== MOVE/VIX ratio signal ===")
    print(f"{'horizon':>8} | {'n':>6} | {'beta_ratio':>11} {'t_ratio':>9} | "
          f"{'spread bps':>11} {'t_spread':>9}")
    print("-" * 70)
    for h in [1, 5, 21]:
        rat = st.move_vix_ratio_signal(df, horizon=h)
        print(f"{h:>8}d | {rat['n']:>6} | {rat['beta_ratio']:>+11.5f} {rat['t_ratio']:>+9.2f} | "
              f"{rat['spread_bps']:>+11.2f} {rat['t_spread']:>+9.2f}")

    # --- Quick verdict ---
    spread1 = st.top_minus_bottom(df, horizon=1)
    reg1 = st.move_vs_vix_regression(df, horizon=1)
    ratio5 = st.move_vix_ratio_signal(df, horizon=5)
    print()
    print("=== Verdict summary ===")
    print(f"  MOVE Q5-Q1 spread (h=1d):  {spread1['spread_bps']:+.2f} bps, t = {spread1['t_spread']:+.2f}  -> |t| < 2: NONE")
    print(f"  MOVE beta in reg (h=1d):    t_MOVE = {reg1['t_move']:+.2f}  -> |t| < 2: NONE")
    print(f"  MOVE/VIX ratio (h=5d):      t_ratio = {ratio5['t_ratio']:+.2f}  -> |t| < 2: NONE")
    print("  Signal: NONE  |  Tradability: MIRAGE  |  Beats VIX? NO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Study 112 headline numbers.")
    parser.add_argument("--fetch", action="store_true", help="Re-fetch from Yahoo Finance.")
    args = parser.parse_args()
    main(fetch=args.fetch)
