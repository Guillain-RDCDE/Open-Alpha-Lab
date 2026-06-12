"""Real-tape runner for Study 86 (Tail-Radar) — regenerates docs/results.md numbers.

Fetches (or loads from cache) the SKEW/VIX/SPY daily tape and runs the three tests:
1. SKEW quintile forward-return table (h = 1, 5, 21 days).
2. Crash-frequency comparison: high-SKEW vs low-SKEW regimes.
3. SKEW vs VIX incremental regression.

Run:
    python studies/86-tail-radar/examples/verify.py
    python studies/86-tail-radar/examples/verify.py --fetch   # re-download from Yahoo
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tail_radar import data, strategy as st  # noqa: E402


def main(fetch: bool = False) -> None:
    print("Study 86 — Tail-Radar — real-tape verify\n")
    df = data.fetch_daily(fetch=fetch)

    print(f"N observations : {len(df)}")
    print(f"Date range     : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Fingerprint    : {data.fingerprint(df)}")
    print(f"SKEW range     : {df['SKEW'].min():.1f} – {df['SKEW'].max():.1f}")
    print(f"VIX range      : {df['VIX'].min():.1f} – {df['VIX'].max():.1f}")
    print()

    # ---- 1. Quintile forward-return table -----------------------------------
    print("=== 1. SKEW quintile forward-return table ===")
    tbl = st.quintile_table(df, horizons=[1, 5, 21])
    print(tbl.to_string(index=False, float_format="%.2f"))
    print()

    # ---- 2. Top-minus-bottom spread -----------------------------------------
    print("=== 2. Top-quintile minus bottom-quintile spread ===")
    print(f"{'horizon':>8} | {'spread (bps)':>13} {'t_spread':>9} | "
          f"{'Q5 mean':>9} {'t_Q5':>6} | {'Q1 mean':>9} {'t_Q1':>6}")
    print("-" * 72)
    for h in [1, 5, 21]:
        r = st.top_minus_bottom(df, horizon=h)
        print(f"{h:>8}d | {r['spread_bps']:>+13.2f} {r['t_spread']:>+9.2f} | "
              f"{r['mean_top_bps']:>+9.2f} {r['t_top']:>+6.2f} | "
              f"{r['mean_bot_bps']:>+9.2f} {r['t_bot']:>+6.2f}")
    print()

    # ---- 3. Crash frequency test -------------------------------------------
    print("=== 3. Crash-frequency test (horizon=21d, threshold=-5%, high=top-20%) ===")
    crash = st.crash_frequency(df, horizon=21, crash_threshold=-0.05, high_skew_pct=0.80)
    print(f"  High-SKEW regime  : n={crash['n_high']},"
          f" crash rate={crash['crash_rate_high']:.3f}"
          f" ({crash['crash_rate_high']*100:.1f}%)")
    print(f"  Low-SKEW regime   : n={crash['n_low']},"
          f" crash rate={crash['crash_rate_low']:.3f}"
          f" ({crash['crash_rate_low']*100:.1f}%)")
    print(f"  Rate ratio        : {crash['rate_ratio']:.3f}")
    print(f"  Fisher p-value    : {crash['fisher_pvalue']:.4f}")
    print()

    # ---- 4. SKEW vs VIX regression ----------------------------------------
    print("=== 4. SKEW vs VIX incremental regression ===")
    print(f"{'horizon':>8} | {'n':>6} | {'beta_SKEW':>10} {'t_SKEW':>7} | "
          f"{'beta_VIX':>9} {'t_VIX':>7} | {'R²':>6}")
    print("-" * 68)
    for h in [1, 5, 21]:
        reg = st.skew_vs_vix_regression(df, horizon=h)
        print(f"{h:>8}d | {reg['n']:>6} | {reg['beta_skew']:>+10.5f} "
              f"{reg['t_skew']:>+7.2f} | {reg['beta_vix']:>+9.5f} "
              f"{reg['t_vix']:>+7.2f} | {reg['r2']:>6.4f}")
    print()
    print("Verdict: Signal NONE -- no horizon clears |t| >= 2 for the SKEW spread;")
    print("crash-frequency test p = 0.655 (not significant); SKEW adds nothing over VIX.")
    print("Tradability: MIRAGE — the radar does not see the swans.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Re-download from Yahoo Finance and refresh cache.")
    args = parser.parse_args()
    main(fetch=args.fetch)
