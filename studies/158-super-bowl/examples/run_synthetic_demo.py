"""Offline, deterministic demo — Study 158 (Super-Bowl).

No network. Uses the hardcoded Super Bowl table and a synthetic S&P 500
return series to show the study's spine in one screen: the Super Bowl
Indicator is exposed as a coincidence against an unconditional 70% up-rate,
not a predictive signal — detectable *only* when we plant an absurdly large
effect, and even then the honest baseline (70% up unconditionally) explains
almost everything. Run:

    python studies/158-super-bowl/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from super_bowl import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 158 — Super-Bowl Indicator — synthetic positive control\n")
    print(
        "The key insight: the S&P rises in ~70% of all years. Any binary signal\n"
        "that 'predicts up' will look like it 'works' -- because the market cooperates\n"
        "7 times in 10 regardless of what a football team does.\n"
    )

    # 1. Show the hardcoded Super Bowl table summary
    sb = data.superbowl_table()
    nfc_count = (sb["conference"] == "NFC").sum()
    afc_count = (sb["conference"] == "AFC").sum()
    print(f"Super Bowl history: {len(sb)} games (1967–2025)")
    print(f"  NFC wins: {nfc_count}  |  AFC wins: {afc_count}")
    print()

    # 2. Sweep planted signal strengths on synthetic data
    print(
        f"{'planted NFC premium':>22} | {'hit_rate%':>9} "
        f"{'uncond%':>8} {'binom_p':>8} {'perm_p':>7} | signal?"
    )
    print("-" * 72)
    for signal_bps in (0.0, 200.0, 500.0, 2000.0):
        df, _ = data.synthetic_annual(n_years=500, signal_bps=signal_bps, seed=158)
        df = df.rename(columns={"return": "sp500_return"})
        df["mkt_up"] = df["sp500_return"] > 0
        df["game_number"] = range(1, 501)
        df["winner"] = "SyntheticTeam"
        sig = st.signal_nfc(df)
        r = st.hit_rate_stats(df, sig, n_permutations=1_000, seed=158)
        detected = "YES" if r["binom_p"] < 0.05 and r["perm_p"] < 0.05 else "no"
        print(
            f"{signal_bps:>+22.0f} bps/yr | "
            f"{r['hit_rate_all']*100:>9.1f} "
            f"{r['uncond_up_rate']*100:>8.1f} "
            f"{r['binom_p']:>8.4f} "
            f"{r['perm_p']:>7.4f} | {detected}"
        )

    print()
    print("With 0 planted effect, hit-rate ~= 50% -- just noise.")
    print("You'd need a ~2,000 bps/yr NFC premium to 'detect' the indicator reliably.")
    print("The actual S&P up-rate is ~73% regardless of conference: see docs/results.md.")
    print()

    # 3. Show the brutal truth about n=59
    print("The n=59 problem:")
    print(f"  Total Super Bowls:  59")
    print(f"  Per-conference:    ~30 NFC, ~29 AFC observations")
    print(f"  At vol ~17%/yr, you need |t| >= 2 on n=30 to claim significance.")
    print(f"  That requires a mean difference of ~17% * 2 / sqrt(30) ~= 6.2%/yr.")
    print(f"  The actual NFC vs AFC gap is only a few percent -- well inside noise.")


if __name__ == "__main__":
    main()
