"""Offline, deterministic demo -- Study 165 (Chinese-Zodiac).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Dragon-year bonus is detectable only when one is planted -- on a null tape
(no zodiac seasonality) all 12 animals look the same. Run:

    python studies/165-chinese-zodiac/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 output on Windows so the print statements do not trip on special chars.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chinese_zodiac import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 165 -- Chinese-Zodiac -- synthetic positive control\n")
    print("Chinese New Year dates hardcoded: 1990-2026 (37 years, 12 animals)\n")

    # Show the first few CNY dates
    print("Sample CNY table (first 6 rows):")
    for yr, row in list(data.CNY_DF.iterrows())[:6]:
        print(f"  {yr}: {row['cny_date'].date()}  ({row['animal']})")
    print("  ...\n")

    print("Testing: do we find a Dragon-year effect only when one is planted?\n")
    print(f"{'Scenario':35s} | {'Dragon%':>9} {'Rest%':>8} {'Diff%':>8} {'Welch-t':>9} | Detected?")
    print("-" * 88)

    configs = [
        ("null (no bonus)",             0.0),
        ("mild Dragon bonus (+50bp/d)", 50.0),
        ("strong bonus (+200bp/d)",     200.0),
    ]

    for label, bonus in configs:
        ret, _ = data.synthetic_daily(n_years=36, dragon_bonus_bp=bonus, seed=165)
        result = st.dragon_vs_rest(ret)
        # Handle possible nan
        d_m = result.get("dragon_mean_pct", float("nan"))
        r_m = result.get("rest_mean_pct", float("nan"))
        di = result.get("diff_pct", float("nan"))
        wt = result.get("welch_t", float("nan"))
        n_d = result.get("n_dragon", 0)
        # Note: with n_dragon ~3, |t|>2 is not meaningful -- the 'detected' column
        # shows the raw t-threshold only; the small-n caveat is the whole point.
        t_flag = f"|t|={abs(wt):.2f}" if not (wt != wt) else "nan"
        print(
            f"{label:35s} | {d_m:>+9.1f} {r_m:>+8.1f} {di:>+8.1f} {wt:>+9.2f} | "
            f"n_dragon={n_d} ({t_flag})"
        )

    print()
    print("Pre-CNY rally test (null tape, 36 years of synthetic data):")
    ret, _ = data.synthetic_daily(n_years=36, dragon_bonus_bp=0.0, seed=165)
    stats = st.pre_cny_stats(ret)
    print(f"  n_events = {stats['n_events']}")
    print(f"  mean pre-CNY return = {stats['mean_ret_pct']:+.2f}%")
    print(f"  baseline (random windows) = {stats['baseline_mean_pct']:+.2f}%")
    print(f"  HAC t-stat = {stats['hac_t']:+.2f}")
    print()
    print("Zodiac-year effect (null tape): n per animal ~3 -- too small to say anything.")
    zs = st.zodiac_year_stats(ret)
    print(f"  Max |raw_t| across 12 animals: {zs['raw_t'].abs().max():.2f}")
    print(f"  Min Bonferroni-p: {zs['bonferroni_p'].min():.3f}")
    print()
    print("Key insight: with n_dragon~3, the Welch t is *always* noisy and unreliable.")
    print("A large t-stat on n=3 means nothing -- it flips with a single different year.")
    print("A large planted bonus does push the Dragon mean up, but 3 data points is too few")
    print("to distinguish a real effect from a random draw. This is the study's whole point.")
    print()
    print("On real FXI (2004-2026) with ~3 Dragon years, no honest test can be conclusive.")
    print("The zodiac-year claim is a small-sample mirage.")


if __name__ == "__main__":
    main()
