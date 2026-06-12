"""Offline, deterministic demo — Study 83 (Half-Life).

No network. Builds a synthetic daily BTC tape and shows the study's spine in one screen:
the post-halving window trade only beats a random-day control when the tape actually
carries a planted halving boost — and on the null tape (boost=0) the halving dates
carry no special information. This directly falsifies the 'n=4 is enough' claim. Run:

    python studies/83-half-life/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from half_life import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 83 — Half-Life — synthetic positive control\n")
    print("The question: do halving dates predict outsized 1-year returns on BTC?")
    print("The control: how often does a *random* 3-window bundle do the same?\n")

    halvings = data.HALVING_DATES[:3]  # 2012 is outside Yahoo range; use 3 for consistency
    n_draws = 500

    print(f"{'halving_boost':>16} | {'halving mean logret':>20} {'ctrl p50':>10} {'p-value':>9} | beats random?")
    print("-" * 80)

    for boost in (0.0, 0.5, 1.0, 2.0, 3.0):
        bars, _ = data.synthetic_daily(n_years=12, halving_boost=boost, seed=83)
        ev = st.window_returns(bars, halvings=halvings)
        ctrl = st.random_control_returns(bars, n_events=len(ev), seed=83, n_draws=n_draws)
        pa = st.power_analysis(bars, halvings=halvings, n_draws=n_draws)
        beats = "YES" if pa["p_value"] < 0.10 else "no"
        print(
            f"{boost:>16.1f} | {ev['log_ret'].mean():>+20.3f} {ctrl.median():>+10.3f} "
            f"{pa['p_value']:>9.3f} | {beats}"
        )

    print("\n--- Even the NULL tape (boost=0): halving windows vs t-stat ---")
    null_bars, _ = data.synthetic_daily(n_years=12, halving_boost=0.0, seed=83)
    ev = st.window_returns(null_bars, halvings=halvings)
    s = st.summarize(ev["log_ret"].to_numpy(), label="null tape halving windows")
    print(f"  n events      : {s['n']}")
    print(f"  mean log-ret  : {s['mean']:+.3f}")
    print(f"  HAC t-stat    : {s['tstat']:+.3f}  (|t| << 2; n=3 has near-zero power)")
    print(f"  low-power flag: {s['low_power_warning']}")

    print("\nConclusion: with n=3 observed halvings and a secular BTC bull market,")
    print("the halving windows are indistinguishable from random lucky windows.")
    print("See docs/results.md for the real-tape verdict.")


if __name__ == "__main__":
    main()
