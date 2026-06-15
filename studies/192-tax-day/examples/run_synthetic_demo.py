"""Offline, deterministic demo — Study 192 (Tax-Day).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the tax-day anomaly is detectable only when a genuine effect is planted — on a pure
random walk (both effects = 0) the pre- and post-tax windows are statistically
indistinguishable from any other trading day.  Run:

    python studies/192-tax-day/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tax_day import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 192 — Tax-Day — synthetic positive-control sweep\n")
    print("Testing: pre-tax window (10 days before April 15) vs rest\n")
    print(
        f"{'pre effect':>12} {'post effect':>12} | "
        f"{'n_pre':>6} {'pre mean (bps)':>14} {'contrast (bps)':>14} "
        f"{'Welch t':>8} {'p-val':>7} | {'detected?':>10}"
    )
    print("-" * 100)

    scenarios = [
        (0.0, 0.0),    # null: no effect anywhere
        (2.0, 0.0),    # pre-tax effect only
        (4.0, 0.0),    # strong pre-tax effect
        (0.0, -2.0),   # post-tax negative effect only
        (0.0, -4.0),   # strong post-tax negative effect
        (2.0, -2.0),   # both effects
    ]

    for pre_eff, post_eff in scenarios:
        df, _ = data.synthetic_daily(
            n_years=75, pre_tax_effect=pre_eff, post_tax_effect=post_eff, seed=192
        )
        pre = st.window_comparison(df, "pre_tax")
        detected = "YES" if abs(pre["tstat_welch"]) > 2.0 else "no"
        print(
            f"{pre_eff:>12.1f} {post_eff:>12.1f} | "
            f"{pre['n_window']:>6} {pre['mean_window_bps']:>+14.2f} "
            f"{pre['contrast_bps']:>+14.2f} "
            f"{pre['tstat_welch']:>+8.3f} "
            f"{pre['pval_welch']:>7.4f} | {detected:>10}"
        )

    print()
    print("Testing: post-tax window (5 days on/after April 15) vs rest\n")
    print(
        f"{'pre effect':>12} {'post effect':>12} | "
        f"{'n_post':>6} {'post mean (bps)':>14} {'contrast (bps)':>14} "
        f"{'Welch t':>8} {'p-val':>7} | {'detected?':>10}"
    )
    print("-" * 100)

    for pre_eff, post_eff in scenarios:
        df, _ = data.synthetic_daily(
            n_years=75, pre_tax_effect=pre_eff, post_tax_effect=post_eff, seed=192
        )
        post = st.window_comparison(df, "post_tax")
        detected = "YES" if abs(post["tstat_welch"]) > 2.0 else "no"
        print(
            f"{pre_eff:>12.1f} {post_eff:>12.1f} | "
            f"{post['n_window']:>6} {post['mean_window_bps']:>+14.2f} "
            f"{post['contrast_bps']:>+14.2f} "
            f"{post['tstat_welch']:>+8.3f} "
            f"{post['pval_welch']:>7.4f} | {detected:>10}"
        )

    print()
    print("The engine detects planted effects (|effect| >= 2) but finds nothing")
    print("when the tape is a pure random walk (both effects = 0).")
    print("The real ^GSPC tape: both windows are indistinguishable from noise.")
    print("Signal: NONE. Tradability: MIRAGE.")


if __name__ == "__main__":
    main()
