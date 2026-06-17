"""Offline, deterministic demo — Study 272 (Champagne-Index).

No network. Uses the synthetic generator to show the study's spine in one screen:
the Champagne Indicator is detectable *only* when a real euphoria->equity link is
planted and the sample is large; with a thin sample and a modest effect (like the
real tape) the robust HAC t-stat stays below 2. Run:

    python studies/272-champagne-index/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from champagne_index import data, strategy as st  # noqa: E402


def _merged(signal_bps: float, n_years: int, seed: int = 272):
    df, _ = data.synthetic_annual(n_years=n_years, signal_bps=signal_bps, seed=seed)
    df = df.rename(columns={"sp500_fwd_return": "sp500_return"})
    df["mkt_up"] = df["sp500_return"] > 0
    return df


def main() -> None:
    print("Study 272 — Champagne-Index — synthetic positive control\n")
    print(
        "The folklore: champagne booms (euphoria) -> weak forward equity returns.\n"
        "A real link shows up as a NEGATIVE predictive slope with |HAC t| >= 2.\n"
    )

    print(f"{'planted premium':>16} | {'n':>5} | {'slope':>8} {'corr':>7} "
          f"{'HAC t':>7} | detected?")
    print("-" * 60)
    # A small effect at real-tape size (n=25) stays UNCONFIRMED; the same effect
    # with a large sample is detected. The contrast is the whole point.
    for label, bps, n in [
        ("0 (null), n=800", 0.0, 800),
        ("small, n=25", 120.0, 25),
        ("small, n=800", 120.0, 800),
        ("medium, n=800", 400.0, 800),
        ("large, n=800", 1000.0, 800),
    ]:
        df = _merged(bps, n)
        o = st.predictive_ols(df)
        detected = "YES" if abs(o["t_hac"]) >= 2 else "no"
        print(f"{label:>16} | {o['n']:>5} | {o['slope']:>+8.4f} "
              f"{o['corr']:>+7.3f} {o['t_hac']:>+7.2f} | {detected}")

    print()
    print("Same small effect: UNCONFIRMED at n=25, CONFIRMED at n=800. Sample size,")
    print("not the method, is the binding constraint -- exactly the situation on the")
    print("real champagne tape (n=25, corr -0.33, HAC t -1.30, unconfirmed).")
    print("See docs/results.md for the real-data headline run.")


if __name__ == "__main__":
    main()
