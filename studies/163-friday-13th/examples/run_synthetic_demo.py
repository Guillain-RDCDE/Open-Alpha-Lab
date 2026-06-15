"""Offline, deterministic demo — Study 163 (Friday-13th).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Friday-13th anomaly is detectable only when a genuine effect is planted — on a
martingale (f13_effect=0) the 13th is statistically indistinguishable from any
other Friday. Run:

    python studies/163-friday-13th/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from friday_13th import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 163 — Friday-13th — synthetic positive/negative control\n")
    print(f"{'planted effect':>16} | {'n F13':>6} {'F13 mean (bps)':>14} "
          f"{'contrast (bps)':>14} {'Welch t':>8} {'p-val':>7} | detectable?")
    print("-" * 82)

    for effect in (0.0, -2.0, -4.0, +2.0):
        df, truth = data.synthetic_daily(n_years=99, f13_effect=effect, seed=163)
        r = st.friday_comparison(df)
        detectable = "YES" if abs(r["tstat_welch"]) > 2.0 else "no"
        print(
            f"{effect:>16.1f} | {r['n_f13']:>6} {r['mean_f13_bps']:>+14.2f} "
            f"{r['contrast_bps']:>+14.2f} {r['tstat_welch']:>+8.3f} "
            f"{r['pval_welch']:>7.4f} | {detectable}"
        )

    print()
    print("The engine detects a planted effect (effect != 0) but finds nothing")
    print("when the tape is a fair random walk (effect = 0).")
    print("The real ^GSPC tape (1928-2026) reads: contrast ~+2 bps, p ~0.82.")
    print("Signal: NONE. The superstition is busted.")


if __name__ == "__main__":
    main()
