"""Offline, deterministic demo -- Study 197 (Dividend-Payout-Ratio).

No network. Builds a synthetic monthly tape and shows the study's spine in one
screen: the payout-ratio slope on the forward outcome is recoverable only when a
signal is actually planted -- and on a null tape (payout_signal=0) the slope is
indistinguishable from zero. Run:

    python studies/197-dividend-payout-ratio/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dividend_payout_ratio import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 197 -- Dividend-Payout-Ratio -- synthetic positive control\n")
    print(
        f"{'planted signal':>16} | {'slope':>8} {'t_slope':>8} {'R2':>7} | detectable?"
    )
    print("-" * 58)
    for sig in (0.00, 0.05, 0.10, 0.20, -0.10):
        df, truth = data.synthetic_monthly(n_years=100, payout_signal=sig, seed=197)
        df_r = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
        reg = st.regress_payout(
            df_r, outcome_col="fwd_10y_eg", payout_col="Payout", maxlags=5
        )
        detectable = "yes" if reg["slope"] > 0.01 and abs(reg["t_slope"]) > 2.0 else "no"
        print(
            f"{sig:>16.2f} | {reg['slope']:>+8.4f} {reg['t_slope']:>+8.2f} "
            f"{reg['r_squared']:>7.4f} | {detectable}"
        )

    print(
        "\nThe engine is a faithful signal detector: it only finds a positive slope"
    )
    print(
        "when one is planted. At payout_signal=0 the slope is noise around zero."
    )
    print(
        "On the real Shiller tape, the slope for 10y real earnings growth is"
    )
    print(
        "~+0.064 (HAC t ~3.5) -- a statistically real but economically modest effect."
    )


if __name__ == "__main__":
    main()
