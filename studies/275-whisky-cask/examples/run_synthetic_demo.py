"""Offline synthetic demo -- Study 275 (Whisky-Cask). No cache, no network.

Shows the appraisal-smoothing artifact end to end: plant a known true Sharpe into a
synthetic alt-asset, smooth it, and confirm that (a) the reported Sharpe is inflated
and (b) Geltner unsmoothing recovers the truth.

    python studies/275-whisky-cask/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from whisky_cask import data, strategy as st  # noqa: E402


def main() -> None:
    rf, true_drift, true_vol = 0.02, 0.07, 0.18
    true_sharpe = (true_drift - rf) / true_vol
    print("Study 275 -- Whisky-Cask -- synthetic positive control (offline)\n")
    print(f"Planted TRUE Sharpe = {true_sharpe:.3f} (drift {true_drift:.0%}, vol {true_vol:.0%})\n")
    print(f"{'smoothing':>10} {'rho':>6} {'reported_Sharpe':>16} {'unsmoothed_Sharpe':>18}")
    for lam in [0.0, 0.3, 0.5, 0.7, 0.85]:
        df, _ = data.synthetic_index(n_years=400, true_drift=true_drift,
                                     true_vol=true_vol, smoothing=lam, seed=275)
        rep = df["reported_return"].to_numpy()
        rep_sh = (rep.mean() - rf) / rep.std(ddof=1)
        uns, rho = st.geltner_unsmooth(rep)
        uns_sh = (uns.mean() - rf) / uns.std(ddof=1)
        print(f"{lam:>10.2f} {rho:>6.2f} {rep_sh:>16.3f} {uns_sh:>18.3f}")
    print("\nReported Sharpe balloons with smoothing; unsmoothed Sharpe tracks the truth.")
    print("The real whisky index sits at the high-smoothing end of this table.")


if __name__ == "__main__":
    main()
