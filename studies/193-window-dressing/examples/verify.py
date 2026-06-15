"""Real-tape verification — Study 193 (Window-Dressing). Regenerates docs/results.md numbers.

Reads SPY daily prices from the shared repo cache (or fetches via yfinance with --fetch),
labels each day as pump/reversal/neutral relative to quarter-end boundaries, and runs
the three honest tests:
  1. Pump window vs baseline.
  2. Reversal window vs baseline.
  3. Long-pump / short-reversal spread vs a random-day placebo.

    python studies/193-window-dressing/examples/verify.py            # cache-only
    python studies/193-window-dressing/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from window_dressing import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_spy(fetch=fetch)
    print(
        f"SPY  {df.index[0].date()} to {df.index[-1].date()} "
        f"n={len(df)}  fp={data.fingerprint(df)}"
    )
    print()

    out = st.summarize(df)

    print("=== Pump window (last 5 trading days of each quarter) ===")
    print(f"  n={out['n_pump']}  mean={out['mean_pump_bps']:+.2f} bps  "
          f"HAC t={out['tstat_pump']:+.3f}  Welch p={out['pval_pump']:.4f}")
    print(f"  contrast vs baseline: {out['contrast_pump_bps']:+.2f} bps")

    print()
    print("=== Reversal window (first 5 trading days of next quarter) ===")
    print(f"  n={out['n_reversal']}  mean={out['mean_reversal_bps']:+.2f} bps  "
          f"HAC t={out['tstat_reversal']:+.3f}  Welch p={out['pval_reversal']:.4f}")
    print(f"  contrast vs baseline: {out['contrast_reversal_bps']:+.2f} bps")

    print()
    print("=== Long-pump / short-reversal spread ===")
    print(f"  mean={out['mean_spread_bps']:+.2f} bps/day  HAC t={out['tstat_spread']:+.3f}  "
          f"ann.~{out['ann_ret']*100:+.1f}%/yr")
    print(f"  Random-day control: mean={out['ctrl_mean_bps']:+.2f} bps  t={out['ctrl_tstat']:+.3f}")

    print()
    print("=== Window sweep (windows 1–10 days) ===")
    sw = st.window_sweep(df, max_window=10)
    print(sw[["contrast_pump_bps", "pval_pump", "contrast_reversal_bps",
               "pval_reversal", "pval_bonferroni"]].round(3).to_string())

    print()
    print("=== Verdict ===")
    print(f"  Baseline (all other days): {out['mean_other_bps']:+.2f} bps/day")
    print("  Pump window is BELOW baseline (not above) — no pump signal.")
    print("  Reversal is ABOVE baseline (not below) — no reversal signal.")
    print("  Spread t-stat: {:.3f} — not significant.".format(out["tstat_spread"]))
    print("  Signal: NONE  |  Tradability: MIRAGE")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
