"""Offline, deterministic demo - Study 108 (ADX-Filter).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the ADX(14) > 25 gate applied to a 20/50 MA cross — three arms compared across
varying degrees of planted trend strength:

  GATED    = trade only when ADX > 25 at signal time
  UNGATED  = trade on every MA-cross signal
  RANDOM   = same entries as UNGATED, direction flipped by a fair coin

The result: on a martingale the ADX gate does nothing real; on a planted trend the
gate follows the trend (ADX does rise) but so does the unfiltered rule.  The
*differential* — gated minus ungated — is nearly always indistinguishable from zero,
which is the study's spine.  Run:

    python studies/108-adx-filter/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adx_filter import data, strategy as st  # noqa: E402


def _arm(bars, entries, seed=None):
    dirs = st.random_directions(len(entries), seed=seed) if seed is not None else None
    led = st.run_trades(bars, entries, hold_days=20, cost_bps=0, directions=dirs)
    return st.summarize(led, "ret_gross")


def main() -> None:
    print("Study 108 - ADX-Filter - synthetic positive-control sweep\n")
    header = (
        f"{'trend':>8} | {'ungated bps':>11} {'t':>6} "
        f"| {'gated bps':>9} {'t':>6} "
        f"| {'random bps':>10} {'t':>6} "
        f"| {'D(gated-ungated)':>17}"
    )
    print(header)
    print("-" * len(header))

    for ts in (0.00, 0.05, 0.10, 0.15, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=600, trend_strength=ts, seed=108)
        ent = st.ma_cross_entries(bars["close"])
        adx_s = st.adx(bars, n=14)
        gated = st.apply_adx_gate(ent, adx_s, threshold=25.0)

        ug = _arm(bars, ent)
        g = _arm(bars, gated)
        r = _arm(bars, ent, seed=1)

        delta = (g["mean_bps"] - ug["mean_bps"])
        print(
            f"{ts:>8.2f} | {ug['mean_bps']:>+11.2f} {ug['tstat']:>+6.2f} "
            f"| {g['mean_bps']:>+9.2f} {g['tstat']:>+6.2f} "
            f"| {r['mean_bps']:>+10.2f} {r['tstat']:>+6.2f} "
            f"| {delta:>+17.2f}  bps"
        )

    print("\nConclusion: the ADX gate reduces trade count without reliably improving")
    print("the per-trade edge -- gated ~= ungated ~= random across plausible trend strengths.")
    print("On the real tape see docs/results.md for the honest verdict.")


if __name__ == "__main__":
    main()
