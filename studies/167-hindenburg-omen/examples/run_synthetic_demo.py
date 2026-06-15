"""Offline, deterministic demo — Study 167 (Hindenburg-Omen).

No network.  Builds a synthetic breadth panel and shows the study's spine:
- The signal fires rarely (breadth conditions are genuinely restrictive).
- Without a planted crash effect, signal-day forward returns are indistinguishable
  from the unconditional base rate — the omen has no predictive power.
- With a planted crash (crash_signal_bps > 0) the engine correctly recovers it.

Run:
    python studies/167-hindenburg-omen/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hindenburg_omen import data, strategy as st  # noqa: E402


def _fmt(x) -> str:
    if x != x:  # nan
        return "  n/a"
    return f"{x:+.2f}"


def run_one(crash_bps: float, label: str) -> None:
    df, truth = data.synthetic_breadth(
        n_stocks=200, n_days=1260, crash_signal_bps=crash_bps, seed=167
    )
    sig = st.cluster_signals(df, window_days=30)
    fwd_s = st.forward_returns(df, sig, horizons=(30, 60, 90, 120))
    fwd_b = st.unconditional_forward_returns(df, horizons=(30, 60, 90, 120))
    cr = st.crash_rate(df, sig, horizon=120, drawdown_threshold=-0.05)

    print(f"\n--- {label} ---")
    print(f"  signals: {truth['n_signals']} days  |  clusters: {int(sig.sum())}")
    print(f"  crash rate after signal : {cr['signal_crash_rate']:.1%}  "
          f"(base rate: {cr['base_crash_rate']:.1%})")
    print(f"  false-alarm rate: {cr['false_alarm_rate']:.1%}")
    print()
    print(f"  {'horizon':>8} | {'sig mean%':>10} | {'base mean%':>10} | {'Welch-t':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for h in (30, 60, 90, 120):
        col = f"ret{h}"
        s_arr = fwd_s[col].dropna().to_numpy()
        b_arr = fwd_b[col].dropna().to_numpy()
        s_mean = float(s_arr.mean() * 100) if len(s_arr) else float("nan")
        b_mean = float(b_arr.mean() * 100) if len(b_arr) else float("nan")
        t = st._welch_t(s_arr, b_arr)
        print(f"  {h:>7}d | {_fmt(s_mean):>10} | {_fmt(b_mean):>10} | {_fmt(t):>8}")


def main() -> None:
    print("Study 167 — Hindenburg-Omen — synthetic positive control\n")
    print("The engine is offline and deterministic.  Two scenarios:")
    print("  (A) crash_signal_bps = 0   -- the null; signal has no predictive power")
    print("  (B) crash_signal_bps = 100 -- planted crash effect; engine should detect it\n")

    run_one(0.0, "Null (no planted crash)")
    run_one(100.0, "Planted crash (100 bps/day negative drift for 30 days post-signal)")

    print("\nInterpretation:")
    print("  (A) signal and base means are similar, Welch-t near zero — null confirmed.")
    print("  (B) signal means should be substantially lower, |Welch-t| > 2 — engine works.")
    print()
    print("On the REAL tape the answer is closer to (A) — see docs/results.md.")


if __name__ == "__main__":
    main()
