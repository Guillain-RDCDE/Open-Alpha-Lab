"""Offline, deterministic demo — Study 170 (Alphabetical-Bias).

No network.  Builds a synthetic cross-sectional panel and shows the study's
spine in one screen: the engine only detects an early-alphabet return premium
when one is *planted*, and on a null panel (no bias) it correctly finds nothing.
Also shows the volume channel: a planted attention premium is detectable even
when the return channel is silent.

    python studies/170-alphabetical-bias/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from alphabetical_bias import data, strategy as st  # noqa: E402


def _fmt(r: dict) -> str:
    t = r.get("tstat", float("nan"))
    m = r.get("mean_bps", float("nan"))
    sig = "DETECTED" if r.get("signal_detected", False) else "none"
    return f"mean={m:+7.2f} bps/mo  t={t:+5.2f}  → {sig}"


def main() -> None:
    print("Study 170 — Alphabetical-Bias — synthetic positive control\n")

    print("Return channel (A–C vs D–Z monthly equal-weight spread):")
    print(f"  {'scenario':>30s} | {'mean (bps/mo)':>13s} | {'HAC t':>7s} | result")
    print("  " + "-" * 68)

    scenarios = [
        ("null (no bias planted)",  0.0, 0.0),
        ("bias=5 bps/mo",          5.0, 0.0),
        ("bias=20 bps/mo",        20.0, 0.0),
        ("bias=50 bps/mo",        50.0, 0.0),
    ]
    for label, rbias, vbias in scenarios:
        panel, _ = data.synthetic_cross(
            n_stocks=300, n_months=240,
            return_bias_bps=rbias, volume_bias=vbias, seed=170,
        )
        result = st.detect_planted_bias(panel)
        t = result.get("tstat", float("nan"))
        m = result.get("mean_bps", float("nan"))
        detected = result.get("signal_detected", False)
        print(f"  {label:>30s} | {m:>+13.2f} | {t:>+7.2f} | {'DETECTED' if detected else 'not detected'}")

    print()
    print("Volume channel (log-volume ratio, A–C vs D–Z):")
    print(f"  {'scenario':>30s} | {'vol ratio':>10s} | {'HAC t':>7s}")
    print("  " + "-" * 55)
    for label, vol_bias in [("null (no volume bias)", 0.0), ("volume bias=20%", 0.20), ("volume bias=50%", 0.50)]:
        panel, _ = data.synthetic_cross(
            n_stocks=300, n_months=120,
            return_bias_bps=0.0, volume_bias=vol_bias, seed=170,
        )
        vr = st.volume_comparison(panel)
        ratio = vr.get("vol_ratio", float("nan"))
        t = vr.get("tstat", float("nan"))
        print(f"  {label:>30s} | {ratio:>10.3f} | {t:>+7.2f}")

    print()
    print("Bonferroni per-letter scan (null panel, 300 stocks × 240 months):")
    panel, _ = data.synthetic_cross(n_stocks=300, n_months=240, return_bias_bps=0.0, seed=170)
    per_letter = st.per_letter_returns(panel)
    n_sig = int(per_letter["bonferroni_significant"].sum())
    print(f"  Letters clearing |t| >= {st.BONFERRONI_T:.2f} (Bonferroni a/27): {n_sig}")
    print(f"  (Expected under null: ~0; one or two spurious hits are plausible)")

    print()
    print("The engine is a faithful detector: it only fires when a bias is planted.")
    print("On the real S&P 500 tape the return channel finds nothing — see docs/results.md.")


if __name__ == "__main__":
    main()
