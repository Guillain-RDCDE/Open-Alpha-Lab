"""Offline, deterministic demo — Study 109 (OBV-Divergence).

No network. Builds synthetic daily tapes and shows the study's spine in one
screen: OBV signals only beat a random-direction coin when the tape actually
carries the planted structure (price momentum for signal A, volume lead for
signal B), and on the double null they do not. Run:

    python studies/109-obv-divergence/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from obv_divergence import data, strategy as st  # noqa: E402

SEED = 42
N_DAYS = 800
HORIZON = 5


def _edge(bars, sig, seed=1):
    """Compute (signal mean bps) - (random-direction mean bps) on the same entries."""
    led = st.run_signals(bars, sig, horizon=HORIZON, cost_bps=0.0)
    if len(led) == 0:
        return float("nan"), float("nan"), float("nan")
    cross_m = st.summarize(led, "ret_gross")
    # Replace direction with coin flip on same signal dates
    rand_dirs = st.random_directions(len(led), seed=seed)
    led_r = led.copy()
    led_r["ret_gross"] = rand_dirs * led["ret_gross"].abs()
    rand_m = st.summarize(led_r, "ret_gross")
    return cross_m["mean_bps"], cross_m["tstat"], cross_m["mean_bps"] - rand_m["mean_bps"]


def main() -> None:
    print("Study 109 — OBV-Divergence — synthetic positive control")
    print(f"  horizon = {HORIZON}d, n_days = {N_DAYS}, seed = {SEED}\n")

    print("=== Signal A: OBV trend/cross (OBV vs SMA(OBV,20)) ===")
    print(f"{'price_momentum':>16} | {'signal bps':>10} {'t':>6} | cross-coin bps | interpretation")
    print("-" * 75)
    for mom in (0.00, 0.10, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=N_DAYS, price_momentum=mom, vol_lead=0.0, seed=SEED)
        sig = st.signal_obv_trend(bars, obv_sma_n=20)
        m, t, delta = _edge(bars, sig)
        interp = "OBV tracks price; has structure to exploit" if mom > 0 else (
            "wrong side of reversion" if mom < 0 else "null — fair die")
        print(f"{mom:>16.2f} | {m:>+10.2f} {t:>+6.2f} | {delta:>+14.2f} | {interp}")

    print()
    print("=== Signal B: OBV-price divergence reversal (lookback=10d) ===")
    print(f"{'vol_lead':>10} | {'signal bps':>10} {'t':>6} | cross-coin bps | interpretation")
    print("-" * 75)
    for vl in (0.0, 0.2, 0.5, 1.0):
        bars, _ = data.synthetic_daily(n_days=N_DAYS, price_momentum=0.0, vol_lead=vl, seed=SEED)
        sig = st.signal_obv_divergence(bars, lookback=10, obv_sma_n=5)
        m, t, delta = _edge(bars, sig)
        interp = f"vol_lead={vl}: volume genuinely leads price" if vl > 0 else "null — no vol lead"
        print(f"{vl:>10.2f} | {m:>+10.2f} {t:>+6.2f} | {delta:>+14.2f} | {interp}")

    print()
    print("Conclusion: on the real tape volume does not lead price — see docs/results.md.")


if __name__ == "__main__":
    main()
