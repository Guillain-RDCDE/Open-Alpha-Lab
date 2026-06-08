"""Offline demo — prove the detector on a series with a KNOWN cycle, no network.

Builds the synthetic log-VIX (fixed 80d & 40d cycles buried in AR(1) red noise), then shows
the whole machine doing its job: the periodogram clears the red-noise envelope at the
injected periods, the rolling period barely moves (a real clock), and the walk-forward
direction forecast beats the random-phase null. This is the control: it is where the method
*works*, so the flat verdict on the real VIX (see verify_real.py) is a fact about the market,
not a bug in the code.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from vix_cycles import backtest, data, robustness, spectral

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    series, injected = data.synthetic_cycle(seed=0)
    print(f"synthetic log-VIX: {len(series)} sessions, injected cycles "
          f"{[round(c.period) for c in injected]} sessions\n")

    env = spectral.red_noise_envelope(series, n_sim=1000, seed=0)
    print(f"AR(1) rho fitted to the series: {env['rho']:.3f}")
    peaks = spectral.significant_peaks(env, q=0.99)
    print("\nsignificant peaks (clear the 99% red-noise envelope):")
    print(peaks.round(3).to_string(index=False) if len(peaks) else "  (none)")
    print(f"\ndetection recall vs injected: "
          f"{robustness.detection_recall(env, injected, q=0.99):.2f}")

    print("\nred-noise p-values at the injected periods:")
    print(robustness.red_noise_pvalues(series, targets=(40.0, 80.0), n_sim=1000).round(4))

    stab = robustness.period_stability(series, band=(20.0, 200.0), window=1000, step=200)
    print(f"\nrolling dominant period: mean {stab['dominant_period'].mean():.1f}, "
          f"std {stab['dominant_period'].std():.1f} sessions (a fixed clock barely moves)")

    skill = backtest.oos_direction_skill(series, band=(20.0, 200.0), horizon=20,
                                         min_train=750, step=10, n_null=500, seed=0)
    print(f"\nwalk-forward direction skill (20d horizon): {skill['skill']:.3f}  "
          f"(null {skill['null_mean']:.3f}, p={skill['p_value']:.3f}, n={skill['n_calls']})")
    print("\n-> on a real cycle the machine lights up. The question is whether the VIX is one.")


if __name__ == "__main__":
    main()
