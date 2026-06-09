"""Offline demo — the whole machine on the synthetic tape, no network.

Generates two toy books of sessions and runs the same teardown the real script runs:

    * a panel WITH a genuine gamma effect (beta > 0): the raw regime gap is real AND a chunk of it
      survives the VIX control — the decomposition recovers the baked-in beta;
    * a panel with NO genuine effect (beta = 0): the raw gap is just as visible, but it *evaporates*
      once VIX is partialled out — the trenchcoat, caught.

    python examples/run_synthetic_demo.py

This is the reproducible core: it proves the decomposition recovers what we *baked in* (a real
gamma effect, or none), so the real-data run (`verify.py`) is a measurement, not a hope.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from gamma_gospel import data, decompose


def _show(panel, truth, label):
    print(f"\n=== {label}: {truth.n_sessions} sessions, "
          f"baked-in beta_de={truth.beta_de}, beta_vol={truth.beta_vol}, "
          f"VIX confound={'on' if truth.confounded else 'off'} ===")
    neg = panel["neg_gamma"].astype(bool)
    print(f"  VIX on neg-gamma days {panel.loc[neg,'vix'].mean():.1f} "
          f"vs pos-gamma {panel.loc[~neg,'vix'].mean():.1f}  (the confound)")
    for y, name in (("rv", "range vol"), ("de", "directional efficiency")):
        raw = decompose.regime_gap(panel, y)
        p = decompose.partial_over_vix(panel, y)
        print(f"  [{name}] raw gap {raw['gap']:+.4f} (t {raw['t']:+.1f})  ->  "
              f"surviving VIX-control {p['surviving_coef']:+.4f} (t {p['surviving_t']:+.1f}), "
              f"share kept {p['survival_share']:.0%}, dR2 {p['delta_r2']:+.3f}")


def main():
    real, truth_real = data.synthetic_panel(beta_vol=0.0020, beta_de=0.060, seed=0)
    mirage, truth_mirage = data.synthetic_panel(beta_vol=0.0, beta_de=0.0, seed=0)

    _show(real, truth_real, "GENUINE EFFECT")
    print("  -> a real gamma effect: the gap survives the VIX control, near the baked-in beta.")
    _show(mirage, truth_mirage, "TRENCHCOAT (no effect)")
    print("  -> no real effect: the raw gap is pure confound and collapses once VIX is controlled.")


if __name__ == "__main__":
    main()
