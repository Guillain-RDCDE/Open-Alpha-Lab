"""Offline demo — the whole machine on the synthetic tape, no network.

Generates a toy book of sessions with a *known* afternoon momentum and a deliberately
uninformative IB-rejection flag, then runs the same teardown the real script runs:

    * the conditional close-red table (OC-red lifts it; the confluence does not beat OC-red);
    * the mechanical-vs-forecast split (most of "OC-red -> red close" is a head-start);
    * the IB-adds-nothing test (the baked-in null: the second signal is redundant);
    * the forking-paths Monte-Carlo (a modest true edge, mined, looks like a headline).

    python examples/run_synthetic_demo.py

This is the reproducible core: it proves the decomposition recovers what we *baked in*
(real momentum, null IB flag), so the real-data run (`verify.py`) is a measurement, not a hope.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from crimson_hour import data, decompose, signals


def main():
    feat, truth = data.synthetic_sessions(seed=0)
    print(f"synthetic tape: {len(feat)} sessions, baked-in momentum = {truth.momentum}, "
          f"baseline red = {truth.baseline_red:.1%}, IB flag is null = {truth.ib_is_null}")

    masks = signals.condition_masks(feat)
    tab = decompose.conditional_table(masks, signals.session_red(feat))
    print("\n[conditional P(session red | morning condition)]")
    print(tab.round(4).to_string())

    split = decompose.mechanical_vs_predictive(feat)
    print("\n[mechanical vs forecast]")
    print(f"  headline P(session red | OC-red)      = {split['headline_rate']:.1%} "
          f"(lift {split['headline_lift_pp']:+.1f} pp)")
    print(f"  forecast P(rest-of-day red | OC-red)  = {split['continuation_rate']:.1%} "
          f"(lift {split['continuation_lift_pp']:+.1f} pp)")
    print(f"  -> {split['mechanical_share']:.0%} of the headline lift is a mechanical head-start")

    inc = decompose.ib_increment(feat)
    print("\n[does IB-rejection add anything over OC-red?]")
    print(f"  confluence {inc['confluence_rate']:.1%} ({inc['confluence_k']}/{inc['confluence_n']}) "
          f"vs OC-red-not-rejected {inc['control_rate']:.1%} ({inc['control_k']}/{inc['control_n']}): "
          f"increment {inc['increment_pp']:+.1f} pp, Fisher p={inc['fisher_p_value']:.2f}")

    post = decompose.beta_binomial(22, 25, thresholds=(0.7,))
    print("\n[the honest read on a quoted 22/25 = 88%]")
    print(f"  posterior mean {post['posterior_mean']:.1%}, "
          f"95% credible [{post['cred_low']:.1%}, {post['cred_high']:.1%}], "
          f"P(true rate > 70%) = {post['P(rate>0.7)']:.1%}")

    mining = decompose.mining_inflation(p_true=0.70, n_cond=25, n_candidates=12,
                                        observed=0.88, seed=0)
    print("\n[forking paths: a true 70% edge, best of 12 confluences on n=25]")
    print(f"  expected best {mining['expected_best_rate']:.1%}, p95 {mining['best_rate_p95']:.1%}, "
          f"P(best >= 88%) = {mining['P(best>=observed)']:.1%}")


if __name__ == "__main__":
    main()
