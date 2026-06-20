"""Real-tape verification — Study 343 (Data-Mining-Roulette). Regenerates docs/results.md.

Spins the roulette (N random rules) on the real tape (default SPY total-return daily) and
on the deterministic synthetic null + positive control, reporting for each:
  - the best rule's naive Sharpe / HAC t (what a p-hacker would brag about),
  - how many of N rules clear |t|>=2 by chance, vs the Bonferroni-corrected count,
  - White's (2000) Reality Check p-value on the best rule (stationary bootstrap).

    python studies/343-data-mining-roulette/examples/verify.py           # cache-only
    python studies/343-data-mining-roulette/examples/verify.py --fetch   # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_mining_roulette import data, strategy as st  # noqa: E402

N_RULES = 1000
COST_BPS = 0.0
RC_BOOT = 1000


def _report(tag, feats, fwd):
    res = st.run_experiment(feats, fwd, n_rules=N_RULES, cost_bps=COST_BPS,
                            rc_boot=RC_BOOT, seed=343)
    b = res["best"]
    pc = res["pass_counts"]
    rc = res["rc"]
    print(f"\n=== {tag} ===")
    print(f"days={len(fwd)}  buy&hold Sharpe={res['bh_sharpe']:.2f}")
    print(f"BEST rule: {b['feature']} {b['op']} q{b['q']}  "
          f"Sharpe={b['sharpe']:.2f}  HAC t={b['hac_t']:.2f}  "
          f"mean={b['mean_excess_ann']*100:+.1f}%/yr  naive p={b['naive_p']:.3g}")
    print(f"rules clearing |t|>=2 by chance: {pc['n_pass_naive']} of {pc['n_rules']}  "
          f"(expected under indep ~{pc['expected_false_positives']:.1f})")
    print(f"Bonferroni t-threshold: {pc['bonferroni_t']:.2f}  "
          f"-> rules passing Bonferroni: {pc['n_pass_bonferroni']}")
    print(f"White Reality Check p (best rule, stationary bootstrap): {rc['reality_check_p']:.3f}")
    return res


def main(fetch: bool) -> None:
    # Synthetic null (nothing to find) and positive control (a planted edge).
    f0, r0, _ = data.synthetic_tape(planted_edge=0.0, vol_cluster=0.6, seed=343)
    _report("SYNTHETIC NULL (planted_edge=0)", f0, r0)

    f1, r1, _ = data.synthetic_tape(planted_edge=0.10, vol_cluster=0.6, seed=343)
    _report("SYNTHETIC POSITIVE CONTROL (planted_edge=0.10)", f1, r1)

    # Real tape.
    try:
        feats, fwd = data.load_real(fetch=fetch)
        print(f"\nreal tape: {feats.index[0].date()} .. {feats.index[-1].date()}  "
              f"({len(feats)} days)")
        print(f"fingerprint: {data.fingerprint(feats, fwd)}")
        _report("REAL TAPE (SPY)", feats, fwd)
    except Exception as e:  # offline / no cache
        print(f"\n[real tape skipped: {e}]")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
