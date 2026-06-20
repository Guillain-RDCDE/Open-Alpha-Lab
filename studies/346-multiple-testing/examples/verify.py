"""Real-tape verification — Study 346 (Multiple-Testing). Regenerates docs/results.md.

Screens a battery of named calendar effects on the real tape (default SPY total-return
daily) and a deterministic synthetic null + positive control, reporting for each:
  - how many of the M effects clear naive |HAC t| >= 2 (the p-hacker's harvest),
  - how many survive Bonferroni / Holm (FWER) and Benjamini-Hochberg (FDR),
  - on the synthetic control, each procedure's realised power, false positives and FDR.

    python studies/346-multiple-testing/examples/verify.py           # cache-only
    python studies/346-multiple-testing/examples/verify.py --fetch   # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multiple_testing import data, strategy as st  # noqa: E402

ALPHA = 0.05
T_THRESH = 2.0


def _report_counts(tag, rets):
    res = st.run_experiment(rets, alpha=ALPHA, t_thresh=T_THRESH)
    c = res["counts"]
    print(f"\n=== {tag} ===")
    print(f"M effects={c['m_effects']}  naive |t|>2={c['naive']}  "
          f"Bonferroni={c['bonferroni']}  Holm={c['holm']}  BH(FDR)={c['bh']}")
    return res


def _report_score(tag, rets, truth):
    res = st.run_experiment(rets, truth, alpha=ALPHA, t_thresh=T_THRESH)
    print(f"\n=== {tag} ===")
    print(f"M={res['counts']['m_effects']}  real planted={res['score']['_n_real']}")
    for p in ("naive", "bonferroni", "holm", "bh"):
        s = res["score"][p]
        print(f"  {p:11s} reject={s['n_reject']:3d}  TP={s['true_pos']:3d}  "
              f"FP={s['false_pos']:3d}  power={s['power']:.2f}  FDR={s['fdr']:.2f}")
    return res


def main(fetch: bool) -> None:
    # Synthetic null: 100 effects, none real -> naive prints false positives, corrections 0.
    r0, t0, _ = data.synthetic_battery(n_days=2520, m_effects=100, n_true=0, seed=346)
    _report_counts("SYNTHETIC NULL (100 effects, 0 real)", r0)

    # Strong positive control: 10 real of 100 -> corrections recover all 10, 0 false.
    r1, t1, _ = data.synthetic_battery(n_days=2520, m_effects=100, n_true=10, seed=346)
    _report_score("SYNTHETIC CONTROL (strong: 10 real of 100)", r1, t1)

    # Weak/mixed control: 20 real of 200 -> the FWER-vs-FDR trade-off is visible.
    r2, t2, _ = data.synthetic_battery(n_days=2520, m_effects=200, n_true=20,
                                       edge=0.0025, seed=346)
    _report_score("SYNTHETIC CONTROL (weak: 20 real of 200, edge 0.0025)", r2, t2)

    # FWER under the global null (a machinery check).
    mc = st.fwer_monte_carlo(m_effects=100, n_days=2520, n_reps=200, seed=346)
    print("\n=== FWER MONTE CARLO (global null, 100 effects, 200 reps) ===")
    for p in ("naive", "bonferroni", "holm", "bh"):
        print(f"  {p:11s} FWER={mc['fwer'][p]:.3f}  avg false positives={mc['avg_false_positives'][p]:.2f}")

    # Real tape.
    try:
        rets, _ = data.load_real(ticker="SPY", fetch=fetch)
        print(f"\nreal tape: {rets.index[0].date()} .. {rets.index[-1].date()}  "
              f"({len(rets)} days, {rets.index.to_period('M').nunique()} months)")
        print(f"fingerprint: {data.fingerprint(rets)}")
        _report_counts("REAL TAPE (SPY calendar effects)", rets)
    except Exception as e:  # offline / no cache
        print(f"\n[real tape skipped: {e}]")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
