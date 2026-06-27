"""Real-tape verification — Study 536 (Anomaly-Decay-Post-Publication).

Splits each classic anomaly's long-short at its publication year on a fixed survivor
basket and reports the pre vs post means, the one-sample *t* of each leg, the
decay ratio (post/pre), a label-shuffle placebo *p*, and the net-of-cost picture —
alongside a deterministic synthetic positive control that recovers a PLANTED decay.

    python studies/536-anomaly-decay-post-publication/examples/verify.py           # cache-only
    python studies/536-anomaly-decay-post-publication/examples/verify.py --fetch   # refresh the basket

Regenerates the numbers behind docs/results.md.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly_decay import data, strategy as st  # noqa: E402

COST_BPS = 10.0
BORROW_BPS = 50.0


def _fmt_leg(d):
    return f"n={d['n']:4d}  mean/mo={d['mean']*100:+.3f}%  ann={d['ann']*100:+.2f}%  t={d['t']:+.2f}"


def main(fetch: bool) -> None:
    # --- Synthetic controls (faithful-engine / power check, NOT a real-tape claim) ---
    print("=== SYNTHETIC CONTROLS (split-engine faithfulness, 20-seed averaged) ===")
    for pd_ in (0.0, 0.5, 1.0):
        sc = st.synthetic_control(post_decay=pd_, n_seeds=20)
        tag = {0.0: "no decay", 0.5: "half decay (McLean-Pontiff)", 1.0: "full decay (null)"}[pd_]
        print(f"  post_decay={pd_:.1f} [{tag:28s}] expected ratio={sc['expected_ratio']:.2f}  "
              f"recovered={sc['recovered_ratio_mean']:+.3f}  "
              f"pre_t={sc['pre_t_mean']:+.2f}  post_t={sc['post_t_mean']:+.2f}")

    # --- Real tape ---
    try:
        panel = data.load_real(fetch=fetch)
    except Exception as e:
        print(f"\n[real tape skipped: {e}]")
        return

    yrs = panel.index
    print(f"\nreal panel: {yrs[0]} .. {yrs[-1]}  "
          f"({panel.shape[0]} months, {panel.shape[1]} names)")
    print(f"fingerprint: {data.fingerprint(panel)}")

    res = st.run_panel(panel, data.ANOMALIES, cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS)

    print("\n=== PER-ANOMALY PUBLICATION SPLIT (gross long-short) ===")
    for name, meta in data.ANOMALIES.items():
        r = res["per"][name]
        g = r["gross"]
        n = r["net"]
        print(f"\n--- {name} ({meta['label']}, published {meta['pub_year']}) ---")
        print(f"  full: {r['n_months']} months, ann={r['full_mean_ann']*100:+.2f}%, t={r['full_t']:+.2f}, "
              f"avg turnover={r['avg_turnover']:.2f}")
        print(f"  PRE : {_fmt_leg(g['pre'])}")
        print(f"  POST: {_fmt_leg(g['post'])}")
        print(f"  decay ratio (post/pre) = {g['decay_ratio']:+.2f}   "
              f"decay = {g['decay_pct']*100:+.0f}%   placebo p = {r['placebo']['p']:.3f}")
        print(f"  NET (10bps/leg + 50bps borrow): pre ann={n['pre']['ann']*100:+.2f}%, "
              f"post ann={n['post']['ann']*100:+.2f}%, post t={n['post']['t']:+.2f}")

    print("\n=== CROSS-ANOMALY DECAY SUMMARY ===")
    print(f"  anomalies tested      : {res['n_anomalies']}")
    print(f"  median decay ratio    : {res['median_decay_ratio']:+.2f}  "
          f"(McLean-Pontiff benchmark ~0.50)")
    print(f"  mean decay ratio      : {res['mean_decay_ratio']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
