"""Real-tape verification — Study 768 (Charm-Decay). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY daily bars from 1993, anchors the pre-/post-OpEx charm
windows on the 3rd Friday of every month, tests the directional drift, the rally-then-fade
asymmetry, the quarterly restriction, the pre/post-2012 break, the calendar-randomisation
placebo, and the tradability of the charm overlay.

    python studies/768-charm-decay/examples/verify.py            # cache-only
    python studies/768-charm-decay/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from charm_decay import data, strategy as st  # noqa: E402

TICKER = "SPY"
AS_OF = "2026-07-10"


def main(fetch: bool) -> None:
    bars = data.fetch_daily(TICKER, start="1993-01-01", fetch=fetch)
    bars = bars[bars.index <= pd.Timestamp(AS_OF)]        # pin the as-of, drop partial tail
    idx = pd.DatetimeIndex(bars.index)

    print("=== DATA STAMP ===")
    print(f"{TICKER}: {idx[0].date()} to {idx[-1].date()}  n={len(bars)}  "
          f"fp={data.fingerprint(bars)}")
    od = data.opex_dates("1993-01-01", "2026-12-31")
    pre_m = data.pre_opex_mask(idx)
    post_m = data.post_opex_mask(idx)
    print(f"OpEx dates defined: {len(od)}  pre-window days: {int(pre_m.sum())}  "
          f"post-window days: {int(post_m.sum())}")

    # ---- Leg 1: directional drift -------------------------------------
    drift = st.charm_drift_test(bars)
    print("\n=== LEG 1 — PRE/POST-OPEX DRIFT ===")
    for k, lbl in (("pre", "Pre-OpEx (charm week)"), ("post", "Post-OpEx (give-back)")):
        r = drift[k]
        print(f"  {lbl:24s} mean={r['mean_window']*1e4:+.2f}bps  "
              f"base={r['mean_baseline']*1e4:+.2f}bps  diff={r['diff']*1e4:+.2f}bps  "
              f"HAC t={r['tstat']:+.2f}  n={r['n_window']}")

    # ---- Leg 2: rally-then-fade asymmetry -----------------------------
    asym = st.pre_post_asymmetry(bars)
    print("\n=== LEG 2 — RALLY-THEN-FADE ASYMMETRY (pre minus post) ===")
    print(f"  pre={asym['mean_pre_bps']:+.2f}bps  post={asym['mean_post_bps']:+.2f}bps  "
          f"diff={asym['diff_bps']:+.2f}bps  HAC t={asym['tstat']:+.2f}")

    # ---- Leg 3: quarterly restriction ---------------------------------
    q = st.quarterly_split(bars)
    print("\n=== LEG 3 — QUARTERLY (TRIPLE-WITCHING) RESTRICTION ===")
    for k, lbl in (("all", "All months"), ("quarterly", "Quarterly only")):
        r = q[k]
        print(f"  {lbl:16s} diff={r['diff']*1e4:+.2f}bps  HAC t={r['tstat']:+.2f}  n={r['n_window']}")

    # ---- Leg 4: placebo / calendar randomisation ----------------------
    pl = st.placebo_randomization(bars)
    print("\n=== LEG 4 — PLACEBO / CALENDAR RANDOMISATION ===")
    print(f"  real-anchor t={pl['true_t']:+.2f}  placebo mean|t|={pl['placebo_mean_abs_t']:.2f}  "
          f"max|t|={pl['placebo_max_abs_t']:.2f}  n={pl['n_placebo']}  "
          f"empirical p={pl['p_value']:.3f}")

    # ---- structural break --------------------------------------------
    sp = st.pre_post_2012_split(bars)
    print("\n=== STRUCTURAL BREAK — PRE/POST-2012 (pre-OpEx drift) ===")
    for k in ("pre_2012", "post_2012"):
        r = sp[k]
        print(f"  {k:10s} diff={r['diff']*1e4:+.2f}bps  HAC t={r['tstat']:+.2f}  n={r['n_window']}")

    # ---- tradability --------------------------------------------------
    ov = st.charm_overlay_returns(bars)
    s = st.summarize(ov[ov != 0])
    ls = st.charm_overlay_returns(bars, short_post=True)
    s2 = st.summarize(ls[ls != 0])
    bh = st.summarize(st.daily_return(bars).dropna())
    print("\n=== TRADABILITY ===")
    print(f"  Long pre-OpEx : n={s['n']}  mean={s['mean_bps']:+.2f}bps  "
          f"sharpe={s['sharpe_ann']:+.2f}  CAGR={s['cagr']:+.2%}  HAC t={s['tstat']:+.2f}")
    print(f"  Long/short    : n={s2['n']}  mean={s2['mean_bps']:+.2f}bps  "
          f"sharpe={s2['sharpe_ann']:+.2f}  CAGR={s2['cagr']:+.2%}  HAC t={s2['tstat']:+.2f}")
    print(f"  Buy-and-hold  : mean={bh['mean_bps']:+.2f}bps  sharpe={bh['sharpe_ann']:+.2f}  "
          f"CAGR={bh['cagr']:+.2%}")
    print("  cost sweep (long-only, bps net/active day):",
          [(c, round(m, 2)) for c, m in st.cost_sweep(s["mean_bps"])])


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
