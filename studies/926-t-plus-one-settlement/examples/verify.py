"""Real-tape verification — Study 926 (T+1). Regenerates the docs/results.md numbers.

Reads cached daily total-return OHLC for SPY (control), IWM (domestic placebo), EFA and
EEM (the settlement-mismatched legs) plus BIL (cash), and runs the whole difference-in-
difference battery around the 28 May 2024 switch from T+2 to T+1: the four daily
outcomes, the block-bootstrap CIs, the window-length sweep, the placebo-switch-date
falsification, the excess-of-cash Sharpe race, the turn-of-month long/short overlay and
its cost x borrow sweep. Network only on ``--fetch``.

    python studies/926-t-plus-one-settlement/examples/verify.py            # cache-only
    python studies/926-t-plus-one-settlement/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from t_plus_one import data, strategy as st  # noqa: E402

CONTROL = data.CONTROL
LEGS = ("EFA", "EEM", "IWM")
OUTCOMES = ("r_on", "abs_on", "on_var_share", "abs_cc")
COST_BPS = 3.0
BORROW_BPS = 50.0


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    panel = data.load_ohlc()
    px = data.load_prices()
    sw, asof = data.SWITCH_DATE, data.AS_OF

    tab_ref = st.did_table(panel, "EFA", CONTROL, sw, asof)
    win = tab_ref["_windows"]
    print(f"switch {sw}   as-of {asof}")
    print(f"symmetric windows: pre {win['pre_start'].date()} -> {win['pre_end'].date()}  |  "
          f"post {win['post_start'].date()} -> {win['post_end'].date()}   n={win['n_each']} each")
    closes = px.dropna()
    print(f"fingerprint (close frame): {data.fingerprint(closes)}   n rows {len(closes)}")
    # A second, churn-proof stamp: the shared cache is refreshed by other studies, and a
    # re-download can shift the *level* of an adjusted close by a rounding tick or a new
    # dividend's back-adjustment factor. Every number in this study is return-based and
    # therefore invariant to that; this fingerprint of the rounded daily returns is too.
    rets = closes.pct_change(fill_method=None).dropna().round(10)
    print(f"fingerprint (daily returns, rounded 1e-10): {data.fingerprint(rets)}   "
          f"n rows {len(rets)}")

    print("\n=== difference-in-difference, treated - SPY, post minus pre ===")
    print("outcome          leg    DiD        t      treated pre->post     control pre->post")
    for leg in LEGS:
        tab = st.did_table(panel, leg, CONTROL, sw, asof)
        for col in OUTCOMES:
            e = tab[col]
            print(f"  {col:<13s} {leg:<5s} {e['did']:+9.3f} {e['did_t']:+7.2f}   "
                  f"{e['treated_pre']:8.3f}->{e['treated_post']:8.3f}   "
                  f"{e['control_pre']:8.3f}->{e['control_post']:8.3f}")

    print("\n=== block-bootstrap CIs on the DiD (2,000 draws, 21-day blocks) ===")
    for leg in ("EFA", "EEM"):
        tab = st.did_table(panel, leg, CONTROL, sw, asof)
        for col in ("on_var_share", "abs_cc"):
            ci = st.did_bootstrap_ci(tab[col], seed=926)
            print(f"  {leg} {col:<13s} point {ci['point']:+8.3f}  95% CI "
                  f"[{ci['ci_low']:+8.3f}, {ci['ci_high']:+8.3f}]  frac<0 {ci['frac_negative']:.3f}")

    print("\n=== window-length sweep (half-window in trading days) ===")
    for leg in ("EFA", "EEM"):
        for col in ("on_var_share", "abs_cc"):
            rows = st.window_length_sweep(panel, leg, CONTROL, sw, asof, outcome=col)
            cells = "  ".join(f"{r['half_window']}d {r['did']:+.3f}(t{r['did_t']:+.2f})" for r in rows)
            print(f"  {leg} {col:<13s} {cells}")

    print("\n=== placebo switch dates (fake events every 63 trading days) ===")
    for leg in LEGS:
        for col in OUTCOMES:
            p = st.placebo_switches(panel, leg, CONTROL, sw, asof, outcome=col)
            print(f"  {leg} {col:<13s} real |t| {abs(p['real_t']):5.2f}  "
                  f"placebo median |t| {p['placebo_median_abs_t']:5.2f}  max {p['placebo_max_abs_t']:5.2f}  "
                  f"exceeding {p['n_placebo_exceeding']}/{p['n_placebo']}")

    print("\n=== excess-of-cash Sharpe (vs BIL), pre vs post ===")
    for tk in ("SPY", "IWM", "EEM", "EFA"):
        e = st.excess_sharpe_pre_post(px[tk], px[data.CASH], sw, asof)
        print(f"  {tk:<4s} exSharpe {e['pre']['sharpe']:+.2f} -> {e['post']['sharpe']:+.2f}  "
              f"(Welch t {e['welch_t']:+.2f})   vol {e['pre']['vol_ann']:.1%} -> {e['post']['vol_ann']:.1%}")

    print(f"\n=== turn-of-month long/short overlay ({COST_BPS} bps one-way per leg, "
          f"{BORROW_BPS} bps/yr borrow) ===")
    for leg in ("EFA", "EEM"):
        r = st.tom_pre_post(px[leg], px[CONTROL], sw, asof,
                            cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS)
        print(f"  long {leg} / short {CONTROL}: on-days {r['n_on_days_pre']} pre / "
              f"{r['n_on_days_post']} post")
        print(f"    gross {r['gross_pre_bps']:+.2f} -> {r['gross_post_bps']:+.2f} bps/on-day   "
              f"net {r['net_pre_bps']:+.2f} -> {r['net_post_bps']:+.2f} bps/on-day")
        print(f"    DiD net {r['did_net_bps']:+.2f} bps (HAC t {r['did_net_t']:+.2f})   "
              f"post-only net HAC t {r['t_post_net']:+.2f}")
        s = r["summary_post_net"]
        print(f"    post-period net: Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.2%}  "
              f"vol {s['vol_ann']:.1%}  maxDD {s['max_drawdown']:+.1%}")

    print("\n=== cost x borrow sweep (post-period net, long EEM / short SPY) ===")
    for row in st.borrow_sweep(px["EEM"], px[CONTROL], sw, asof):
        print(f"  cost {row['cost_bps']:5.1f} bps, borrow {row['borrow_bps_ann']:6.1f} bps/yr  ->  "
              f"net {row['net_post_bps']:+6.2f} bps/on-day (t {row['t_post_net']:+.2f})   "
              f"DiD {row['did_net_bps']:+6.2f} (t {row['did_net_t']:+.2f})")

    print("\n=== synthetic control (machinery proof only) ===")
    for tag, ss in [("planted", 1.0), ("null   ", 0.0)]:
        p, t = data.synthetic_panel(signal_strength=ss, seed=926)
        d = st.synthetic_detect(p, t)
        print(f"  {tag}: DiD r_on {d['did_r_on_bps']:+6.2f} bps (t {d['t_r_on']:+.2f})  "
              f"abs_on {d['did_abs_on_bps']:+6.2f} (t {d['t_abs_on']:+.2f})  "
              f"share {d['did_on_var_share']:+.3f} (t {d['t_on_var_share']:+.2f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
