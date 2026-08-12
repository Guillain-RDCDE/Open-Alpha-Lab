"""Reproducible headline run for Study 864 — Yield-Curve Twist (Butterfly).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached real tape under
``_cache/`` (fetch once with ``--fetch``), and always runs the synthetic control with
no network.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # refresh the real tape first
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from curve_twist import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch_daily(fetch=True)

    df = data.load_daily()
    print("# Yield-Curve Twist (Butterfly) — does curvature predict forward returns?")
    print(f"[data] {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}  "
          f"as-of {data.AS_OF}  fingerprint(IEF_close)={data.fingerprint(df)}")
    print("  SIGNAL: fly = 2*y10 - y5 - y30 (butterfly / curvature), z-scored, lagged 1 day.")
    print("  SURVIVORSHIP: none — ETF/index total-return series, no cross-section.\n")

    print("# HEADLINE — forward log-return ~ lagged z(fly), HAC(NW) t")
    for tgt in ("IEF", "TLT", "SPY"):
        for h in (5, 21, 63):
            r = st.predictive_regression(df, "fly", tgt, h)
            print(f"  {tgt} h={h:>2}: beta {r['beta_bps']:+8.1f} bps/1sigma  "
                  f"t = {r['t']:+.2f}  R2 = {r['r2']*100:+.2f}%  n={r['n']}")

    print("\n# INCREMENTAL (the dedup test) — fly + slope + level, IEF h=21")
    r = st.predictive_regression(df, "fly", "IEF", 21, controls=["slope", "level"])
    print(f"  fly    : beta {r['beta_bps']:+.1f} bps  t = {r['t']:+.2f}")
    for c, tv in r["controls_t"].items():
        print(f"  {c:<6}: beta {r['controls_beta_bps'][c]:+.1f} bps  t = {tv:+.2f}")

    print("\n# QUINTILE SPREAD — Q5(high fly) - Q1(low fly) forward IEF return")
    for h in (5, 21, 63):
        q = st.quintile_spread(df, "fly", "IEF", h)
        print(f"  h={h:>2}: Q5 {q['q5_bps']:+.1f}  Q1 {q['q1_bps']:+.1f}  "
              f"spread {q['spread_bps']:+.1f} bps  t = {q['t_spread']:+.2f}")

    print("\n# THE TWIST (change) — forward IEF ~ lagged z(dfly)")
    for h in (5, 21, 63):
        r = st.predictive_regression(df, "dfly", "IEF", h)
        print(f"  dfly h={h:>2}: beta {r['beta_bps']:+.1f} bps  t = {r['t']:+.2f}")

    print("\n# ROBUSTNESS — three eras, fly -> IEF h=21")
    for lo, hi, lbl in [("2002-01-01", "2010-01-01", "2002-2009"),
                        ("2010-01-01", "2018-01-01", "2010-2017"),
                        ("2018-01-01", "2026-07-01", "2018-2026")]:
        sub = df[(df.index >= lo) & (df.index < hi)]
        r = st.predictive_regression(sub, "fly", "IEF", 21)
        print(f"  {lbl}: beta {r['beta_bps']:+7.1f}  t = {r['t']:+.2f}  "
              f"R2 = {r['r2']*100:+.2f}%  n={r['n']}")

    print("\n# PLACEBO — shuffle the signal vs forward IEF returns (500 permutations)")
    pl = st.placebo_pvalue(df, "fly", "IEF", 21, n_perm=500)
    print(f"  observed |t| = {abs(pl['obs_t']):.3f}; permuted t mean {pl['perm_mean_t']:+.3f} "
          f"(sd {pl['perm_sd_t']:.3f}) -> two-sided p = {pl['p_value']:.4f}")

    print("\n# THE TIMER — own IEF when lagged fly rank > 0.5, else cash; costed")
    for cb in (1.0, 2.0, 5.0):
        t = st.timing_overlay(df, "fly", "IEF", threshold=0.5, cost_bps=cb)
        print(f"  cost={cb:>4.1f} bps: active {t['active_bps']:+.3f} vs passive "
              f"{t['passive_bps']:+.3f} -> spread {t['spread_bps']:+.3f} bps/day "
              f"(t={t['t_spread']:+.2f}, Sharpe {t['sharpe_active']:.3f} vs "
              f"{t['sharpe_passive']:.3f}, {t['switches_per_yr']:.1f} switches/yr)")

    print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
    dfp, _ = data.synthetic_daily(fly_signal=0.02, n_days=4000, seed=864)
    sp = st.synthetic_detect(dfp, 21)
    print(f"  planted (fly_signal=0.02): reg t = {sp['t']:+.2f}, "
          f"Q5-Q1 spread t = {sp['t_spread']:+.2f}")
    ts, qs = [], []
    for s in range(20):
        dfn, _ = data.synthetic_daily(fly_signal=0.0, n_days=2500, seed=864 + s)
        d = st.synthetic_detect(dfn, 21)
        ts.append(d["t"]); qs.append(d["t_spread"])
    ts = np.asarray(ts); qs = np.asarray(qs)
    print(f"  null (fly_signal=0), 20 seeds: reg t mean {ts.mean():+.2f} (sd {ts.std(ddof=1):.2f}), "
          f"|t|>=2 in {(abs(ts) >= 2).sum()}/20")
    print(f"  null (fly_signal=0), 20 seeds: spread t mean {qs.mean():+.2f} (sd {qs.std(ddof=1):.2f}), "
          f"|t|>=2 in {(abs(qs) >= 2).sum()}/20")
    print("  NOTE: the null reg-t sd exceeds 1.0 — the HAC test over-rejects under the")
    print("  persistent-regressor + overlapping-return design, so the real full-sample t")
    print("  is discounted accordingly (see docs/results.md).")


if __name__ == "__main__":
    main("--fetch" in sys.argv)
