"""Real-tape verification — Study 923 (The Cash Lag). Regenerates docs/results.md.

Reads cached daily total-return closes for the cash vehicles (BIL, SGOV, USFR, SHV)
and the 13-week bill quote ^IRX, then runs both halves of the study:

  A. the measurement — realised effective duration, the distributed-lag
     pass-through profile, and the proxy-window sweep that separates the genuine
     repricing lag from the trailing-average artefact;
  B. the trade — the rate-direction switch rule, its two placebos, the
     pre-registered lookback grid, the cost sweep, an era cut and a bootstrap CI.

Network only on ``--fetch``. With no cache present (a fresh checkout) the script falls
back to the **offline synthetic demo** — the same three-world control the notebooks run —
and exits 0, so it is always runnable; it never prints a synthetic figure under a
real-tape banner.

    python studies/923-mmf-yield-lag/examples/verify.py            # cache-only
    python studies/923-mmf-yield-lag/examples/verify.py --fetch    # refresh the tapes
    python studies/923-mmf-yield-lag/examples/verify.py --synthetic  # force the offline demo
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_lag import data, strategy as st  # noqa: E402

COST_BPS = 2.0
LOOKBACK = 21


def synthetic_demo() -> None:
    """The offline fallback — SYNTHETIC ONLY, never a substitute for the real tape."""
    print("=== SYNTHETIC DEMO - these are NOT real-tape numbers ===")
    print("Three worlds, one detector. The panel plants a WAM ladder and a trending rate")
    print("path independently, so each can be switched off.\n")
    for label, kw in [("real ladder + trending rate", dict(signal_strength=1.0)),
                      ("real ladder + random walk  ", dict(signal_strength=1.0, trend_phi=0.0)),
                      ("no ladder (the null)       ", dict(signal_strength=0.0))]:
        d = st.synthetic_detect(*data.synthetic_panel(seed=923, **kw))
        print(f"  {label}: duration spread {d['duration_spread']:+.3f} yr | "
              f"switch gross {d['gross_bp']:+7.1f} bp/yr (t={d['gross_t']:+6.2f})")
    print("\nRun with --fetch (network) to populate the shared cache and reproduce")
    print("docs/results.md on the real tape.")


def main(fetch: bool, synthetic: bool = False) -> None:
    if fetch:
        data.fetch()
    if synthetic or not data.have_real():
        synthetic_demo()
        return
    px = data.load_prices()
    hf = data.headline_frame(px)

    print(f"headline frame: {list(data.HEADLINE_VEHICLES)} + {data.RATE}   "
          f"{hf.index[0].date()} -> {hf.index[-1].date()}  n={len(hf)}  "
          f"fp={data.fingerprint(hf)}")
    print(f"as-of {data.AS_OF}  |  cash benchmark = {data.BENCH} (total return)")

    # ------------------------------------------------------------------ A ---
    print("\n=== A1. realised effective duration (daily r on daily d^IRX, window-free) ===")
    for v in data.VEHICLES:
        sub = px[[v, data.RATE]].dropna()
        sub = sub[sub.index >= pd.Timestamp(data.HEADLINE_START)] if v != "SGOV" else sub
        d = st.effective_duration(sub[v], sub[data.RATE])
        print(f"  {v:5s} n={d['n']:5d}  duration = {d['duration_years']:+.3f} yr  "
              f"(HAC t = {d['tstat']:+.2f})   [nominal WAM {data.NOMINAL_WAM_DAYS[v]}d]")

    print("\n=== A2. distributed-lag pass-through of ^IRX into the 21d realised-yield proxy ===")
    for v in data.VEHICLES:
        sub = px[[v, data.RATE]].dropna()
        sub = sub[sub.index >= pd.Timestamp(data.HEADLINE_START)] if v != "SGOV" else sub
        p = st.lag_profile(sub[v], sub[data.RATE])
        print(f"  {v:5s} n={p['n']:5d}  sum(beta)={p['sum_beta']:+.2f}  "
              f"contemporaneous={p['contemporaneous']:+.2f} (t={p['contemporaneous_t']:+.2f})  "
              f"peak lag={p['peak_lag']}d  centroid={p['centroid_lag']:.1f}d")
        print("        " + "  ".join(f"{k}:{c:+.2f}(t{t:+.1f})"
                                     for k, c, t in zip(p["lags"], p["coefs"], p["tstats"])))

    print("\n=== A3. centroid lag vs proxy window (the averaging-artefact check) ===")
    for w in (5, 10, 21, 42):
        cells = []
        for v in data.VEHICLES:
            sub = px[[v, data.RATE]].dropna()
            sub = sub[sub.index >= pd.Timestamp(data.HEADLINE_START)] if v != "SGOV" else sub
            cells.append(f"{v} {st.lag_profile(sub[v], sub[data.RATE], window=w)['centroid_lag']:5.1f}")
        print(f"  window={w:2d}d :  " + "   ".join(cells))

    print("\n=== A4. is the MEASUREMENT era-robust? (duration by era, split 2020-01-01) ===")
    print("  The Signal stamp rests on this half, so it gets the same era cut as the trade.")
    for tag, sl in [("2014-2019", slice(None, "2020-01-01")), ("2020-2026", slice("2020-01-01", None))]:
        sub = hf.loc[sl]
        cells = []
        for v in data.HEADLINE_VEHICLES:
            d = st.effective_duration(sub[v], sub[data.RATE])
            cells.append(f"{v} {d['duration_years']:+.3f} (t={d['tstat']:+5.2f})")
        print(f"  {tag} n={len(sub):5d}:  " + "   ".join(cells))

    # ------------------------------------------------------------------ B ---
    print(f"\n=== B1. the switch rule (rates up -> USFR, rates down -> SHV), "
          f"lookback {LOOKBACK}d, {COST_BPS:.0f} bps one-way ===")
    ev = st.evaluate(hf, lookback=LOOKBACK, cost_bps=COST_BPS)
    ev0 = st.evaluate(hf, lookback=LOOKBACK, cost_bps=0.0)
    for tag, s in [("switch (net)", ev["switch"]), ("switch (gross)", ev0["switch"]),
                   ("reversed (net)", ev["reversed"]), ("placebo (net)", ev["placebo"]),
                   ("static USFR", ev["static_fast"]), ("static SHV", ev["static_slow"])]:
        print(f"  {tag:16s} n={s['n_days']:5d}  excess vs BIL {s['ann_bp']:+8.1f} bp/yr  "
              f"HAC t={s['tstat']:+6.2f}  exSharpe={s['sharpe']:+6.2f}  "
              f"vol={s['vol_ann_bp']:6.0f} bp")
    print(f"  switches={ev['n_switches']}  ({ev['n_switches'] / (len(hf) / 252):.1f}/yr)  "
          f"days in USFR={ev['frac_fast']:.1%}")

    print("\n  -- allocation vs timing (does the TIMING add anything, gross?) --")
    bt0 = ev0["bt"]
    f = ev0["frac_fast"]
    blend_ex = (f * bt0["r_fast"] + (1.0 - f) * bt0["r_slow"] - bt0["r_bench"]).dropna()
    timing = (bt0["r_switch"] - (f * bt0["r_fast"] + (1.0 - f) * bt0["r_slow"])).dropna()
    ci_t = st.block_bootstrap_mean_ci(timing)
    print(f"  passive {f:.0%}/{1 - f:.0%} USFR/SHV blend, no trading: "
          f"{blend_ex.mean() * 252 * 1e4:+7.1f} bp/yr (HAC t={st.newey_west_t(blend_ex.to_numpy()):+5.2f})")
    print(f"  the timing decision alone (rule - that blend):          "
          f"{timing.mean() * 252 * 1e4:+7.1f} bp/yr (HAC t={st.newey_west_t(timing.to_numpy()):+5.2f})"
          f"  95% CI [{ci_t['ci_low']:+.1f}, {ci_t['ci_high']:+.1f}]")
    print("  i.e. the gross rule = a mediocre static allocation MINUS a zero timing bet;")
    print("  none of the (small, positive) blend number is earned by knowing anything.")

    print("\n  -- the turnover-matched placebo across 5 seeds (net, one draw is not a control) --")
    pl_bp = [st.evaluate(hf, lookback=LOOKBACK, cost_bps=COST_BPS,
                         placebo_seed=s)["placebo"]["ann_bp"] for s in range(1, 6)]
    print("  " + "  ".join(f"{b:+.1f}" for b in pl_bp) +
          f"   -> range [{min(pl_bp):+.1f}, {max(pl_bp):+.1f}] bp/yr, "
          f"bracketing the rule's {ev['switch']['ann_bp']:+.1f}")

    print("\n=== B2. block-bootstrap CI on the switch rule's annualised excess ===")
    for tag, series in [("gross", ev0["excess"]), (f"net {COST_BPS:.0f}bp", ev["excess"])]:
        ci = st.block_bootstrap_mean_ci(series)
        print(f"  {tag:9s}: {ci['point']:+8.1f} bp/yr  95% CI [{ci['ci_low']:+.1f}, "
              f"{ci['ci_high']:+.1f}]  share<0 {ci['frac_negative']:.1%}")

    print("\n=== B3. pre-registered lookback grid (all of it, not the best of it) ===")
    for row in st.lookback_sweep(hf, cost_bps=COST_BPS):
        print(f"  L={row['lookback']:4d}d  gross {row['gross_bp']:+8.1f} bp/yr (t={row['gross_t']:+5.2f})  "
              f"net {row['net_bp']:+8.1f} (t={row['net_t']:+5.2f})  "
              f"reversed-net {row['reversed_net_bp']:+8.1f}  switches={row['n_switches']}")

    print("\n=== B4. cost sweep (the one-way cost is a PROXY/ASSUMPTION) ===")
    for row in st.cost_sweep(hf, lookback=LOOKBACK):
        print(f"  cost={row['cost_bps']:5.1f} bps  excess {row['ann_bp']:+8.1f} bp/yr  "
              f"HAC t={row['tstat']:+6.2f}  switches={row['n_switches']}")

    print("\n=== B5. era cut (split 2020-01-01) ===")
    for tag, e in st.era_cut(hf, split="2020-01-01", cost_bps=COST_BPS,
                             lookback=LOOKBACK).items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_days']:5d}: switch gross {e['gross_bp']:+7.1f} (t={e['gross_t']:+5.2f})  "
              f"net {e['net_bp']:+7.1f} (t={e['net_t']:+5.2f})  |  "
              f"static USFR {e['static_fast_bp']:+6.1f} (t={e['static_fast_t']:+5.2f})  "
              f"static SHV {e['static_slow_bp']:+6.1f} (t={e['static_slow_t']:+5.2f})")

    # ------------------------------------------------------- cross-check ---
    print("\n=== B6. SGOV four-vehicle cross-check (2020-06 onward) ===")
    sg = px[["SGOV", "BIL", "USFR", "SHV", data.RATE]].dropna()
    print(f"  window {sg.index[0].date()} -> {sg.index[-1].date()}  n={len(sg)}")
    ev_sg = st.evaluate(sg, lookback=LOOKBACK, cost_bps=COST_BPS)
    ev_sg0 = st.evaluate(sg, lookback=LOOKBACK, cost_bps=0.0)
    print(f"  switch gross {ev_sg0['switch']['ann_bp']:+.1f} bp/yr (t={ev_sg0['switch']['tstat']:+.2f})  "
          f"net {ev_sg['switch']['ann_bp']:+.1f} (t={ev_sg['switch']['tstat']:+.2f})")
    for v in ("SGOV", "USFR", "SHV"):
        e = sg[v].pct_change() - sg["BIL"].pct_change()
        print(f"  static {v:5s} vs BIL: {e.mean() * 252 * 1e4:+7.1f} bp/yr  "
              f"(HAC t={st.newey_west_t(e.dropna().to_numpy()):+5.2f})")

    print("\n=== B7. the SGOV-over-BIL side finding, stress-tested ===")
    print("  SELECTION: SGOV is the best of the three static arms printed in B6 - it was")
    print("  picked AFTER seeing them. Its defence is not its t-stat but that the fee gap")
    print("  driving it is published ex ante and that it survives a split-half cut.")
    e_sg = (sg["SGOV"].pct_change() - sg["BIL"].pct_change()).dropna()
    ci = st.block_bootstrap_mean_ci(e_sg)
    print(f"  full  n={ci['n_obs']:5d}  {ci['point']:+6.1f} bp/yr  "
          f"(HAC t={st.newey_west_t(e_sg.to_numpy()):+5.2f})  "
          f"95% CI [{ci['ci_low']:+.1f}, {ci['ci_high']:+.1f}]")
    half = len(e_sg) // 2
    for tag, sub in [("early", e_sg.iloc[:half]), ("late ", e_sg.iloc[half:])]:
        c = st.block_bootstrap_mean_ci(sub)
        print(f"  {tag} {sub.index[0].date()} -> {sub.index[-1].date()}  n={len(sub):5d}  "
              f"{c['point']:+6.1f} bp/yr  (HAC t={st.newey_west_t(sub.to_numpy()):+5.2f})  "
              f"95% CI [{c['ci_low']:+.1f}, {c['ci_high']:+.1f}]")
    rt = 2.0 * COST_BPS  # one round trip: sell BIL, buy SGOV
    print(f"  one-off swap cost {rt:.0f} bps at {COST_BPS:.0f} bps/leg vs {ci['point']:+.1f} bp/yr "
          f"-> paid back in {12.0 * rt / ci['point']:.1f} months, then free forever.")

    # --------------------------------------------------------- synthetic ---
    print("\n=== synthetic control (machinery proof only - never supports the stamp) ===")
    pl = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=923))
    print(f"  planted: duration spread {pl['duration_spread']:+.3f} yr, "
          f"durations {{{', '.join(f'{k} {v:.3f}' for k, v in pl['durations'].items())}}}, "
          f"switch gross {pl['gross_bp']:+.1f} bp/yr (t={pl['gross_t']:+.2f})")
    rw = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, trend_phi=0.0, seed=923))
    print(f"  power floor (real ladder, UNPREDICTABLE rate): duration spread "
          f"{rw['duration_spread']:+.3f} yr, switch gross {rw['gross_bp']:+.1f} bp/yr "
          f"(t={rw['gross_t']:+.2f}) - carry rotation with no forecasting at all")
    nulls = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=923 + s))
             for s in range(8)]
    gt = np.array([n["gross_t"] for n in nulls])
    dsp = np.array([n["duration_spread"] for n in nulls])
    print(f"  null x8: duration spread mean {dsp.mean():+.4f} yr (sd {dsp.std(ddof=1):.4f}); "
          f"switch gross t mean {gt.mean():+.2f} (sd {gt.std(ddof=1):.2f}), "
          f"|t|>=2 on {int((np.abs(gt) >= 2).sum())}/8 seeds")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv, synthetic="--synthetic" in sys.argv)
