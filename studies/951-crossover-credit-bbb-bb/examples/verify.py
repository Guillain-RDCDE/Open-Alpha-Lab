"""Real-tape verification — Study 951 (The Crossover Rung). Regenerates docs/results.md.

Reads the cached daily total-return closes of the credit ladder (AGG, LQD, ANGL, HYG), the
cash leg (BIL) and the two risk factors (IEF duration, SPY equity beta), then prints: the
ladder table (excess-of-cash stats and two-factor alphas), the head-to-head alphas on the
return difference, the joint block-bootstrap CIs, the era cut, the calendar-year table and
the one-year-out jackknife, the HAC bandwidth sweep, the FALN/USHY sibling-fund
cross-checks, the two tradable expressions with the borrow sweep, and the synthetic control.
Network only on ``--fetch``.

    python studies/951-crossover-credit-bbb-bb/examples/verify.py            # cache-only
    python studies/951-crossover-credit-bbb-bb/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crossover_credit import data, strategy as st  # noqa: E402

CROSSOVER = "ANGL"   # the crossover-rung proxy (fallen angels sit on the BBB/BB boundary)
IG = "LQD"           # the rung above
HY = "HYG"           # the rung below
SPLIT = "2019-01-01"
CRISIS_YEARS = (2016, 2020)


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()

    core = px[list(data.RUNGS) + list(data.FACTORS) + [data.CASH]].dropna()
    print(f"ladder window {core.index[0].date()} -> {core.index[-1].date()}  "
          f"n={len(core)}  fp={data.fingerprint(core)}")
    print(f"as-of {data.AS_OF}  (rungs {' < '.join(data.RUNGS)}; cash {data.CASH}; "
          f"factors {', '.join(data.FACTORS)} = duration, equity)")

    print("\n=== the ladder — excess-of-cash, then duration- and equity-adjusted ===")
    tbl = st.ladder_table(core, rungs=data.RUNGS, factors=data.FACTORS, cash=data.CASH)
    for rung, row in tbl.iterrows():
        print(f"  {rung:5s}: exSharpe {row['excess_sharpe']:+.3f}  "
              f"excess {row['excess_return_ann']*100:+5.2f}%/yr  vol {row['vol_ann']*100:5.2f}%  "
              f"maxDD {row['max_drawdown_abs']*100:+6.1f}%  |  alpha {row['alpha_ann']*100:+5.2f}%/yr "
              f"(t {row['t_alpha']:+.2f})  bDur {row['beta_dur']:+.2f}  bEq {row['beta_eq']:+.2f}  "
              f"R2 {row['r2']:.2f}")

    print("\n=== the head-to-head — two-factor alpha on the return difference ===")
    print("  (all three pairs on the SAME ANGL-gated window, so the rows are comparable)")
    pairs = {}
    for long, short in [(CROSSOVER, HY), (CROSSOVER, IG), (HY, IG)]:
        res = st.pair_alpha(core, long, short, factors=data.FACTORS, cash=data.CASH)
        pairs[(long, short)] = res
        print(f"  {long} - {short:5s}: alpha {res['alpha_ann']*100:+5.2f}%/yr (t {res['t_alpha']:+.2f})  "
              f"| raw diff {res['raw_diff_ann']*100:+5.2f}%/yr (t {res['t_raw_diff']:+.2f})  "
              f"n={res['n']}  [bDur {res['betas'][data.FACTORS[0]]:+.2f}, "
              f"bEq {res['betas'][data.FACTORS[1]]:+.3f}]")
    # The HY-IG step is also available on a much longer (BIL-gated) window; reported
    # separately and labelled, never mixed into the table above.
    long_hl = st.pair_alpha(px, HY, IG, factors=data.FACTORS, cash=data.CASH)
    print(f"  aside — {HY} - {IG} on its own longest window "
          f"({long_hl['start'].date()} -> {long_hl['end'].date()}, n={long_hl['n']}): "
          f"alpha {long_hl['alpha_ann']*100:+.2f}%/yr (t {long_hl['t_alpha']:+.2f})")

    print("\n=== joint block-bootstrap CIs on the pair alpha (2,000 draws, 21-day blocks) ===")
    for key in [(CROSSOVER, HY), (CROSSOVER, IG)]:
        res = pairs[key]
        ci = st.bootstrap_alpha_ci(res["diff"], res["factors"], seed=951)
        print(f"  {key[0]} - {key[1]:5s}: alpha {ci['alpha_ann']*100:+5.2f}%/yr  95% CI "
              f"[{ci['ci_low']*100:+.2f}%, {ci['ci_high']*100:+.2f}%]  share<0 {ci['frac_negative']:.3f}")

    print(f"\n=== robustness 1 — the era cut (split {SPLIT}) ===")
    for long, short in [(CROSSOVER, HY), (CROSSOVER, IG)]:
        eras = st.era_cut(px, long, short, split=SPLIT, factors=data.FACTORS, cash=data.CASH)
        for tag, r in eras.items():
            if r is None:
                continue
            print(f"  {long} - {short:5s} {tag:5s} (n={r['n']}): alpha {r['alpha_ann']*100:+5.2f}%/yr "
                  f"(t {r['t_alpha']:+.2f})  raw {r['raw_diff_ann']*100:+5.2f}%/yr")

    print(f"\n=== robustness 2 — calendar years and the one-year-out jackknife ({CROSSOVER} - {HY}) ===")
    cal = st.calendar_year_table(px, CROSSOVER, HY, cash=data.CASH)
    print("  " + "  ".join(f"{y}:{row['diff_pct']:+.2f}" for y, row in cal.iterrows()))
    jack = st.year_jackknife(px, CROSSOVER, HY, factors=data.FACTORS, cash=data.CASH)
    survive = int((jack["t_alpha"].abs() >= 2.0).sum())
    print(f"  |t| >= 2 survives {survive}/{len(jack)} single-year deletions")
    for y, row in jack.iterrows():
        print(f"    drop {y}: alpha {row['alpha_ann']*100:+5.2f}%/yr (t {row['t_alpha']:+.2f})  n={int(row['n'])}")
    dropped = st.drop_years(px, CROSSOVER, HY, CRISIS_YEARS, factors=data.FACTORS, cash=data.CASH)
    print(f"  drop {CRISIS_YEARS} together: alpha {dropped['alpha_ann']*100:+.2f}%/yr "
          f"(t {dropped['t_alpha']:+.2f})  n={dropped['n']}")

    print("\n=== robustness 3 — HAC bandwidth sweep and the sibling-fund cross-checks ===")
    head = pairs[(CROSSOVER, HY)]
    lagtbl = st.lag_sensitivity(head["diff"], head["factors"], lag_grid=(5, 8, 10, 21, 42))
    for lg, row in lagtbl.iterrows():
        print(f"  HAC lags {lg:3d}: alpha {row['alpha_ann']*100:+.2f}%/yr (t {row['t_alpha']:+.2f})")
    for long, short in [("FALN", "USHY"), ("FALN", HY), (CROSSOVER, "USHY")]:
        res = st.pair_alpha(px, long, short, factors=data.FACTORS, cash=data.CASH)
        print(f"  {long} - {short:5s} ({res['start'].date()} -> {res['end'].date()}, n={res['n']}): "
              f"alpha {res['alpha_ann']*100:+.2f}%/yr (t {res['t_alpha']:+.2f})")

    # Is the sibling-fund collapse about fund IDENTITY or about the shorter LATER window?
    # Run all four crossover/broad-HY pairs on the one window all four funds share.
    sib = px[["FALN", "USHY", CROSSOVER, HY] + list(data.FACTORS) + [data.CASH]].dropna()
    print(f"  same-window decomposition ({sib.index[0].date()} -> {sib.index[-1].date()}, "
          f"n={len(sib)}) — window effect vs fund effect:")
    for long, short in [(CROSSOVER, HY), ("FALN", HY), ("FALN", "USHY"), (CROSSOVER, "USHY")]:
        res = st.pair_alpha(sib, long, short, factors=data.FACTORS, cash=data.CASH)
        tag = "  <- the HEADLINE pair, same window" if (long, short) == (CROSSOVER, HY) else ""
        print(f"    {long} - {short:5s}: alpha {res['alpha_ann']*100:+.2f}%/yr "
              f"(t {res['t_alpha']:+.2f}){tag}")

    late_rungs = (IG, "FALN", CROSSOVER, "USHY", HY)
    late = px[list(late_rungs) + list(data.FACTORS) + [data.CASH]].dropna()
    print(f"  the five-fund ladder on the FALN/USHY-gated window "
          f"({late.index[0].date()} -> {late.index[-1].date()}, n={len(late)}):")
    late_tbl = st.ladder_table(late, rungs=late_rungs, factors=data.FACTORS, cash=data.CASH)
    for rung, row in late_tbl.iterrows():
        print(f"    {rung:5s}: exSharpe {row['excess_sharpe']:+.3f}  "
              f"alpha {row['alpha_ann']*100:+.2f}%/yr (t {row['t_alpha']:+.2f})")

    print("\n=== expression A — own the crossover fund instead of the broad-HY fund ===")
    swap = st.long_only_swap(px, CROSSOVER, HY, cost_bps=5.0, cash=data.CASH)
    print(f"  {HY:5s}: exSharpe {swap['sharpe_short']:+.3f}  excess {swap['excess_ann_short']*100:+.2f}%/yr  "
          f"vol {swap['vol_short']*100:.2f}%  maxDD {swap['dd_short_abs']*100:+.1f}%")
    print(f"  {CROSSOVER:5s}: exSharpe {swap['sharpe_long']:+.3f}  excess {swap['excess_ann_long']*100:+.2f}%/yr  "
          f"vol {swap['vol_long']*100:.2f}%  maxDD {swap['dd_long_abs']*100:+.1f}%")
    print(f"  Sharpe gain {swap['sharpe_gain']:+.3f} for {(swap['dd_long_abs']-swap['dd_short_abs'])*100:+.1f} pp "
          f"of extra worst drawdown (switch charged both sides, 2 x 5 bps, amortised over "
          f"{swap['n_days']} days)")

    print("\n=== expression B — the pure spread (long crossover, short broad HY), borrow swept ===")
    sweep = st.borrow_sweep(px, CROSSOVER, HY, borrow_grid=(0.0, 25.0, 50.0, 100.0),
                            cost_bps=5.0, factors=data.FACTORS, cash=data.CASH)
    for b, row in sweep.iterrows():
        print(f"  borrow {b:6.1f} bps/yr: alpha {row['alpha_ann']*100:+.2f}%/yr (t {row['t_alpha']:+.2f})  "
              f"mean {row['mean_ann']*100:+.2f}%/yr  Sharpe {row['sharpe']:+.3f}  "
              f"turnover {row['turnover_per_year']:.2f}x/yr")

    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    planted, truth = data.synthetic_panel(signal_strength=1.0, seed=951)
    d_pl = st.synthetic_detect(planted)
    print(f"  planted {truth['alpha_planted']*100:+.2f}%/yr -> recovered {d_pl['alpha_ann']*100:+.2f}%/yr "
          f"(t {d_pl['t_alpha']:+.2f}); loadings "
          + ", ".join(f"{k} {v:+.2f}" for k, v in d_pl["betas"].items()))
    nulls = [st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=951 + s)[0])
             for s in range(8)]
    a = np.array([d["alpha_ann"] for d in nulls])
    t = np.array([d["t_alpha"] for d in nulls])
    print(f"  null x8: alpha mean {a.mean()*100:+.2f}%/yr (sd {a.std(ddof=1)*100:.2f}), "
          f"|t| >= 2 fires {(np.abs(t) >= 2).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
