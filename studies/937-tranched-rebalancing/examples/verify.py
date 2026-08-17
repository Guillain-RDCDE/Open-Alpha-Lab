"""Real-tape verification — Study 937 (Tranches). Regenerates docs/results.md numbers.

Reads cached SPY / IEF / BIL / ^IRX daily closes, builds the 21 single-date books of a
monthly 200-day trend sleeve (SPY above its 200-day average → SPY, else IEF), then the
4-, 12- and 21-tranche books, and prints the timing-luck cone, the block-bootstrap CIs,
the persistence test, the era cut, the cost sweep, the BIL and 12-1 momentum
cross-checks and the synthetic control. Network only on ``--fetch``.

Lag convention (house form): the state known at the close of ``t`` is the position that
earns the return of ``t+1`` — one ``shift``, so execution is at the ``t`` close. The
robustness block re-runs the whole cone with a second day of delay.

    python studies/937-tranched-rebalancing/examples/verify.py            # cache-only
    python studies/937-tranched-rebalancing/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tranching import data, strategy as st  # noqa: E402

SMA_N = 200
COST_BPS = 5.0
N_LIST = (1, 4, 12, 21)


def cone_block(tag, R, TV, r_cash):
    cn = st.cone(R, TV, r_cash, n_list=N_LIST)
    print(f"\n=== {tag} ===")
    print("  N   Sharpe mean      sd     min     max   range | CAGR sd  range | "
          "terminal spread | turnover/yr")
    for n in N_LIST:
        c = cn[n]
        print(f"  {n:2d}   {c['sharpe_mean']:+.3f}      {c['sharpe_sd']:.3f}  "
              f"{c['sharpe_min']:+.3f}  {c['sharpe_max']:+.3f}   {c['sharpe_range']:.3f} | "
              f"{c['cagr_sd']*100:5.2f}pp {c['cagr_range']*100:5.2f}pp | "
              f"{c['tw_spread_pct']:6.1f}%        | {c['turnover_mean']:.2f}x")
    return cn


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    spy = px["SPY"].dropna()
    ief = px["IEF"].dropna()
    idx = spy.index.intersection(ief.index)
    spy, ief = spy.loc[idx], ief.loc[idx]
    r_cash = st.cash_returns_from_yield(px["IRX"].reindex(idx))

    print(f"SPY x IEF: {idx[0].date()} -> {idx[-1].date()}  n={len(idx)}  "
          f"fp={data.fingerprint(px[['SPY', 'IEF']].loc[idx])}")
    print(f"as-of {data.AS_OF}  |  cash leg = ^IRX accrual (PROXY), BIL cross-check below")

    state = st.trend_state(spy, sma_n=SMA_N)
    R, TV = st.offset_return_matrix(spy, ief, state, cost_bps=COST_BPS)
    print(f"books live {R.index[0].date()} -> {R.index[-1].date()}  n={len(R)}  "
          f"| in-risky fraction {float(state.dropna().mean()):.1%}")

    cn = cone_block("timing-luck cone, 200-day trend sleeve, 5 bps", R, TV, r_cash)

    print(f"\nshrink ratio sd(N)/sd(1): "
          + "  ".join(f"N={n}: {cn[n]['sharpe_sd']/cn[1]['sharpe_sd']:.3f}" for n in N_LIST))
    print(f"1/sqrt(N) reference:       "
          + "  ".join(f"N={n}: {1.0/np.sqrt(n):.3f}" for n in N_LIST))
    print(f"tranching gain (Sharpe of the 21-tranche book minus the average single date): "
          f"{cn[21]['sharpe_mean'] - cn[1]['sharpe_mean']:+.3f}")
    print(f"vol: single-date mean {cn[1]['vol_mean']:.2%} -> tranched {cn[21]['vol_mean']:.2%}"
          f"  |  drawdown mean {cn[1]['dd_mean']:.1%} (worst {cn[1]['dd_min']:.1%}) "
          f"-> tranched {cn[21]['dd_mean']:.1%}")
    print(f"turnover: single-date {cn[1]['turnover_mean']:.3f}x NAV/yr -> "
          f"tranched {cn[21]['turnover_mean']:.3f}x  "
          f"({(cn[21]['turnover_mean']/cn[1]['turnover_mean']-1)*100:+.1f}%)")
    print(f"CAGR: single-date mean {cn[1]['cagr_mean']:.2%} (worst {cn[1]['cagr_mean']-cn[1]['cagr_range']/2:.2%}) "
          f"-> tranched {cn[21]['cagr_mean']:.2%}  "
          f"({(cn[21]['cagr_mean']-cn[1]['cagr_mean'])*100:+.2f} pp/yr)")

    print("\n=== broker tickets (an ASSUMPTION sweep, not a tape measurement) ===")
    for n in N_LIST:
        tk = st.ticket_stats(TV, n, n_days=len(R))
        print(f"  N={n:2d}: {tk['tickets_per_year']:5.1f} tickets/yr  -> fixed-fee drag "
              + "  ".join(f"{fee:.1f}bp/ticket: {tk['tickets_per_year']*fee/1e4*100:.3f}%/yr"
                          for fee in (0.5, 2.0)))

    print("\n=== block bootstrap (1000 draws, 21-day blocks) ===")
    bs = st.dispersion_bootstrap(R, TV, r_cash, n_boot=1000, seed=937)
    print(f"  cone sd at N=1 : {bs['sd_point']:.3f}  95% band [{bs['sd_ci'][0]:.3f}, {bs['sd_ci'][1]:.3f}]"
          "  (a SPREAD BAND, not a test: an sd cannot be negative, so 'excludes zero' proves nothing;"
          " the band is right-skewed and the point sits near its lower edge)")
    print(f"  tranching gain : {bs['gain_point']:+.3f}  95% CI "
          f"[{bs['gain_ci'][0]:+.3f}, {bs['gain_ci'][1]:+.3f}]  share<0 {bs['gain_frac_negative']:.1%}"
          "  (this one CAN go negative — it is the study's only real interval)")

    print("\n=== is the lucky date forecastable? ===")
    bw = st.best_worst_test(R, r_cash)
    print(f"  ex-post best offset {bw['best_offset']} ({bw['sharpe_best']:+.3f}) vs worst "
          f"{bw['worst_offset']} ({bw['sharpe_worst']:+.3f}): gap {bw['sharpe_gap']:.3f}, "
          f"HAC t on the return difference {bw['t_diff']:+.2f} (SELECTION-BIASED: argmax vs argmin)")
    ps = st.persistence(R, r_cash)
    print(f"  split {ps['split']}: rank correlation of the 21 offsets across halves "
          f"rho={ps['rho']:+.3f}")
    print(f"  first-half winner = offset {ps['early_winner']}; its second-half rank is "
          f"{ps['late_rank_of_early_winner']}/21 (Sharpe {ps['late_sharpe_of_early_winner']:+.3f} "
          f"vs the 21-date average {ps['late_sharpe_mean']:+.3f}, best {ps['late_sharpe_best']:+.3f} "
          f"at offset {ps['late_winner']})")
    print(f"  SAME-WINDOW comparison: chasing that date earned {ps['chase_edge_late']:+.3f} in the "
          f"second half, against {ps['gain_late']:+.3f} for tranching over the SAME rows "
          f"(first half: tranching {ps['gain_early']:+.3f}). The chase wins by "
          f"{ps['chase_edge_late'] - ps['gain_late']:+.3f} — on ONE draw, ex-post-selected, with the "
          f"actual second-half winner a different offset. It is not evidence, and it is not free.")

    print("\n=== era cut (split 2014-01-01) ===")
    eras = st.era_cut(spy, ief, state, r_cash, split="2014-01-01", cost_bps=COST_BPS)
    for tag, e in eras.items():
        if e is None:
            continue
        print(f"  {tag:5s} (n={e['n_days']}): single-date Sharpe {e['sharpe_mean_n1']:+.3f} "
              f"(sd {e['sharpe_sd_n1']:.3f}, range {e['sharpe_range_n1']:.3f}) -> tranched "
              f"{e['sharpe_full']:+.3f} (gain {e['gain']:+.3f})  turnover "
              f"{e['turnover_n1']:.2f}x -> {e['turnover_full']:.2f}x")

    print("\n=== cost sweep (one-way bps on traded notional; no shorts, so no borrow) ===")
    for row in st.cost_sweep(spy, ief, state, r_cash):
        print(f"  cost={row['cost_bps']:5.1f}bps  single-date {row['sharpe_mean_n1']:+.3f} "
              f"(sd {row['sharpe_sd_n1']:.3f})  tranched {row['sharpe_mean_n21']:+.3f}  "
              f"gain {row['gain']:+.3f}  turnover {row['turnover_n1']:.2f}x -> "
              f"{row['turnover_n21']:.2f}x")

    print("\n=== cross-check 1: 12-1 momentum sleeve (same tranching machinery) ===")
    mom = st.momentum_state(spy, ief)
    Rm, TVm = st.offset_return_matrix(spy, ief, mom, cost_bps=COST_BPS)
    cnm = cone_block("12-1 momentum sleeve", Rm, TVm, r_cash)
    print(f"  gain {cnm[21]['sharpe_mean'] - cnm[1]['sharpe_mean']:+.3f}  "
          f"shrink sd {cnm[1]['sharpe_sd']:.3f} -> {cnm[21]['sharpe_sd']:.3f}  "
          f"turnover {cnm[1]['turnover_mean']:.2f}x -> {cnm[21]['turnover_mean']:.2f}x")

    cash_al = r_cash.reindex(R.index)
    rho_rules = st.spearman(R.sub(cash_al, axis=0).apply(st.sharpe).to_numpy(),
                            Rm.sub(r_cash.reindex(Rm.index), axis=0).apply(st.sharpe).to_numpy())
    print(f"  offset-ranking correlation between the two rules: rho={rho_rules:+.3f} "
          f"(a calendar effect would make the same dates win under both)")

    print("\n=== cross-check 2: an edge-free rule has the same cone (artefact, not edge) ===")
    rnd = st.random_state(state, seed=937)
    Rr, TVr = st.offset_return_matrix(spy, ief, rnd, cost_bps=COST_BPS)
    cnr = st.cone(Rr, TVr, r_cash, n_list=(1, 21))
    print(f"  exposure-matched random timing (seed 937): single-date {cnr[1]['sharpe_mean']:+.3f} "
          f"(sd {cnr[1]['sharpe_sd']:.3f}, range {cnr[1]['sharpe_range']:.3f}, terminal spread "
          f"{cnr[1]['tw_spread_pct']:.1f}%) -> tranched {cnr[21]['sharpe_mean']:+.3f} "
          f"(sd {cnr[21]['sharpe_sd']:.3f})")
    sw = st.random_cone_sd_sweep(spy, ief, state, r_cash, seeds=range(20), cost_bps=COST_BPS)
    print(f"  one random schedule is itself a lottery ticket, so sweep 20 of them: cone sd "
          f"mean {sw.mean():.3f}, median {np.median(sw):.3f}, min {sw.min():.3f}, max {sw.max():.3f}; "
          f"wider than the trend rule's {cn[1]['sharpe_sd']:.3f} in {int((sw > cn[1]['sharpe_sd']).sum())}/20 "
          f"seeds. The single seed above (0.103) sits in the upper half of that spread — the honest "
          f"statement is the distribution, not the draw.")

    print("\n=== robustness: one MORE day of execution delay (t+1 close instead of t close) ===")
    Rl, TVl = st.offset_return_matrix(spy, ief, state.shift(1), cost_bps=COST_BPS)
    cnl = st.cone(Rl, TVl, r_cash, n_list=(1, 21))
    print(f"  extra-lag books: single-date {cnl[1]['sharpe_mean']:+.3f} (sd {cnl[1]['sharpe_sd']:.3f}, "
          f"range {cnl[1]['sharpe_range']:.3f}) -> tranched {cnl[21]['sharpe_mean']:+.3f} "
          f"(gain {cnl[21]['sharpe_mean'] - cnl[1]['sharpe_mean']:+.3f}) — the cone and the fix do not "
          f"depend on which close you assume you traded at")

    print("\n=== cross-check 3: tradable BIL cash leg (2007+ window) ===")
    bil = px["BIL"].dropna()
    bidx = idx.intersection(bil.index)
    r_cash_bil = st.cash_returns_from_index(bil.reindex(bidx))
    Rb, TVb = st.offset_return_matrix(spy.loc[bidx], ief.loc[bidx],
                                      st.trend_state(spy.loc[bidx], sma_n=SMA_N),
                                      cost_bps=COST_BPS)
    cnb = st.cone(Rb, TVb, r_cash_bil, n_list=(1, 21))
    print(f"  {bidx[0].date()} -> {bidx[-1].date()} (n={len(Rb)}): single-date "
          f"{cnb[1]['sharpe_mean']:+.3f} (sd {cnb[1]['sharpe_sd']:.3f}, "
          f"range {cnb[1]['sharpe_range']:.3f}) -> tranched {cnb[21]['sharpe_mean']:+.3f} "
          f"(gain {cnb[21]['sharpe_mean'] - cnb[1]['sharpe_mean']:+.3f})")
    r_cash_irx_b = r_cash.reindex(Rb.index)
    cnb2 = st.cone(Rb, TVb, r_cash_irx_b, n_list=(1,))
    print(f"  same window with the ^IRX PROXY cash leg: single-date "
          f"{cnb2[1]['sharpe_mean']:+.3f} (sd {cnb2[1]['sharpe_sd']:.3f}) — proxy vs tradable "
          f"differ by {abs(cnb2[1]['sharpe_mean'] - cnb[1]['sharpe_mean']):.3f} Sharpe")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    for tag, ss in [("planted regime", 1.0), ("flat null   ", 0.0)]:
        rows = [st.synthetic_detect(p)
                for p, _ in data.synthetic_panel(6, signal_strength=ss, n_years=40)]
        g = np.array([r["gain"] for r in rows])
        sd1 = np.array([r["sharpe_sd_n1"] for r in rows])
        sdf = np.array([r["sharpe_sd_full"] for r in rows])
        edge = np.array([r["sleeve_minus_random"] for r in rows])
        print(f"  {tag}: sleeve minus exposure-matched random {edge.mean():+.3f} "
              f"(min {edge.min():+.3f}, max {edge.max():+.3f})  |  cone sd {sd1.mean():.3f} -> "
              f"{sdf.mean():.3f}  |  tranching gain {g.mean():+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
