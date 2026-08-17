"""Real-tape verification — Study 932 (Trust Yield). Regenerates docs/results.md numbers.

Reads the cached unadjusted pre-deal SPAC closes and the BIL cash leg, builds every
buy-below-trust position the rule would have taken (month-end signal, next-close
execution, hold to the buffered redemption date, redeem at trust), and prints the
position-level excess over cash, the cluster bootstrap over SPACs, the daily accrual and
mark-to-market books, the era cut, every assumption sweep, the yield path and the
synthetic control. Network only on ``--fetch``.

    python studies/932-spac-trust-yield/examples/verify.py            # cache-only
    python studies/932-spac-trust-yield/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trust_yield import data, strategy as st  # noqa: E402

COST_BPS = 15.0


def synthetic_only() -> None:
    """Offline fallback when the shared cache is absent — clearly labelled as synthetic.

    It never prints a real-tape number and never pretends to; it just proves the harness
    runs end to end on a fresh checkout, recovering a planted trust put and staying quiet
    when the put is a fiction.
    """
    import numpy as np

    print("shared _cache absent - REAL-TAPE RESULTS ARE NOT SHOWN.")
    print("Run `python examples/verify.py --fetch` once to populate studies/_cache.\n")
    print("=== offline synthetic demo (SYNTHETIC, not the tape) ===")
    for ss, world in ((1.0, "trust put binds"), (0.0, "trust put is a fiction (null)")):
        outs = [st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=s,
                                                          n_days=700)[:3], n_boot=600)
                for s in range(932, 940)]
        m = np.array([o["mean_excess"] for o in outs], dtype=float)
        fires = sum(1 for o in outs if o["ci_low"] > 0)
        print(f"  ss={ss:.0f} [{world}]: mean excess {m.mean():+.3%} "
              f"(sd {m.std(ddof=1):.3%})  cluster-bootstrap CI>0 on {fires}/8 panels")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    if not data.have_real():
        synthetic_only()
        return
    raw = data.load_spac_closes()
    px, bad = data.clean_quotes(raw)
    cashf = data.load_cash()
    cash = cashf["BIL"].dropna()

    win = st.spac_windows(px)
    print(f"as-of {data.AS_OF}   SPACs listed {len(data.SPACS)}   with a usable window {len(win)}")
    print(f"pre-deal tape {win['first'].min().date()} -> {win['deadline'].max().date()}  "
          f"fp={data.fingerprint(px)}  cash fp={data.fingerprint(cashf[['BIL']])}")
    print(f"bad pre-deal prints replaced: {bad if bad else 'none'}")
    print(f"assumptions: trust0=${data.TRUST0:.2f}  buffer={data.BUFFER_DAYS}d  "
          f"fee={data.TRUST_FEE_BPS:.0f}bp  max_discount={data.MAX_DISCOUNT:.0%}  cost={COST_BPS:.0f}bps one-way")

    r = st.race(px, cash, cost_bps=COST_BPS)
    pos = r["positions"]
    print("\n=== headline: buy below trust, redeem at the deadline ===")
    print(f"  positions {r['n_pos']} across {r['n_spacs']} SPACs   mean hold {r['mean_hold_days']:.0f} days")
    print(f"  mean discount at entry     {r['mean_discount']:+.2%}")
    print(f"  mean implied YTR at entry  {r['mean_implied_ytr']:+.2%}")
    print(f"  mean excess over cash      {r['mean_excess']:+.2%} per position "
          f"({r['mean_excess_ann']:+.2%} annualised)")
    print(f"  hit rate {r['hit_rate']:.1%}   one-sample t {r['t_pos']:+.2f}   HAC t {r['t_pos_hac']:+.2f}")
    b = r["boot"]
    print(f"  cluster bootstrap over {b['n_clusters']} SPACs: mean {b['mean']:+.2%} "
          f"95% CI [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}]  share<0 {b['frac_negative']:.1%}")
    bb = r["boot_best"]
    print(f"  redeem-or-sell upper bound: mean {r['mean_excess_best']:+.2%} "
          f"CI [{bb['ci_low']:+.2%}, {bb['ci_high']:+.2%}]  t {r['t_pos_best']:+.2f}")

    r1 = st.race(px, cash, cost_bps=COST_BPS, one_per_spac=True)
    b1 = r1["boot"]
    print(f"\n  one position per SPAC (near-independent, n={r1['n_pos']}): "
          f"mean {r1['mean_excess']:+.2%} ({r1['mean_excess_ann']:+.2%} ann)  t {r1['t_pos']:+.2f}  "
          f"CI [{b1['ci_low']:+.2%}, {b1['ci_high']:+.2%}]  hit {r1['hit_rate']:.0%}")

    print("\n=== WHAT THAT HEADLINE IS: an identity, not a measurement ===")
    idr = r["identity"]
    print(f"  excess == (1+cash_ret) x (trust_at_signal / cost-loaded entry - 1) to "
          f"{idr['max_abs_residual']:.1e} across {idr['n']} positions "
          f"(corr with the entry discount {idr['corr_with_discount']:.4f})")
    print("  => the t, the hit rate and the cluster CI describe the ENTRY DISCOUNT.")
    print("     The redemption payoff is IMPOSED by the model, never measured. The two")
    print("     reads below are the only real-tape evidence for the trust line itself.")

    print("\n  (a) the deadline-day QUOTE against the assumed accrued trust, per SPAC:")
    anc = r["anchor"]
    gap = anc["gap"]
    print(f"      median {gap.median():+.2%}   at-or-above the line {int((gap >= 0).sum())}/{len(gap)}   "
          f"within +/-2% {int((gap.abs() <= 0.02).sum())}/{len(gap)}   worst {gap.min():+.2%} ({anc.index[0]})")
    print("      worst five: " + "  ".join(f"{t} {g:+.2%}" for t, g in gap.head(5).items()))

    bs = r["boot_sell"]
    print(f"\n  (b) SELL into that quote instead of redeeming (assumes NO trust payoff): "
          f"mean {r['mean_excess_sell']:+.2%}")
    print(f"      t {r['t_pos_sell']:+.2f}  cluster CI [{bs['ci_low']:+.2%}, {bs['ci_high']:+.2%}]  "
          f"share<0 {bs['frac_negative']:.1%}")
    bs1 = st.cluster_bootstrap_mean(r1["positions"], column="excess_sell")
    e1 = r1["positions"]["excess_sell"]
    print(f"      one per SPAC (n={len(e1)}): mean {e1.mean():+.2%}  "
          f"t {st.one_sample_t(e1.to_numpy()):+.2f}  "
          f"CI [{bs1['ci_low']:+.2%}, {bs1['ci_high']:+.2%}]  hit {(e1 > 0).mean():.0%}")

    print("\n  (c) ADVERSARIAL PAYOFF - pay the WORSE of the assumed trust and that quote,")
    print("      i.e. let the tape veto the $10.00 assumption name by name:")
    for tag, kw in (("all", {}), ("one per SPAC", {"one_per_spac": True})):
        pf = st.build_positions(px, cash, cost_bps=COST_BPS, market_floor=True, **kw)
        bf = st.cluster_bootstrap_mean(pf)
        sf = st.summary(st.portfolio_daily(pf, px, cash, mode="accrual"))
        print(f"      {tag:12s} n={len(pf):3d}  mean {pf['excess'].mean():+.2%}  "
              f"t {st.one_sample_t(pf['excess'].to_numpy()):+.2f}  "
              f"CI [{bf['ci_low']:+.2%}, {bf['ci_high']:+.2%}]  "
              f"hit {(pf['excess'] > 0).mean():.0%}  book exCAGR {sf['cagr']:+.2%}")
    gneg = gap[gap < 0]
    print(f"      ({len(gneg)}/{len(gap)} shells quoted below the assumed line on the deadline, "
          f"by {gneg.mean():+.2%} on average)")

    print("\n=== annualisation: the per-position yield is NOT the book's return ===")
    hold = pos["days"].mean()
    geo = (1.0 + r["mean_excess"]) ** (365.25 / hold) - 1.0
    print(f"  mean of the per-position annualised excess : {r['mean_excess_ann']:+.2%}  "
          f"<- over-weights short holds, do not quote as a rate of return")
    print(f"  the mean position, annualised over its {hold:.0f}-day hold : {geo:+.2%}")
    print(f"  the BOOK (equal-weighted, net of cost, excess-of-cash CAGR) : {r['acc']['cagr']:+.2%}")

    print("\n=== look-ahead check: the horizon filter uses the REALISED deal date ===")
    for md in (730, 3000):
        p = st.build_positions(px, cash, cost_bps=COST_BPS, max_days=md)
        bmd = st.cluster_bootstrap_mean(p)
        print(f"  max_days={md:5d}  n={len(p):3d}  SPACs {p['ticker'].nunique():2d}  "
              f"excess {p['excess'].mean():+.2%}  CI [{bmd['ci_low']:+.2%}, {bmd['ci_high']:+.2%}]")
    print("  the 730-day cap is ex-post knowledge; lifting it moves nothing.")

    print("\n=== the two daily books (excess-of-cash) ===")
    for tag, s in [("accrual (hold-to-redemption)", r["acc"]), ("mark-to-market", r["mtm"])]:
        print(f"  {tag:28s}: exSharpe {s['sharpe']:+.2f}  exCAGR {s['cagr']:+.2%}  "
              f"vol {s['vol_ann']:.2%}  maxDD {s['max_drawdown']:+.2%}  HAC t {s['tstat']:+.2f}")
    ci = st.bootstrap_sharpe_ci(r["e_mtm"], seed=932)
    print(f"  mark-to-market Sharpe 95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  "
          f"share<0 {ci['frac_negative']:.1%}")

    print("\n=== era cut (entry year, split 2022) ===")
    for tag, e in st.era_cut(pos).items():
        if e is None:
            continue
        eb = e["boot"]
        print(f"  {tag:5s} n={e['n_pos']:3d} ({e['n_spacs']} SPACs)  disc {e['mean_discount']:+.2%}  "
              f"excess {e['mean_excess']:+.2%} ({e['mean_excess_ann']:+.2%} ann)  "
              f"t {e['t_pos']:+.2f}  CI [{eb['ci_low']:+.2%}, {eb['ci_high']:+.2%}]  hit {e['hit_rate']:.0%}")

    print("\n=== assumption sweeps (mean excess per position, cluster CI) ===")
    grids = [
        ("cost_bps", (0.0, 15.0, 50.0, 100.0, 200.0)),
        ("trust0", data.TRUST0_GRID),
        ("buffer_days", data.BUFFER_GRID),
        ("fee_bps", data.TRUST_FEE_GRID),
        ("max_discount", data.MAX_DISCOUNT_GRID),
    ]
    for param, grid in grids:
        base = {"cost_bps": COST_BPS}
        base.pop(param, None)
        print(f"  {param}:")
        for row in st.sweep(px, cash, param, grid, **base):
            if row.get("n_pos", 0) == 0:
                print(f"    {row[param]!s:>8}  no positions")
                continue
            print(f"    {row[param]!s:>8}  n={row['n_pos']:3d}  excess {row['mean_excess']:+.2%} "
                  f"({row['mean_excess_ann']:+.2%} ann)  t {row['t_pos']:+.2f}  "
                  f"CI [{row['ci_low']:+.2%}, {row['ci_high']:+.2%}]  hit {row['hit_rate']:.0%}")

    print("\n  the same cost grid, read as the BOOK's excess-of-cash CAGR (the bankable rate):")
    for c in (0.0, 15.0, 50.0, 100.0, 200.0):
        p = st.build_positions(px, cash, cost_bps=c)
        s = st.summary(st.portfolio_daily(p, px, cash, mode="accrual"))
        print(f"    {c:5.0f} bps  book exCAGR {s['cagr']:+.2%}")

    print("\n=== cross-checks ===")
    sgov = cashf["SGOV"].dropna()
    r_sgov = st.race(px, sgov, cost_bps=COST_BPS)
    print(f"  SGOV as the cash/accrual leg: excess {r_sgov['mean_excess']:+.2%} "
          f"({r_sgov['mean_excess_ann']:+.2%} ann)  t {r_sgov['t_pos']:+.2f}  n={r_sgov['n_pos']}")
    drop = tuple(s for s in data.SPACS if s[0] not in ("CXAI",))
    r_drop = st.race(px, cash, spacs=drop, cost_bps=COST_BPS)
    print(f"  drop the print-cleaned name (CXAI): excess {r_drop['mean_excess']:+.2%} "
          f"({r_drop['mean_excess_ann']:+.2%} ann)  t {r_drop['t_pos']:+.2f}  n={r_drop['n_pos']}")

    print("\n=== the trust-yield path (median across live pre-deal SPACs) ===")
    yp = st.yield_path(px, cash)
    irx = cashf["IRX"].dropna()
    irx_m = irx.groupby(irx.index.to_period("M")).last() / 100.0
    for m, row in yp.iterrows():
        bill = irx_m.get(m, float("nan"))
        print(f"  {m}  live {int(row['n_live']):2d}  below trust {int(row['n_below']):2d}  "
              f"med disc {row['median_discount']:+.2%}  med YTR {row['median_ytr']:+.2%}  "
              f"3m bill {bill:+.2%}")

    print("\n=== synthetic control (machinery proof; never supports the stamp) ===")
    import numpy as np
    n_seeds = 16
    for ss in (1.0, 0.0):
        outs = []
        for seed in range(932, 932 + n_seeds):
            sp, sc, meta, _ = data.synthetic_panel(signal_strength=ss, seed=seed)
            outs.append(st.synthetic_detect(sp, sc, meta, cost_bps=COST_BPS))
        m = np.array([o["mean_excess"] for o in outs], dtype=float)
        t = np.array([o["t_pos"] for o in outs], dtype=float)
        fires_t = int((t > 2).sum())
        fires_ci = int(sum(1 for o in outs if o["ci_low"] > 0))
        world = "trust put binds" if ss > 0.5 else "trust put is a fiction (null)"
        print(f"  ss={ss:.0f} [{world}]: mean excess {m.mean():+.3%} (sd {m.std(ddof=1):.3%})  "
              f"naive position t>2 on {fires_t}/{n_seeds}  "
              f"cluster-bootstrap CI>0 on {fires_ci}/{n_seeds}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
