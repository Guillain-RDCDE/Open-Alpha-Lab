"""Real-tape verification — Study 931 (the CEF IPO hole). Regenerates docs/results.md.

Reads the shared cache (28 closed-end-fund IPOs, 2012-2022 vintages, plus one
asset-class benchmark ETF each) and runs the whole event study: the abnormal
total return from the offering price at 1/3/6/12 months, the one-sample t, the
vintage-cluster bootstrap, the calendar-time portfolio with its HAC t, the
time-to-discount proxy, the vintage era cut, the benchmark-mapping and entry
sweeps, the pseudo-IPO placebo, the seasoned-window mirror, and the costed
short trade with its borrow sweep. Network only on ``--fetch``.

    python studies/931-cef-ipo-decay/examples/verify.py            # cache-only
    python studies/931-cef-ipo-decay/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cef_ipo import data, strategy as st  # noqa: E402


def synthetic_demo() -> None:
    """Offline fallback when the shared cache is absent: the machinery on synthetic data."""
    print("No shared _cache found - running the OFFLINE SYNTHETIC demo instead.")
    print("(These are simulated funds. Nothing below is a real-tape number.)\n")
    planted, truth = data.synthetic_panel(signal_strength=1.0, seed=931)
    null, _ = data.synthetic_panel(signal_strength=0.0, seed=931)
    p = st.synthetic_detect(planted)
    n = st.synthetic_detect(null)
    print(f"  planted -{truth['planted_slide']*100:.0f}% first-year hole -> "
          f"measured {p['mean_abn_pct']:+.2f}%  t={p['tstat']:+.2f}  "
          f"negative {p['share_negative']:.0%}")
    print(f"  null (nothing planted)         -> "
          f"measured {n['mean_abn_pct']:+.2f}%  t={n['tstat']:+.2f}")
    prices, ipos, raw = _synthetic_frame(planted)
    sp = st.calendar_time_spread(prices, ipos, window=252)
    s = st.calendar_time_stats(sp)
    print(f"  calendar-time spread on the planted panel: {s['ann_pct']:+.2f}%/yr  "
          f"HAC t={s['tstat_hac']:+.2f}")
    tr = st.short_trade(prices, ipos, cost_bps=20.0, borrow_bps_year=500.0)
    print(f"  mirror short (20 bps, 500 bps borrow): {tr['mean_net_pct']:+.2f}%  "
          f"t={tr['tstat']:+.2f}")
    print("\nRun with --fetch to populate the shared cache and get the real-tape run.")


def _synthetic_frame(panel):
    cols, ipos = {}, []
    for key, px in panel.items():
        cols[key] = px["fund"]
        cols[key + "_B"] = px["bench"]
        ipos.append((key, 20.0, "synthetic", key + "_B"))
    prices = pd.DataFrame(cols).sort_index()
    prices.index.name = "date"
    return prices, tuple(ipos), prices[[k for k, _, _, _ in ipos]].copy()


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    if not data.have_real():
        synthetic_demo()
        return
    px = data.load_prices()
    raw = data.load_raw_closes()

    tbl = st.build_table(px, data.CEF_IPOS, raw=raw, entry="offer")
    first_ipo = tbl["ipo_date"].min()
    stamp = px.loc[px.index >= first_ipo]
    print(f"{len(tbl)} CEF IPOs, vintages {tbl['vintage'].min()}-{tbl['vintage'].max()}")
    print(f"tape {stamp.index[0].date()} -> {stamp.index[-1].date()}  "
          f"n={len(stamp)}  fp={data.fingerprint(stamp)}   as-of {data.AS_OF}")
    near = int((tbl["day1_gap_pct"].abs() <= 1.0).sum())
    print(f"offering-price assumption check: day-1 close within 1% of the assumed "
          f"offer for {near}/{len(tbl)} funds "
          f"(worst {tbl['day1_gap_pct'].min():+.1f}%)")
    counts = tbl["vintage"].value_counts().sort_index()
    empty = [y for y in range(int(counts.index.min()), int(counts.index.max()) + 1)
             if y not in counts.index]
    print("vintages: " + "  ".join(f"{int(y)}:{int(n)}" for y, n in counts.items()))
    print(f"  EMPTY vintages in the stated window: {empty} - the list is a hand-built "
          f"convenience sample of large launches, NOT a census of CEF IPOs")

    print("\n=== abnormal total return vs asset-class benchmark (entry = offering price) ===")
    summ = st.horizon_summary(tbl)
    for lab, row in summ.iterrows():
        print(f"  {lab:>3s}: mean {row['mean_abn_pct']:+6.2f}%  median {row['median_abn_pct']:+6.2f}%  "
              f"negative {row['share_negative']:.0%} "
              f"[{row['share_neg_lo']:.2f},{row['share_neg_hi']:.2f}]  t={row['tstat']:+.2f}")

    print("\n=== vintage-cluster bootstrap of the mean (resamples whole IPO years) ===")
    for lab in ("3m", "6m", "12m"):
        ci = st.cluster_bootstrap_ci(tbl[f"abn_{lab}"], tbl["vintage"])
        print(f"  {lab:>3s}: mean {ci['mean']*100:+6.2f}%  95% CI "
              f"[{ci['ci_low']*100:+6.2f}%, {ci['ci_high']*100:+6.2f}%]  "
              f"share>0 {ci['frac_positive']:.3f}  clusters={ci['n_clusters']}")

    print("\n=== calendar-time portfolio (cross-correlation-proof, HAC t; entered t+1) ===")
    for w, lab in ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        sp = st.calendar_time_spread(px, data.CEF_IPOS, window=w)
        s = st.calendar_time_stats(sp)
        b = st.block_bootstrap_mean_ci(sp)
        print(f"  age<={lab:>3s}: n={s['n_days']:4d}  {s['ann_pct']:+7.2f}%/yr  "
              f"HAC t={s['tstat_hac']:+.2f}  block-boot CI "
              f"[{b['ci_low']:+.2f}%, {b['ci_high']:+.2f}%]")
    by_fund = []
    for tk, o, s_, b_ in data.CEF_IPOS:
        one = ((tk, o, s_, b_),)
        sp1 = st.calendar_time_spread(px, one, window=252)
        if len(sp1):
            by_fund.append(sp1.mean() * st.TRADING_DAYS * 100)
    print(f"  (equal-weighting by FUND instead of by DAY over the same first-year windows: "
          f"{np.mean(by_fund):+.2f}%/yr - the day-weighted book under-weights the crowded "
          f"2019-2021 cohort)")
    n_names = []
    for tk, o, s_, b_ in data.CEF_IPOS:
        f = px[tk].dropna()
        age = np.arange(len(f))
        n_names.append(pd.Series(1.0, index=f.index[(age > 1) & (age <= 63)], name=tk))
    book = pd.concat(n_names, axis=1, sort=True).notna().sum(axis=1)
    book = book[book > 0]
    print(f"  (the 3m book holds {book.mean():.2f} funds on an average day, max {int(book.max())} "
          f"- the calendar-time test is conservative, not more powerful)")

    print("\n=== beta control: is the hole just leverage? (market-model CAR, t+1 entry) ===")
    bt = st.beta_adjusted_table(px, data.CEF_IPOS)
    bs = st.beta_adjusted_summary(bt)
    print(f"  seasoned beta to own benchmark: mean {bs['mean_beta']:.2f}  "
          f"median {bs['median_beta']:.2f}  "
          f"(min {bt['beta'].min():.2f}, max {bt['beta'].max():.2f})")
    print(f"  CAR beta=1        : mean {bs['beta1_mean_pct']:+6.2f}%  "
          f"negative {bs['beta1_share_neg']:.0%}  t={bs['beta1_t']:+.2f}")
    print(f"  CAR beta=fitted   : mean {bs['betafit_mean_pct']:+6.2f}%  "
          f"negative {bs['betafit_share_neg']:.0%}  t={bs['betafit_t']:+.2f}   "
          f"(beta fitted on months 12-36 = AFTER the event window: a DIAGNOSTIC, not a rule)")

    print("\n=== time-to-discount PROXY (days until the fund is X% behind its benchmark) ===")
    for thr in (3.0, 5.0, 10.0):
        t2 = st.time_to_discount_table(px, data.CEF_IPOS, raw, threshold_pct=thr)
        s = st.time_to_discount_stats(t2)
        print(f"  -{thr:.0f}%: {s['n_crossed']}/{s['n_funds']} funds cross within 2 years  "
              f"median {s['median_days']:.0f} trading days  "
              f"IQR [{s['q1_days']:.0f}, {s['q3_days']:.0f}]")

    print("\n=== era cut by IPO vintage (12m) ===")
    for tag, e in st.era_cut(tbl, split_year=2019).items():
        print(f"  {tag:5s} n={e['n_funds']:2d}  vintages {e['vintages']}  "
              f"mean {e['mean_abn_pct']:+6.2f}%  negative {e['share_negative']:.0%}  t={e['tstat']:+.2f}")

    print("\n=== benchmark-mapping sweep (ASSUMPTION stress, 12m) ===")
    for r in st.mapping_sweep(px, data.CEF_IPOS, raw, data.ALT_MAPPINGS):
        print(f"  {r['mapping']:12s}: mean {r['mean_abn_pct']:+6.2f}%  "
              f"negative {r['share_negative']:.0%}  t={r['tstat']:+.2f}")

    print("\n=== entry sweep (12m) ===")
    for r in st.entry_sweep(px, data.CEF_IPOS, raw):
        print(f"  entry={r['entry']:6s}: mean {r['mean_abn_pct']:+6.2f}%  t={r['tstat']:+.2f}")

    print("\n=== pseudo-IPO placebo (same funds, fake seasoned event dates, 12m) ===")
    pl = st.pseudo_ipo_placebo(px, data.CEF_IPOS)
    print(f"  mean {pl['mean_pct']:+.2f}%  sd {pl['sd_pct']:.2f}%  "
          f"[p05 {pl['p05_pct']:+.2f}%, p95 {pl['p95_pct']:+.2f}%]  n_draws={pl['n_draws']}")

    print("\n=== the mirror: the same funds once seasoned (months 12 -> 36) ===")
    se = st.seasoned_table(px, data.CEF_IPOS)["abn_seasoned"].dropna()
    print(f"  n={len(se)}  mean {se.mean()*100:+.2f}%  median {se.median()*100:+.2f}%  "
          f"negative {(se < 0).mean():.0%}  t={st.one_sample_t(se):+.2f}")
    sp_sea = st.calendar_time_spread(px, data.CEF_IPOS, window=756, start_day=252)
    s = st.calendar_time_stats(sp_sea)
    print(f"  calendar-time (age 12-36m): {s['ann_pct']:+.2f}%/yr  HAC t={s['tstat_hac']:+.2f}")

    print("\n=== the tradable mirror: short the new CEF, long the benchmark, 12m ===")
    for r in st.cost_borrow_sweep(px, data.CEF_IPOS):
        print(f"  cost={r['cost_bps']:4.0f}bps  borrow={r['borrow_bps_year']:5.0f}bps/yr  "
              f"net {r['mean_net_pct']:+6.2f}%  win {r['share_positive']:.0%}  t={r['tstat']:+.2f}")
    r0 = st.short_trade(px, data.CEF_IPOS, cost_bps=20.0, borrow_bps_year=0.0)
    print(f"  break-even borrow rate ~ {r0['mean_net_pct'] * 100:.0f} bps/yr "
          f"(net of 20 bps x 4 legs)")
    print("  3-month version (where the hole actually opens):")
    for bo in (0.0, 500.0, 1500.0, 3000.0):
        r = st.short_trade(px, data.CEF_IPOS, horizon=63, cost_bps=20.0, borrow_bps_year=bo)
        print(f"    borrow={bo:5.0f}bps/yr  net {r['mean_net_pct']:+6.2f}%  "
              f"win {r['share_positive']:.0%}  t={r['tstat']:+.2f}")

    print("\n=== synthetic control (machinery proof only) ===")
    pl_panel, tr = data.synthetic_panel(signal_strength=1.0, seed=931)
    nl_panel, _ = data.synthetic_panel(signal_strength=0.0, seed=931)
    dp = st.synthetic_detect(pl_panel)
    dn = st.synthetic_detect(nl_panel)
    print(f"  planted -{tr['planted_slide']*100:.0f}%/yr: recovered {dp['mean_abn_pct']:+.2f}%  "
          f"t={dp['tstat']:+.2f}  negative {dp['share_negative']:.0%}")
    print(f"  null            : recovered {dn['mean_abn_pct']:+.2f}%  t={dn['tstat']:+.2f}")
    nulls = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=931 + s)[0])["tstat"]
        for s in range(8)
    ])
    print(f"  null x8 seeds   : t mean {nulls.mean():+.2f} sd {nulls.std(ddof=1):.2f}  "
          f"|t|>=2 in {(np.abs(nulls) >= 2).sum()}/8")
    lev_panel, _ = data.synthetic_panel(signal_strength=0.0, beta=1.6, seed=931)
    lev_px, lev_ipos = _synthetic_frame(lev_panel)[:2]
    bs = st.beta_adjusted_summary(st.beta_adjusted_table(lev_px, lev_ipos))
    print(f"  levered null (beta 1.6 planted, NO hole): fitted beta {bs['mean_beta']:.2f}  "
          f"CAR beta=1 {bs['beta1_mean_pct']:+.2f}% (biased)  "
          f"CAR beta=fitted {bs['betafit_mean_pct']:+.2f}% (unbiased)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
