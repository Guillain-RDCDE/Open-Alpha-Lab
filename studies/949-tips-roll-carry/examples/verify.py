"""Real-tape verification — Study 949 (Riding the TIPS Curve). Regenerates docs/results.md.

Reads the cached daily total-return closes for the four inflation-linked maturity
buckets (VTIP, SCHP, TIP, LTPZ), their duration-matched nominal Treasury funds (SHY,
IEI, IEF, TLT) and the cash leg (BIL), then:

1. races the ladder **excess-of-cash** — does extending along the real curve pay?
2. strips duration out of each linker with a rolling, one-day-lagged hedge against its
   nominal match, and asks whether the residual carry is reliably positive;
3. cuts the sample around the 2021-2023 inflation shock, jackknifes 2021, sweeps costs,
   borrow, pairings and the beta window, and runs the offline synthetic control.

    python studies/949-tips-roll-carry/examples/verify.py            # cache-only
    python studies/949-tips-roll-carry/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tips_roll import data, strategy as st  # noqa: E402

COST_BPS = 2.0          # one-way, charged on |change in short weight| x NAV
BORROW_BPS_YR = 30.0    # PROXY: annual borrow on the short nominal leg, swept below
WINDOW = 252            # trailing days for the hedge beta (lagged one day)
HEADLINE = ("VTIP", "SHY")

ERAS = [
    ("pre-shock 2013-2020", "2013-01-01", "2020-12-31"),
    ("shock 2021-2023", data.SHOCK_START, data.SHOCK_END),
    ("post-shock 2024+", "2024-01-01", data.AS_OF),
]


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    head = px[list(data.HEADLINE_COLS)]

    try:
        from quantlab import repro
        stamp = repro.data_stamp("TIPS ladder + nominal matches + BIL", head,
                                 asof=data.AS_OF)
    except Exception:  # quantlab not importable from this path — fall back
        stamp = (f"[data] TIPS ladder: {len(head):,} rows  "
                 f"{head.index[0].date()} -> {head.index[-1].date()}  "
                 f"as-of {data.AS_OF}  fingerprint={data.fingerprint(head)}")
    print(stamp)

    ex = st.excess_returns(px, cash_col=data.CASH)
    cash_cagr = st.summary(ex[data.CASH])["cagr"]
    print(f"cash leg (BIL) CAGR over the window: {cash_cagr:+.2%}")

    # ---------------------------------------------------------------- ladder
    print("\n=== 1. the ladder, excess-of-cash (buy-and-hold, no timing, no turnover) ===")
    race = st.ladder_race(ex, data.TIPS_LEGS)
    absol = st.absolute_stats(px, data.TIPS_LEGS)
    for lg in data.TIPS_LEGS:
        s = race["legs"][lg]
        a = absol[lg]
        print(f"  {lg:5s} {data.TIPS_LABEL[lg]:42s} exret {s['ann_return']:+6.2%}  "
              f"vol {s['vol_ann']:6.2%}  exSharpe {s['sharpe']:+.3f}  HAC t {s['tstat']:+5.2f}  "
              f"exDD {s['max_drawdown']:+6.1%}  |  abs CAGR {a['cagr']:+.2%}  absDD {a['max_drawdown']:+.1%}")
    print(f"\n  differences vs {race['base']} (did extending duration pay?):")
    for lg, d in race["diff_vs_base"].items():
        print(f"    {lg:5s} - {race['base']}: {d['ann_diff']:+.2%}/yr  HAC t {d['t_diff']:+.2f}  "
              f"Sharpe gap {d['sharpe_gap']:+.3f}")

    print("\n  bootstrap excess-Sharpe CIs (2,000 draws, 21-day blocks):")
    for lg in data.TIPS_LEGS:
        ci = st.bootstrap_sharpe_ci(ex[lg], seed=949)
        print(f"    {lg:5s} Sharpe {ci['sharpe']:+.3f}  95% CI [{ci['ci_low']:+.3f}, "
              f"{ci['ci_high']:+.3f}]  share<0 {ci['frac_negative']:.1%}")

    # ---------------------------------------------------- duration-hedged carry
    print("\n=== 2. duration-hedged residual carry (rolling 252d beta, lagged 1 day) ===")
    print("    net = gross - 2 bps one-way on |dbeta| x NAV - 30 bps/yr borrow on |beta| x NAV")
    pairs = [(lg, data.NOMINAL_MATCH[lg]) for lg in data.TIPS_LEGS]
    for a, b in pairs:
        s = st.sleeve_stats(st.hedged_sleeve(ex[a], ex[b], window=WINDOW,
                                             cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR))
        print(f"  {a:5s} - {b:4s}: beta {s['mean_beta']:.2f}  gross {s['gross_ann']:+.2%}/yr "
              f"(t {s['t_gross']:+.2f})  net {s['net_ann']:+.2%}/yr (t {s['t_net']:+.2f})  "
              f"net Sharpe {s['sharpe_net']:+.3f}  vol {s['vol_ann']:.2%}  "
              f"turnover {s['turnover_ann']:.1f}x/yr")

    print("\n  full-sample OLS alphas (IN-SAMPLE hedge -- reference only, not tradable):")
    for a, b in pairs:
        f = st.static_alpha(ex[a], ex[b])
        print(f"    {a:5s} ~ {b:4s}: alpha {f['alpha_ann']:+.2%}/yr  beta {f['beta']:.3f}  "
              f"HAC t {f['t_alpha']:+.2f}  R2 {f['r2']:.3f}")

    a, b = HEADLINE
    sleeve = st.hedged_sleeve(ex[a], ex[b], window=WINDOW,
                              cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR)
    print(f"\n  bootstrap CIs on the {a}-{b} carry (2,000 draws, 21-day blocks):")
    for tag in ("gross", "net"):
        ci = st.bootstrap_mean_ci(sleeve[tag], seed=949)
        print(f"    {tag:5s}: {ci['point']:+.2%}/yr  95% CI [{ci['ci_low']:+.2%}, "
              f"{ci['ci_high']:+.2%}]  share<0 {ci['frac_negative']:.1%}")

    # ------------------------------------------------------------- robustness
    print("\n=== 3. era cut around the 2021-2023 inflation shock (GROSS and net) ===")
    print("    NB the 'pre-shock' era starts at the sleeve's first day, not 2013-01-01:")
    print("    the 252-day beta warmup eats the first year, so it is ~7.2 years, not 8.")
    for a, b in pairs:
        eras = st.era_cut(ex, a, b, ERAS, window=WINDOW,
                          cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR)
        for label, _, _ in ERAS:
            e = eras[label]
            print(f"  {a:5s} - {b:4s}  {label:20s} {e['start']} -> {e['end']} "
                  f"({e['n_days']:4d}d)  gross {e['gross_ann']:+.2%} (t {e['t_gross']:+.2f})  "
                  f"net {e['net_ann']:+.2%} (t {e['t_net']:+.2f})")

    print("\n=== 4. jackknife -- drop 2021, the year the shock landed ===")
    print("    (2021's RETURNS are excised; the hedge beta is NOT re-estimated without")
    print("     2021, so betas in early 2022 still carry a 2021 estimation window.)")
    for a, b in pairs:
        k = st.drop_year(ex, a, b, 2021, window=WINDOW,
                         cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR)
        print(f"  {a:5s} - {b:4s} ex-2021: gross {k['gross_ann']:+.2%}/yr (t {k['t_gross']:+.2f})  "
              f"net {k['net_ann']:+.2%}/yr (t {k['t_net']:+.2f})")

    print(f"\n=== 5. calendar years of the {HEADLINE[0]}-{HEADLINE[1]} sleeve (%) ===")
    print("    2013 and 2026 are PART years (beta warmup / as-of cut) -- flagged below,")
    print("    so the unweighted average of calendar years is not a return.")
    tbl = st.calendar_year_table(sleeve)
    for y, row in tbl.iterrows():
        part = "  <- part year" if int(row["n_days"]) < 200 else ""
        print(f"  {y}: gross {row['gross_pct']:+6.2f}%  net {row['net_pct']:+6.2f}%  "
              f"(n={int(row['n_days'])}){part}")

    print("\n=== 6. friction sweeps (VTIP-SHY) ===")
    sw = st.cost_borrow_sweep(ex, *HEADLINE, window=WINDOW)
    for r in sw["cost_sweep"]:
        print(f"  one-way cost {r['cost_bps']:5.1f} bps (borrow 30): net {r['net_ann']:+.2%}/yr  t {r['t_net']:+.2f}")
    for r in sw["borrow_sweep"]:
        print(f"  borrow {r['borrow_bps_yr']:5.1f} bps/yr (cost 2): net {r['net_ann']:+.2%}/yr  "
              f"t {r['t_net']:+.2f}  Sharpe {r['sharpe_net']:+.3f}")

    print("\n=== 7. pairing & window robustness ===")
    for lg in ("VTIP", "SCHP", "LTPZ"):
        for r in st.pairing_sweep(ex, lg, data.ALT_MATCH[lg], window=WINDOW,
                                  cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR):
            print(f"  {lg:5s} hedged with {r['nominal']:4s}: beta {r['mean_beta']:.2f}  "
                  f"gross {r['gross_ann']:+.2%} (t {r['t_gross']:+.2f})  "
                  f"net {r['net_ann']:+.2%} (t {r['t_net']:+.2f})")
    for r in st.window_sweep(ex, *HEADLINE, cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR):
        print(f"  VTIP-SHY beta window {r['window']:3d}d: beta {r['mean_beta']:.2f}  "
              f"net {r['net_ann']:+.2%}/yr  t {r['t_net']:+.2f}")

    # ---------------------------------------------------- every t, in one place
    print("\n=== 8. EVERY real-tape HAC t in the study, ranked (the multiplicity audit) ===")
    print("    The headline sleeve VTIP-SHY was picked AFTER the fact as the best of four.")
    print("    So here is every t this study computed on the real tape, best first.")
    cands = []
    for lg in data.TIPS_LEGS:
        cands.append((race["legs"][lg]["tstat"], f"ladder leg {lg} excess-of-cash"))
    for lg, d in race["diff_vs_base"].items():
        cands.append((d["t_diff"], f"ladder {lg}-{race['base']} difference"))
    for a, b in pairs:
        cands.append((st.static_alpha(ex[a], ex[b])["t_alpha"],
                      f"{a}~{b} full-sample OLS alpha (IN-SAMPLE)"))
    for lg in data.TIPS_LEGS:
        for alt in data.ALT_MATCH[lg]:
            s = st.sleeve_stats(st.hedged_sleeve(ex[lg], ex[alt], window=WINDOW,
                                                 cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR))
            cands.append((s["t_gross"], f"{lg}-{alt} sleeve, gross, 252d beta"))
            cands.append((s["t_net"], f"{lg}-{alt} sleeve, net, 252d beta"))
    for r in st.window_sweep(ex, *HEADLINE, cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR):
        cands.append((r["t_net"], f"VTIP-SHY sleeve, net, {r['window']}d beta"))
    for a, b in pairs:
        eras = st.era_cut(ex, a, b, ERAS, window=WINDOW,
                          cost_bps=COST_BPS, borrow_bps_yr=BORROW_BPS_YR)
        for label, _, _ in ERAS:
            cands.append((eras[label]["t_gross"], f"{a}-{b} gross, sub-window '{label}'"))
            cands.append((eras[label]["t_net"], f"{a}-{b} net, sub-window '{label}'"))
    cands.sort(key=lambda z: -abs(z[0]))
    for t, name in cands[:8]:
        flag = "  <== CLEARS |t|>=2" if abs(t) >= 2.0 else ""
        print(f"  t {t:+5.2f}  {name}{flag}")
    over = [c for c in cands if abs(c[0]) >= 2.0]
    print(f"  ... {len(cands)} statistics computed on the real tape; "
          f"{len(over)} reach |t| >= 2.")
    if over:
        print("    The ONLY one that does is a hand-picked 3-year sub-window of the")
        print("    best-of-four sleeve -- i.e. the inflation shock the study names as the")
        print("    explanation. On the FULL sample nothing exceeds |t| = "
              f"{max(abs(t) for t, nm in cands if 'sub-window' not in nm):.2f}.")

    # --------------------------------------------------------------- synthetic
    print("\n=== 9. synthetic control (offline; machinery proof, never supports the stamp) ===")
    pl = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=949)[0])
    nu = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=949)[0])
    print(f"  planted 2.00%/yr carry: recovered {pl['gross_ann']:+.2%}/yr  HAC t {pl['t_gross']:+.2f}  "
          f"(beta {pl['mean_beta']:.2f})")
    print(f"  null (no carry planted): recovered {nu['gross_ann']:+.2%}/yr  HAC t {nu['t_gross']:+.2f}")
    ts = []
    for s in range(8):
        d = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=949 + s)[0])
        ts.append(d["t_gross"])
    ts = np.asarray(ts)
    print(f"  null across 8 seeds: mean t {ts.mean():+.2f}  sd {ts.std(ddof=1):.2f}  "
          f"fires |t|>=2 on {(np.abs(ts) >= 2).sum()}/8")
    panel_pl, truth_pl = data.synthetic_panel(signal_strength=1.0, seed=949)
    dl = st.synthetic_ladder_detect(panel_pl, truth_pl)
    carries = [dl["carries"][f"bucket_{i}"]["gross_ann"] for i in range(truth_pl["n_buckets"])]
    print("  ladder panel (same carry planted in all three buckets): recovered "
          + ", ".join(f"{c:+.2%}" for c in carries)
          + f"  (planted {dl['planted_carry_ann']:+.2%}) -> flat term structure, as it should be")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
