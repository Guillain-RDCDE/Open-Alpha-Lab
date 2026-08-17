"""Real-tape verification — Study 957 (Holdco Discount). Regenerates docs/results.md.

Reads the cached price-only and total-return closes for seven listed holding companies and
their dominant listed stakes, builds the observable-NAV discount, and answers the two
questions in order: (a) does the discount mean-revert, (b) does buying the wide discount
pay after costs and borrow. Prints the panel, the Driscoll-Kraay pooled regression, the
timed-versus-always-on hedged-pair race, bootstrap Sharpe CIs, the era cut, the cost and
borrow sweeps, the NAV-calibration sweep, leave-one-out, the long-only race and the
synthetic control. Network only on ``--fetch``.

    python studies/957-holdco-nav-discount/examples/verify.py            # cache-only
    python studies/957-holdco-nav-discount/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from holdco_nav import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()

    px = data.load_price_only()
    tr = data.load_prices()
    panel = st.build_panel(px, tr, data.STAKES, safe=data._safe)
    cash = data.cash_index_from_irx(px["IRX"])

    starts = min(f.index[0] for f in panel.values())
    ends = max(f.index[-1] for f in panel.values())
    fp_cols = sorted({data._safe(c["holdco"]) for c in data.STAKES.values()}
                     | {data._safe(c["stake"]) for c in data.STAKES.values()})
    print(f"panel {len(panel)} holdcos  {starts.date()} -> {ends.date()}  "
          f"fp={data.fingerprint(px[fp_cols])}   as-of {data.AS_OF}")

    print("\n=== the panel (NAV proxy = k * stake price + other; both are ASSUMPTIONS) ===")
    for name, f in panel.items():
        cfg = data.STAKES[name]
        print(f"  {name:18s} {cfg['holdco']:8s}/{cfg['stake']:8s} n={len(f):5d} "
              f"{f.index[0].date()}->{f.index[-1].date()}  k={cfg['k']:.5f} "
              f"o={cfg['other_per_share']:+.1f}  discount mean {f['d'].mean():+.1%} "
              f"sd {f['d'].std():.1%}  AR(1) half-life {st.halflife(f['d']):.0f}d")

    print("\n=== (a) does the discount mean-revert? "
          "fwd 126-day change in d regressed on today's trailing z ===")
    mr = st.mean_reversion_test(panel)
    for name, v in mr["per_name"].items():
        print(f"  {name:18s} n={v.get('n', 0):5d}  beta={v.get('beta', float('nan')):+.4f}  "
              f"HAC t={v.get('t_beta', float('nan')):+.2f}  R2={v.get('r2', float('nan')):.3f}")
    p = mr["pooled"]
    print(f"  POOLED (Driscoll-Kraay) n={p['n']:,} over {p['n_days']:,} days  "
          f"beta={p['beta']:+.4f}  t={p['t_beta']:+.2f}  R2={p['r2']:.4f}   "
          f"negative beta in {mr['n_names_negative_beta']}/{mr['n_names']} names")

    print("\n=== (b) the hedged-pair race (both arms are excess-of-cash by construction) ===")
    cmp = st.compare(panel)
    for tag, arm in [("timed (wide-discount)", cmp["timed"]), ("always-on", cmp["always_on"])]:
        s, g = arm["summary_net"], arm["summary_gross"]
        print(f"  {tag:22s} net Sharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.2%}  "
              f"vol {s['vol_ann']:.2%}  maxDD {s['max_drawdown']:+.1%}  HAC t {s['tstat']:+.2f}"
              f"   | gross Sharpe {g['sharpe']:+.3f} (t {g['tstat']:+.2f})"
              f"  invested {arm['invested_frac']:.1%}  avg names {arm['avg_names_held']:.2f}")
    print(f"  timing advantage (timed - always-on): {cmp['sharpe_adv']:+.3f}  "
          f"HAC t on the daily difference {cmp['t_diff']:+.2f}")

    print("\n=== hedge variant: shorting the whole look-through stake value instead ===")
    cmp_lt = st.compare(panel, hedge="lookthrough")
    print(f"  timed net Sharpe {cmp_lt['sharpe_timed']:+.3f} (t={cmp_lt['t_timed']:+.2f})  "
          f"gross {cmp_lt['timed']['summary_gross']['sharpe']:+.3f} "
          f"(t={cmp_lt['t_timed_gross']:+.2f})  always-on {cmp_lt['sharpe_always_on']:+.3f}")

    print("\n=== bootstrap Sharpe CIs (2,000 draws, 21-day blocks) ===")
    for tag, r in [("timed net", cmp["timed"]["net"]), ("timed gross", cmp["timed"]["gross"]),
                   ("always-on net", cmp["always_on"]["net"])]:
        ci = st.bootstrap_sharpe_ci(r, seed=957)
        print(f"  {tag:14s} Sharpe {ci['sharpe']:+.3f}  95% CI "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  share<0 {ci['frac_negative']:.1%}")

    print("\n=== era cut (split 2016-01-01) ===")
    for tag, e in st.era_cut(panel).items():
        if e is None:
            continue
        print(f"  {tag:5s} names={e['n_names']} n={e['n_days']:5d}  timed {e['sharpe_timed']:+.3f} "
              f"(t={e['t_timed']:+.2f})  always-on {e['sharpe_always_on']:+.3f}  "
              f"adv {e['sharpe_adv']:+.3f}  | pooled beta {e['pooled_beta']:+.4f} "
              f"(t={e['pooled_t']:+.2f})")

    print("\n=== cost sweep (one-way bps per leg, borrow held at 100 bps) ===")
    for r in st.cost_sweep(panel):
        print(f"  {r['cost_bps']:5.1f} bps: timed {r['sharpe_timed']:+.3f} (t={r['t_timed']:+.2f})  "
              f"always-on {r['sharpe_always_on']:+.3f}  adv {r['sharpe_adv']:+.3f}")

    print("\n=== borrow sweep (annualised bps on the short leg, cost held at 10 bps) ===")
    for r in st.borrow_sweep(panel):
        print(f"  {r['borrow_bps']:5.1f} bps: timed {r['sharpe_timed']:+.3f} (t={r['t_timed']:+.2f})  "
              f"always-on {r['sharpe_always_on']:+.3f}")

    print("\n=== threshold sweep: the rule's only free parameters (headline enter 1.0 / exit 0.0) ===")
    rows = st.threshold_sweep(panel)
    for r in rows:
        flag = "   <- headline" if (r["enter"] == 1.0 and r["exit"] == 0.0) else ""
        print(f"  enter {r['enter']:+.2f} exit {r['exit']:+.2f}: gross {r['sharpe_gross']:+.3f} "
              f"(t={r['t_gross']:+.2f})  net {r['sharpe_timed']:+.3f} (t={r['t_timed']:+.2f})  "
              f"invested {r['invested_frac']:.0%}{flag}")
    tn = [r["t_timed"] for r in rows]
    tg = [r["t_gross"] for r in rows]
    print(f"  ACROSS THE WHOLE GRID: net t in [{min(tn):+.2f}, {max(tn):+.2f}] "
          f"(cells with net t >= 2: {sum(1 for v in tn if v >= 2.0)}/{len(tn)});  "
          f"gross t in [{min(tg):+.2f}, {max(tg):+.2f}]")

    print("\n=== trailing-standardisation window sweep (headline 504 days) ===")
    for r in st.zwindow_sweep(px, tr, data.STAKES, safe=data._safe):
        print(f"  z window {r['z_window']:5d}: gross {r['sharpe_gross']:+.3f} "
              f"(t={r['t_gross']:+.2f})  net {r['sharpe_timed']:+.3f} (t={r['t_timed']:+.2f})  "
              f"pooled beta {r['pooled_beta']:+.4f} (t={r['pooled_t']:+.2f})")

    print("\n=== NAV-calibration sweep (is any of this an artefact of k and o?) ===")
    for r in st.calibration_sweep(px, tr, data.STAKES, safe=data._safe):
        print(f"  k x{r['k_mult']:.1f}  other x{r['other_mult']:.1f}: timed Sharpe "
              f"{r['sharpe_timed']:+.3f} (t={r['t_timed']:+.2f})  "
              f"pooled beta {r['pooled_beta']:+.4f} (t={r['pooled_t']:+.2f})")

    print("\n=== leave-one-out ===")
    for r in st.leave_one_out(panel):
        print(f"  drop {r['dropped']:18s} timed {r['sharpe_timed']:+.3f} (t={r['t_timed']:+.2f})  "
              f"adv {r['sharpe_adv']:+.3f}  pooled beta {r['pooled_beta']:+.4f} "
              f"(t={r['pooled_t']:+.2f})")

    print("\n=== leave-one-out on the ALWAYS-ON arm (did the discounts widen?) ===")
    for drop in list(panel.keys()) + [None]:
        sub = {k: v for k, v in panel.items() if k != drop}
        arm = st.run_pairs(sub, always_on=True)
        s = arm["summary_net"]
        tag = "full panel" if drop is None else f"drop {drop}"
        print(f"  {tag:24s} always-on Sharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.2%}  "
              f"HAC t {s['tstat']:+.2f}")

    print("\n=== per-name attribution: each name's own timed pair, standalone ===")
    for name in panel:
        cmp1 = st.compare({name: panel[name]})
        s = cmp1["timed"]["summary_net"]
        print(f"  {name:18s} net Sharpe {s['sharpe']:+.3f}  HAC t {s['tstat']:+.2f}  "
              f"gross {cmp1['timed']['summary_gross']['sharpe']:+.3f}  "
              f"invested {cmp1['timed']['invested_frac']:.0%}")

    print("\n=== stale-quote robustness: primary listings only (no thin OTC ADRs) ===")
    liquid = ["HEINEKEN_HOLDING", "CHRISTIAN_DIOR", "LIBERTY_BROADBAND", "BOLLORE"]
    sub = {k: v for k, v in panel.items() if k in liquid}
    cmp_l = st.compare(sub)
    mr_l = st.mean_reversion_test(sub)
    s = cmp_l["timed"]["summary_net"]
    print(f"  {len(sub)} names (HEIO, CDI, LBRDK, BOL): timed net Sharpe {s['sharpe']:+.3f} "
          f"(t={s['tstat']:+.2f})  gross {cmp_l['timed']['summary_gross']['sharpe']:+.3f} "
          f"(t={cmp_l['t_timed_gross']:+.2f})  always-on {cmp_l['sharpe_always_on']:+.3f}")
    print(f"  pooled mean-reversion beta {mr_l['pooled']['beta']:+.4f} "
          f"(t={mr_l['pooled']['t_beta']:+.2f})")
    adr = {k: v for k, v in panel.items() if k not in liquid}
    cmp_a = st.compare(adr)
    sa = cmp_a["timed"]["summary_net"]
    print(f"  the 3 ADR pairs alone (NPSNY, PROSY, SFTBY): timed net Sharpe {sa['sharpe']:+.3f} "
          f"(t={sa['tstat']:+.2f})  gross {cmp_a['timed']['summary_gross']['sharpe']:+.3f} "
          f"(t={cmp_a['t_timed_gross']:+.2f})")

    print("\n=== long-only race, explicitly excess-of-cash (^IRX proxy cash leg) ===")
    lo = st.long_only_race(panel, cash)
    for tag in ("timed", "holdcos", "stakes"):
        s = lo[tag]
        print(f"  {tag:8s} excess Sharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.2%}  "
              f"vol {s['vol_ann']:.2%}  maxDD {s['max_drawdown']:+.1%}  HAC t {s['tstat']:+.2f}")
    print(f"  timed vs always-hold-the-holdcos: HAC t on the difference "
          f"{lo['t_diff_vs_holdcos']:+.2f}   invested {lo['invested_frac']:.1%}")
    cc = st.cash_cross_check(cash, tr["BIL"])
    print(f"  cash-leg cross-check over BIL's window (n={cc['n_days']:,}): "
          f"^IRX proxy {cc['irx_cagr']:.2%}/yr vs BIL total return {cc['bil_cagr']:.2%}/yr")

    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    for ss in (1.0, 0.0):
        rows = [st.synthetic_detect(data.synthetic_panel(signal_strength=ss, seed=957 + 37 * s)[0])
                for s in range(4)]
        pt = np.array([r["pooled_t"] for r in rows])
        gs = np.array([r["sharpe_timed_gross"] for r in rows])
        tag = "planted OU" if ss == 1.0 else "random-walk null"
        print(f"  {tag:17s} pooled beta t: mean {pt.mean():+.2f} "
              f"({', '.join(f'{v:+.1f}' for v in pt)})   gross timed Sharpe: mean {gs.mean():+.3f} "
              f"({', '.join(f'{v:+.2f}' for v in gs)})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
