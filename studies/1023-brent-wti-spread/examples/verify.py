"""Real-tape verification — Study 1023 (The Cushing Glut). Regenerates docs/results.md.

Reads the frozen monthly Brent/WTI spot tape shipped inside ``arch`` (SHA-256 pinned through
``quantlab.bundled``), then:

1. tests the spread (dollars and log ratio) for a unit root with ADF and KPSS over the full
   sample and sub-samples, and the two log prices for Engle–Granger cointegration;
2. locates a mean break with a sup-F scan (bootstrap p-value under a persistent no-break
   AR(1)), an HAC-scaled CUSUM and a Bai–Perron dynamic program, and estimates the AR(1)
   half-life inside each data-chosen regime;
3. books a real-time z-score fader (expanding and rolling windows; one lag; costs and a roll
   toll) and the trader frozen on 1987–2009 parameters — gross, net, maximum adverse
   excursion, months underwater, a cost sweep;
4. measures, on synthetic OU spreads calibrated to the pre-break regime, the power of ADF and
   KPSS at n = 393 and what a planted level break does to them.

    python studies/1023-brent-wti-spread/examples/verify.py

No network. Prints a ``##HEADLINE## {json}`` line at the end.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cushing import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
N_BOOT_BREAK = 499
N_REPS_POWER = 100


def _d(ts) -> str:
    return "—" if ts is None else str(pd.Timestamp(ts).date())[:7]


def _be(v: float) -> str:
    return f"{v:.0f}" if v > 0 else "n/a (loses gross)"


def _hl(v: float) -> str:
    return "∞" if not np.isfinite(v) else f"{v:.1f}"


def report() -> dict:
    t0 = time.time()
    import arch
    px = data.load_crude()
    sf = st.spread_frame(px)
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(px),
               "sha256": data.SHA256, "package": f"arch {arch.__version__}",
               "n_months": int(len(px)), "first": _d(px.index[0]), "last": _d(px.index[-1])}
    print(f"as-of {data.AS_OF}  fingerprint {h['fingerprint']}  sha256 {data.SHA256[:12]}…")
    print(f"  {len(px)} months {h['first']} → {h['last']}")
    chk = data.check_monthly_average(px)
    h["avg_check"] = chk
    print(f"  monthly WTI vs daily FRED WTI: median |gap| to monthly MEAN "
          f"${chk['median_abs_gap_avg']:.4f}, to month-END ${chk['median_abs_gap_end']:.2f}"
          f"  -> monthly average: {chk['is_monthly_average']}")

    # ------------------------------------------------------------------ 1
    print("\n=== 1. is the spread stationary? ===")
    sfm = st.sup_f_mean(sf["log_ratio"])
    bdate = sfm["break_date"]
    pre_end = (bdate - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    windows = [("full sample", None, None),
               ("1987–2003", None, "2003-12-31"),
               ("1987–2009 (pre-2010)", None, "2009-12-31"),
               (f"pre-break (→ {_d(pd.Timestamp(pre_end))})", None, pre_end),
               (f"post-break ({_d(bdate)} →)", bdate, None),
               ("2010–2014 (the glut)", "2010-01-31", "2014-12-31"),
               ("2015–2020", "2015-01-31", None)]
    stab = st.stationarity_table(sf, windows)
    print(stab.round(3).to_string())
    h["stationarity"] = stab.to_dict("records")
    full_lr = stab[(stab.window == "full sample") & (stab.series == "log_ratio")].iloc[0]
    h["adf_p_full"], h["kpss_p_full"] = float(full_lr.adf_p), float(full_lr.kpss_p)
    full_usd = stab[(stab.window == "full sample") & (stab.series == "spread")].iloc[0]
    h["adf_p_full_usd"], h["kpss_p_full_usd"] = float(full_usd.adf_p), float(full_usd.kpss_p)

    eg = []
    for label, a, b in windows:
        e = st.engle_granger(px, a, b)
        if e:
            eg.append({"window": label, **e})
    h["engle_granger"] = eg
    print(pd.DataFrame(eg).round(3).to_string())

    # ------------------------------------------------------------------ 2
    print("\n=== 2. did the tether move? ===")
    brk = {}
    for col in ("log_ratio", "spread"):
        b = st.sup_f_bootstrap(sf[col], n_boot=N_BOOT_BREAK)
        c = st.cusum_mean(sf[col])
        brk[col] = {"sup_f": b["sup_f"], "p": b["p"], "date": _d(b["break_date"]),
                    "null_phi": b["null_phi"], "null_q95": b["null_q95"],
                    "null_q99": b["null_q99"], "cusum": c["stat"], "cusum_peak": _d(c["peak_date"]),
                    "cusum_reject": c["reject5"]}
        print(f"  {col:9s}: sup-F {b['sup_f']:.0f} at {_d(b['break_date'])}  bootstrap p "
              f"{b['p']:.3f} (null phi {b['null_phi']:.3f}, null 95%/99% {b['null_q95']:.0f}/"
              f"{b['null_q99']:.0f}; Andrews 5% asymptotic {st.ANDREWS_CV_15['5%']})  "
              f"CUSUM {c['stat']:.2f} peak {_d(c['peak_date'])}")
    h["breaks"] = brk
    h["break_date"] = brk["log_ratio"]["date"]
    h["break_p"] = brk["log_ratio"]["p"]
    h["sup_f"] = brk["log_ratio"]["sup_f"]
    h["break_date_usd"] = brk["spread"]["date"]
    h["break_p_usd"] = brk["spread"]["p"]

    bp = {}
    for col in ("log_ratio", "spread"):
        r = st.bai_perron(sf[col], max_breaks=5, min_seg=24)
        bp[col] = {"table": r["table"].reset_index().to_dict("records"), "m_lwz": r["m_lwz"],
                   "m_bic": r["m_bic"],
                   "segments": [{**s, "start": _d(s["start"]), "end": _d(s["end"])}
                                for s in r["segments"]]}
        print(f"  Bai–Perron {col}: LWZ picks {r['m_lwz']} breaks "
              f"({', '.join(_d(d) for d in r['break_dates'])}); BIC picks {r['m_bic']}")
    h["bai_perron"] = bp

    hl1 = st.halflife_by_regime(sf["log_ratio"], [bdate])
    hl1_usd = st.halflife_by_regime(sf["spread"], [bdate])
    print(hl1.drop(columns=["start", "end"]).round(3).to_string())
    h["hl_regimes"] = [{**r, "start": _d(r["start"]), "end": _d(r["end"])}
                       for r in hl1.to_dict("records")]
    h["hl_regimes_usd"] = [{**r, "start": _d(r["start"]), "end": _d(r["end"])}
                           for r in hl1_usd.to_dict("records")]
    h["hl_pre"], h["hl_post"] = float(hl1.halflife.iloc[0]), float(hl1.halflife.iloc[1])
    h["hl_pre_lo"], h["hl_pre_hi"] = float(hl1.halflife_lo.iloc[0]), float(hl1.halflife_hi.iloc[0])
    h["hl_post_lo"], h["hl_post_hi"] = float(hl1.halflife_lo.iloc[1]), float(hl1.halflife_hi.iloc[1])
    h["adf_p_pre"], h["adf_p_post"] = float(hl1.adf_p.iloc[0]), float(hl1.adf_p.iloc[1])
    h["adf_p_pre_usd"] = float(hl1_usd.adf_p.iloc[0])
    h["mean_pre_usd"] = float(sf["spread"][sf.index < bdate].mean())
    h["mean_post_usd"] = float(sf["spread"][sf.index >= bdate].mean())
    h["mean_pre_lr"] = float(sf["log_ratio"][sf.index < bdate].mean())
    h["mean_post_lr"] = float(sf["log_ratio"][sf.index >= bdate].mean())
    segs = st.bai_perron(sf["log_ratio"])["break_dates"]
    hlbp = st.halflife_by_regime(sf["log_ratio"], segs)
    h["hl_bp"] = [{**r, "start": _d(r["start"]), "end": _d(r["end"])}
                  for r in hlbp.to_dict("records")]

    # ------------------------------------------------------------------ 3
    print("\n=== 3. the trader's experience ===")
    rules = {"expanding": st.zscore_realtime(sf["log_ratio"]),
             "rolling 60m": st.zscore_realtime(sf["log_ratio"], window=60),
             "frozen 1987–2009": st.zscore_frozen(sf["log_ratio"])}
    perf_rows, books, eps = [], {}, {}
    for name, z in rules.items():
        pos = st.positions(z)
        bk = st.book(px, pos)
        books[name] = bk
        g, n = st.perf(bk["gross"]), st.perf(bk["net"])
        uw = st.underwater(bk["net"], "2010-01-31" if name.startswith("frozen") else None)
        e = st.episodes(bk)
        eps[name] = e
        perf_rows.append({"rule": name, "ann_gross": g["ann"], "sr_gross": g["sharpe"],
                          "t_gross": g["t_hac"], "sr_gross_lo": g["sr_lo"],
                          "sr_gross_hi": g["sr_hi"], "ann_net": n["ann"],
                          "sr_net": n["sharpe"], "t_net": n["t_hac"], "sr_net_lo": n["sr_lo"],
                          "sr_net_hi": n["sr_hi"], "maxdd": n["maxdd"],
                          "months_in": int((bk["pos"].shift(1).fillna(0) != 0).sum()),
                          "trades": int(len(e)),
                          "uw_months": uw["months"], "uw_peak": _d(uw["peak"]),
                          "uw_recovered": _d(uw["recovered"]),
                          "break_even_bps": st.break_even_cost(px, st.positions(z))})
        print(f"  {name:17s} gross {g['ann']:+.2%}/yr SR {g['sharpe']:+.2f} t {g['t_hac']:+.2f}"
              f" | net {n['ann']:+.2%}/yr SR {n['sharpe']:+.2f} t {n['t_hac']:+.2f} | maxDD "
              f"{n['maxdd']:.1%} | longest underwater {uw['months']}m from {_d(uw['peak'])}")
    h["rules"] = perf_rows
    h["episodes"] = {k: [{**r, "entry": _d(r["entry"]), "exit": _d(r["exit"]),
                          "mae_date": _d(r["mae_date"])} for r in v.to_dict("records")]
                     for k, v in eps.items()}
    ex = perf_rows[0]
    h["t_gross"], h["t_net"] = ex["t_gross"], ex["t_net"]
    h["ann_net"], h["sr_net"] = ex["ann_net"], ex["sr_net"]
    h["ann_gross"], h["sr_gross"] = ex["ann_gross"], ex["sr_gross"]
    h["sr_gross_lo"], h["sr_gross_hi"] = ex["sr_gross_lo"], ex["sr_gross_hi"]
    h["t_gross_roll"] = perf_rows[1]["t_gross"]
    h["t_net_roll"] = perf_rows[1]["t_net"]

    fz = eps["frozen 1987–2009"]
    first = fz.iloc[0]
    h["frozen_entry"] = _d(first["entry"])
    h["frozen_side"] = first["side"]
    h["frozen_mae"] = float(first["mae"])
    h["frozen_mae_usd"] = float(first["mae_usd"])
    h["frozen_mae_date"] = _d(first["mae_date"])
    h["frozen_closed"] = bool(first["closed"])
    h["frozen_exit"] = _d(first["exit"])
    fuw = st.underwater(books["frozen 1987–2009"]["net"], "2010-01-31")
    h["frozen_underwater_months"] = int(fuw["months"])
    h["frozen_recovered"] = fuw["recovered"] is not None
    h["frozen_net_total"] = float((1 + books["frozen 1987–2009"]["net"]).prod() - 1)
    # the expanding trader's worst episode
    ee = eps["expanding"]
    worst = ee.loc[ee["mae"].idxmin()]
    h["exp_worst"] = {"entry": _d(worst["entry"]), "exit": _d(worst["exit"]),
                      "months": int(worst["months"]), "mae": float(worst["mae"]),
                      "mae_usd": float(worst["mae_usd"]), "mae_date": _d(worst["mae_date"]),
                      "net": float(worst["net_return"])}
    # sub-periods for the expanding fader (gross and net)
    sub = []
    for label, a, b in [("1987–2009", None, "2009-12-31"), ("2010–2014", "2010-01-31", "2014-12-31"),
                        ("2015–2020", "2015-01-31", None)]:
        bk = st._window(books["expanding"], a, b)
        g, n = st.perf(bk["gross"], n_boot=500), st.perf(bk["net"], n_boot=500)
        sub.append({"period": label, "months": int(len(bk)), "ann_gross": g["ann"],
                    "t_gross": g["t_hac"], "ann_net": n["ann"], "t_net": n["t_hac"]})
    h["sub_periods"] = sub
    cs = st.cost_sweep(px, st.positions(rules["expanding"]))
    h["cost_sweep"] = cs.reset_index().to_dict("records")
    print(cs.round(3).to_string())

    # ------------------------------------------------------------------ 4
    print("\n=== 4. power at n = 393 (synthetic, calibrated to the pre-break regime) ===")
    pw = st.power_curve(strengths=(0.0, 0.1, 0.25, 0.5, 1.0), n_reps=N_REPS_POWER)
    pwb = st.power_curve(strengths=(0.0, 0.1, 0.25, 0.5, 1.0), n_reps=N_REPS_POWER,
                         break_size=h["mean_post_lr"] - h["mean_pre_lr"])
    pw["adf_reject_with_break"] = pwb["adf_reject"]
    pw["kpss_reject_with_break"] = pwb["kpss_reject"]
    print(pw.round(3).to_string())
    h["power"] = pw.reset_index().to_dict("records")
    h["planted_break_log"] = float(h["mean_post_lr"] - h["mean_pre_lr"])
    tp = st.synthetic_trade_power(strengths=(0.0, 0.5, 1.0), n_reps=40)
    print(tp.round(3).to_string())
    h["trade_power"] = tp.reset_index().to_dict("records")
    # a break-free planted world through the same bootstrap: does the break test stay quiet?
    sp, _ = data.synthetic_spread(signal_strength=1.0, seed=7)
    q = st.sup_f_bootstrap(np.log(sp["brent"] / sp["wti"]), n_boot=199)
    spb, trb = data.synthetic_spread(signal_strength=1.0, seed=7,
                                     break_size=h["planted_break_log"])
    qb = st.sup_f_bootstrap(np.log(spb["brent"] / spb["wti"]), n_boot=199)
    h["placebo_break_p"] = q["p"]
    h["planted_break_p"] = qb["p"]
    h["planted_break_found"] = _d(qb["break_date"])
    h["planted_break_true"] = _d(trb["break_date"])
    print(f"  break test on a break-free OU: p = {q['p']:.3f}; with the real-size break "
          f"planted at {_d(trb['break_date'])}: p = {qb['p']:.3f}, found at "
          f"{_d(qb['break_date'])}")

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = round(time.time() - t0, 1)
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    chk = h["avg_check"]

    def fp(p):
        return "≤ 0.01" if p <= 0.01 else ("≥ 0.10" if p >= 0.10 else f"{p:.3f}")

    st_rows = "\n".join(
        f"| {r['window']} | {'$ spread' if r['series'] == 'spread' else 'log ratio'} | "
        f"{r['n']} | {r['adf_stat']:.2f} | {r['adf_p']:.3f} | {r['kpss_stat']:.2f} | "
        f"{fp(r['kpss_p'])} | {r['reading']} |" for r in h["stationarity"])
    eg_rows = "\n".join(
        f"| {r['window']} | {r['n']} | {r['beta']:.3f} | {r['stat']:.2f} | {r['p']:.4f} |"
        for r in h["engle_granger"])
    b = h["breaks"]
    br_rows = "\n".join(
        f"| {'log ratio' if k == 'log_ratio' else '$ spread'} | {v_['date']} | "
        f"{v_['sup_f']:.0f} | {v_['null_phi']:.3f} | {v_['null_q95']:.0f} / {v_['null_q99']:.0f} | "
        f"**{v_['p']:.3f}** | {v_['cusum']:.2f} ({v_['cusum_peak']}) |"
        for k, v_ in b.items())
    bp_lr = h["bai_perron"]["log_ratio"]
    bp_rows = "\n".join(
        f"| {r['m']} | {r['ssr']:.3f} | {r['lwz']:.3f} | {r['bic']:.3f} | {r['breaks'] or '—'} |"
        for r in bp_lr["table"])
    seg_rows = "\n".join(
        f"| {r['start']} → {r['end']} | {r['n']} | {r['mean']:+.3f} | {r['sd']:.3f} | "
        f"{r['phi']:.3f} | {_hl(r['halflife'])} ({_hl(r['halflife_lo'])}–{_hl(r['halflife_hi'])}) | "
        f"{r['adf_p']:.3f} |" for r in h["hl_bp"])
    hl_rows = "\n".join(
        f"| {lab} | {r['start']} → {r['end']} | {r['n']} | {r['mean']:+.3f} | {r['phi']:.3f} | "
        f"**{_hl(r['halflife'])}** ({_hl(r['halflife_lo'])}–{_hl(r['halflife_hi'])}) | {r['adf_p']:.3f} |"
        for lab, rr in (("log ratio", h["hl_regimes"]), ("$ spread", h["hl_regimes_usd"]))
        for r in rr)
    rule_rows = "\n".join(
        f"| {r['rule']} | {r['ann_gross']:+.2%} | {r['sr_gross']:+.2f} "
        f"[{r['sr_gross_lo']:+.2f}, {r['sr_gross_hi']:+.2f}] | {r['t_gross']:+.2f} | "
        f"{r['ann_net']:+.2%} | {r['sr_net']:+.2f} | **{r['t_net']:+.2f}** | {r['maxdd']:.1%} | "
        f"{r['months_in']} | {r['trades']} | {r['uw_months']} (from {r['uw_peak']}, "
        f"{'recovered ' + r['uw_recovered'] if r['uw_recovered'] != '—' else 'never recovered'}) | "
        f"{_be(r['break_even_bps'])} |" for r in h["rules"])
    ep_rows = "\n".join(
        f"| {k} | {r['entry']} | {r['exit']}{'' if r['closed'] else ' (open)'} | {r['side']} | "
        f"{r['months']} | {r['net_return']:+.1%} | **{r['mae']:.1%}** ({r['mae_date']}) | "
        f"{r['usd_pnl']:+.2f} | {r['mae_usd']:+.2f} |"
        for k in ("expanding", "frozen 1987–2009") for r in h["episodes"][k])
    sub_rows = "\n".join(
        f"| {r['period']} | {r['months']} | {r['ann_gross']:+.2%} | {r['t_gross']:+.2f} | "
        f"{r['ann_net']:+.2%} | {r['t_net']:+.2f} |" for r in h["sub_periods"])
    cs_rows = "\n".join(
        f"| {r['cost_bps']:.0f} | {r['ann']:+.2%} | {r['sharpe']:+.2f} | {r['t_hac']:+.2f} |"
        for r in h["cost_sweep"])
    pw_rows = "\n".join(
        f"| {r['signal_strength']:.2f} | {_hl(r['halflife'])} | {r['adf_reject']:.0%} | "
        f"{r['kpss_reject']:.0%} | {r['adf_reject_with_break']:.0%} | "
        f"{r['kpss_reject_with_break']:.0%} |" for r in h["power"])
    tp_rows = "\n".join(
        f"| {r['signal_strength']:.1f} | {r['median_t']:+.2f} | {r['share_t_ge_2']:.0%} |"
        for r in h["trade_power"])
    ew = h["exp_worst"]
    fuw_txt = ("still underwater when the tape ends" if not h["frozen_recovered"]
               else "recovered")
    return f"""# Results — Study 1023 (The Cushing Glut) on the real monthly tape

*Generated by [`examples/verify.py`](../examples/verify.py). As-of
**{h['as_of']}** (last full month on the tape); fingerprint `{h['fingerprint']}`.*

## 0. Data provenance — read this before any number

| | |
|---|---|
| Source | `arch/data/crude/crude.csv.gz` inside the **{h['package']}** wheel (FRED-sourced), read via [`quantlab/bundled.py`](../../../quantlab/bundled.py) |
| SHA-256 pin | `{h['sha256']}` — the loader refuses any other bytes |
| Content fingerprint | `{h['fingerprint']}` (`bundled.fingerprint` of the Brent/WTI frame) |
| Sample | {h['n_months']} months, {h['first']} → {h['last']} |
| Units | **nominal** US$/bbl, **spot** |
| Construction | **calendar-month averages** of daily prints — checked against `arch`'s daily FRED WTI tape over {chk['n_months']} full months: median gap to the monthly *mean* ${chk['median_abs_gap_avg']:.4f} ({chk['share_within_1c_avg']:.0%} within 1¢), to the month-*end* print ${chk['median_abs_gap_end']:.2f} |

> ⚠️ **Spot, monthly-averaged prices are not investable.** A real Brent–WTI trade is two
> futures legs (ICE Brent, NYMEX WTI) rolled every month, and it earns or pays the **roll
> yield** of each — which diverged most exactly when Cushing was full. It would also trade at
> a price, not at a month's average. Every P&L below is therefore an **upper bound**, booked
> with a one-way cost per leg and a monthly roll *toll* (bid–ask) but no roll *yield*.

## 1. Is the spread stationary?

ADF (null: unit root, AIC lags ≤ 12) and KPSS (null: stationary). KPSS p-values come from a
lookup table clipped to [0.01, 0.10].

| Window | Series | n | ADF stat | ADF p | KPSS stat | KPSS p | Joint reading |
|---|---|--:|--:|--:|--:|--:|---|
{st_rows}

Engle–Granger on `ln Brent = a + b·ln WTI + u` (MacKinnon p-values):

| Window | n | slope b | EG stat | p |
|---|--:|--:|--:|--:|
{eg_rows}

Over the full sample the log ratio is **not** stationary by either test (ADF p =
{h['adf_p_full']:.2f}; KPSS rejects). Yet Engle–Granger rejects no-cointegration on the full
sample. It is free to pick a slope b ≠ 1, and a slope above one can recast part of the
post-2010 level shift as a price-level effect (prices were high when the discount was wide).
The claim is about the 1:1 spread — "same commodity" — and the 1:1 spread did not stay put.
Note also the 2010–2014 window alone: Engle–Granger cannot reject no-cointegration there.

## 2. Did the tether move? — data-driven break search

Sup-F for one shift in the mean (Andrews 1993, 15% trimming). The asymptotic 5% critical value
({st.ANDREWS_CV_15['5%']}) assumes weakly dependent errors; with a half-life of months it is
far too lenient, so the p-value comes from {N_BOOT_BREAK} AR(1) bootstrap replications of a
**break-free** spread with the full-sample persistence (break included, hence conservative).

| Series | Break (first month of new regime) | sup-F | null φ | null 95% / 99% | bootstrap p | HAC-CUSUM (peak) |
|---|---|--:|--:|--:|--:|--:|
{br_rows}

The CUSUM 5% critical value is {st.CUSUM_CV_5}. The two series pick neighbouring months
({h['break_date']} for the log ratio, {h['break_date_usd']} for dollars); in dollars the
spread's average moved from **{h['mean_pre_usd']:+.2f}** to **{h['mean_post_usd']:+.2f} $/bbl**.

**Bai–Perron (global SSR, min segment 24 months), log ratio:**

| Breaks m | SSR | LWZ | BIC | Dates |
|--:|--:|--:|--:|---|
{bp_rows}

LWZ picks **{bp_lr['m_lwz']}** breaks. With errors this persistent every information criterion
over-counts, so the count is descriptive; what survives every method is the dominant shift at
the turn of 2010/2011. Regimes and their half-lives at the LWZ choice:

| Regime | n | mean log ratio | s.d. | φ | half-life, months (95%) | ADF p |
|---|--:|--:|--:|--:|--:|--:|
{seg_rows}

**Half-life before and after the single sup-F break:**

| Series | Regime | n | mean | φ | half-life, months (95%) | ADF p |
|---|---|--:|--:|--:|--:|--:|
{hl_rows}

Before the break the ratio reverted fast (half-life {h['hl_pre']:.1f} months); after it,
{h['hl_post']:.1f} months around a new, higher mean. OLS φ is biased down in samples this
short (Kendall 1954), so both half-lives lean fast, and the post-break band is wide.

## 3. The trader's experience

The rule: z-score of the log ratio from **past data only** (expanding window from 36 months, or
rolling 60 months); short the spread above +{st.ENTRY_Z}σ, long below −{st.ENTRY_Z}σ, flat inside
±{st.EXIT_Z}σ. Position decided on month *t*'s prices earns month *t+1* (**one lag**). Book: $1
long Brent / $1 short WTI per unit; costs **{st.COST_BPS:.0f} bp one-way per leg** × NAV traded
plus a **{st.ROLL_BPS:.0f} bp per leg per month** roll toll. Sharpe CIs: circular block bootstrap,
12-month blocks. The frozen trader estimated the mean and s.d. on 1987–2009 and never updated.

| Rule | Gross /yr | Gross SR [95% CI] | Gross HAC t | Net /yr | Net SR | Net HAC t | Max DD | Months in | Trades | Longest underwater (months) | Break-even bp |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
{rule_rows}

Every trade, with its **maximum adverse excursion** (MAE: worst cumulative net return while
open, and the same bet in $/bbl per barrel pair):

| Rule | Entry | Exit | Side | Months | Net | MAE (date) | $/bbl P&L | MAE $/bbl |
|---|---|---|---|--:|--:|--:|--:|--:|
{ep_rows}

**When the regime broke.** The expanding-window trader was already short the spread from
{ew['entry']} and held for {ew['months']} months; at the worst point ({ew['mae_date']}) the trade
was down **{ew['mae']:.1%}** of NAV ({ew['mae_usd']:+.2f} $/bbl). The frozen pre-2010 trader opened a
{h['frozen_side']} position in **{h['frozen_entry']}**, hit an MAE of **{h['frozen_mae']:.1%}**
({h['frozen_mae_usd']:+.2f} $/bbl) and was underwater for **{h['frozen_underwater_months']}
months** — {fuw_txt} ({h['frozen_net_total']:+.1%} cumulative net since 2010).

Expanding-window fader by sub-period:

| Period | Months | Gross /yr | Gross HAC t | Net /yr | Net HAC t |
|---|--:|--:|--:|--:|--:|
{sub_rows}

Cost sweep (expanding window; roll toll fixed at {st.ROLL_BPS:.0f} bp):

| One-way cost per leg (bp) | Net /yr | Net SR | Net HAC t |
|--:|--:|--:|--:|
{cs_rows}

The rolling-60-month variant reads better (gross t {h['t_gross_roll']:+.2f}, net
{h['t_net_roll']:+.2f}) because it forgets the old mean faster. It is shown, not headlined: it is
one of several window lengths a reader could try, and picking the best one after the fact is
exactly the selection the desk does not certify.

## 4. Power — what n = 393 months can see

Synthetic OU log ratios calibrated to the pre-break regime (half-life 3 months at strength 1,
residual s.d. 0.027), {N_REPS_POWER} paths per row, ADF/KPSS at 5%. Right-hand columns plant
the real tape's break ({h['planted_break_log']:+.3f} in logs) at 72% of the sample.

| signal_strength | half-life (months) | ADF rejects | KPSS rejects | ADF rejects, with break | KPSS rejects, with break |
|--:|--:|--:|--:|--:|--:|
{pw_rows}

With a three-month half-life and no break, ADF finds the tether essentially every time, and
still does most of the time at a one-year half-life; at thirty months it is down to about a
quarter. So at n = 393 the tests are not short of power *for a spread like the pre-break one*
— the full-sample non-rejection is not a power failure. Plant one level shift of the size the
real tape carries and ADF loses most of its power even on a fast-reverting spread, while KPSS
rejects almost every time — exactly the real tape's pattern. A mean break looks like a unit
root to a test that assumes one constant mean (Perron 1989); the faster the reversion, the
larger the break is relative to the spread's own noise, which is why ADF power with a break is
not monotone in the reversion speed.

Real-time fader on the synthetic worlds (gross HAC t, 40 paths, no costs):

| signal_strength | median t | share with t ≥ 2 |
|--:|--:|--:|
{tp_rows}

Break test on a break-free planted OU: bootstrap p = {h['placebo_break_p']:.3f} (quiet). With the
real-size break planted at {h['planted_break_true']}: p = {h['planted_break_p']:.3f}, located at
{h['planted_break_found']}.

## Caveats

- **Spot, not futures — an upper bound.** No roll yield is in these numbers. During the glut
  the WTI curve sat in steep contango, so a long-WTI leg rolled at a loss: the real trader who
  faded the spread (short Brent, long WTI) bled roll on top of the spot loss shown here.
- **Monthly averages.** The tape is calendar-month means of daily prints. Nobody trades at the
  average; averaging also smooths the series and adds an MA(1) to its changes (Working 1960),
  which flatters a monthly mean-reversion rule slightly.
- **n ≈ 390 months, one break.** All break inference rests on essentially one episode. The
  bootstrap null is an AR(1); a richer null (fat tails, regime-switching volatility) would widen
  it. The sub-period windows in §1 are descriptive, and only the break date comes from data.
- **The pre-2010 calibration date is the claim's, not ours,** and it is close to the detected
  break, which is the point: a trader who learned the tether on the data available in 2009
  had no way to know it was about to move.
- **One rule family, one threshold pair** (±{st.ENTRY_Z}σ in, ±{st.EXIT_Z}σ out), fixed before
  the run. The rolling-window variant is reported for transparency, not as a rescue.
- **Nominal dollars.** A $10 discount meant more in 1990 than in 2012; the log ratio is the
  scale-free reading and is the one the verdict uses.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py). It reads the log ratio throughout.

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1023-brent-wti-spread](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    keys = ("as_of", "fingerprint", "n_months", "break_date", "break_p", "sup_f",
            "mean_pre_usd", "mean_post_usd", "hl_pre", "hl_post", "adf_p_full", "kpss_p_full",
            "adf_p_pre", "adf_p_pre_usd", "t_gross", "t_net", "ann_net", "sr_net",
            "t_gross_roll", "t_net_roll", "frozen_entry", "frozen_mae", "frozen_mae_usd",
            "frozen_underwater_months", "frozen_recovered", "runtime_s")
    head = {k: h[k] for k in keys}
    head["signal"] = h["_verdict"]["signal"]
    head["trad"] = h["_verdict"]["trad"]
    print("##HEADLINE## " + json.dumps(head, default=str))


if __name__ == "__main__":
    main()
