"""Real-tape verification — Study 1025 (Don't Look). Regenerates docs/results.md.

Rebuilds Benartzi & Thaler's (1995) myopic-loss-aversion calculation on frozen,
SHA-pinned tapes (``quantlab.bundled``): how often each asset shows a loss at each evaluation
horizon, how a Tversky-Kahneman (1992) investor values stocks against bonds and bills at each
horizon, where the preference flips (the "equilibrium" horizon) and how sure we can be of it
(circular block bootstrap of the joint monthly returns), how the answer moves with λ, α, real
vs nominal returns and the pre/post-1970 split — and, for the Tradability axis, what the
myopic behaviour "don't look" is meant to prevent actually costs: an investor who flees to bills
after seeing a loss and returns after a gain, at every checking clock from daily to yearly.

    python studies/1025-myopic-loss-aversion/examples/verify.py

Offline; finishes in well under three minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dontlook import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

N_BOOT = 1000          # headline bootstrap (stocks/bonds/bills, full sample)
N_BOOT_ROB = 400       # robustness bootstraps (sub-periods, real, block length)
SHOW_M = (1, 3, 6, 12, 24, 36, 60, 120)
COLS = ["stock", "bond", "bill"]


def fm(m: float) -> str:
    """Months, readable; inf → 'never ≤ 10y'."""
    if not np.isfinite(m):
        return "never ≤ 10y"
    return f"{m:.1f} mo" if m < 12 else f"{m:.0f} mo ({m / 12:.1f}y)"


def report() -> dict:
    t0 = time.time()
    m = data.load_monthly()
    X = m[COLS]
    d = data.load_daily()
    h: dict = {"as_of": data.AS_OF,
               "fp_monthly": data.fingerprint(m), "fp_daily": data.fingerprint(d),
               "n_months": int(len(X)), "first_month": str(X.index[0].date()),
               "n_days": int(len(d)), "first_day": str(d.index[0].date()),
               "provenance": data.provenance()}
    print(f"as-of {data.AS_OF}   monthly fp {h['fp_monthly']} ({len(X)} months)   "
          f"daily fp {h['fp_daily']} ({len(d)} days)")

    # ------------------------------------------------------------------ 1
    print("\n=== 1. how often do you SEE a loss? ===")
    L = st.loss_probability_curve(X, st.HORIZONS_M)
    boot = st.bootstrap_curves(X, st.HORIZONS_M, COLS, n_boot=N_BOOT, block=st.BLOCK_M)
    band = st.loss_band(boot)
    rows = []
    for hz in SHOW_M:
        r = {"h": hz}
        for c in COLS:
            r[c] = float(L.loc[hz, c])
            r[c + "_lo"] = float(band[c].loc[hz, "lo"])
            r[c + "_hi"] = float(band[c].loc[hz, "hi"])
        rows.append(r)
    h["loss_monthly"] = rows
    Ld = st.loss_probability_curve(d[["stock"]], st.HORIZONS_D)
    bd = st.bootstrap_curves(d[["stock", "bill"]], st.HORIZONS_D, ["stock", "bill"],
                             n_boot=300, block=st.BLOCK_D)
    bandd = st.loss_band(bd)
    h["loss_daily"] = [{"h": int(k), "stock": float(Ld.loc[k, "stock"]),
                        "lo": float(bandd["stock"].loc[k, "lo"]),
                        "hi": float(bandd["stock"].loc[k, "hi"])} for k in st.HORIZONS_D]
    h["loss_1d"] = float(Ld.loc[1, "stock"])
    h["loss_1m"] = float(L.loc[1, "stock"])
    h["loss_12m"] = float(L.loc[12, "stock"])
    h["loss_120m"] = float(L.loc[120, "stock"])
    print(L.loc[list(SHOW_M)].round(3).to_string())
    print(Ld.round(3).T.to_string())

    # ------------------------------------------------------------------ 2
    print("\n=== 2. prospect-theory value by evaluation horizon ===")
    P = st.pt_curve(X, st.HORIZONS_M)
    sb = st.summarise_gap(P, boot, "stock", "bond")
    sbill = st.summarise_gap(P, boot, "stock", "bill")
    h["be"], h["be_lo"], h["be_hi"] = sb["break_even"], sb["be_lo"], sb["be_hi"]
    h["be_median"], h["be_share_finite"] = sb["be_median"], sb["share_finite"]
    h["be_share_6_24"] = sb["share_contains_12"]
    h["be_bills"], h["be_bills_lo"], h["be_bills_hi"] = (sbill["break_even"], sbill["be_lo"],
                                                          sbill["be_hi"])
    h["be_bills_share_finite"] = sbill["share_finite"]
    tb, tbl = sb["table"], sbill["table"]
    h["gap_short"], h["p_short"] = float(tb.loc[1, "gap"]), float(tb.loc[1, "p"])
    h["gap_long"], h["p_long"] = float(tb.loc[120, "gap"]), float(tb.loc[120, "p"])
    h["gap_bills_short"], h["p_bills_short"] = float(tbl.loc[1, "gap"]), float(tbl.loc[1, "p"])
    h["gap_bills_long"], h["p_bills_long"] = float(tbl.loc[120, "gap"]), float(tbl.loc[120, "p"])
    h["gap_rows"] = [{"h": int(k), "pt_stock": float(P.loc[k, "stock"]),
                      "pt_bond": float(P.loc[k, "bond"]), "pt_bill": float(P.loc[k, "bill"]),
                      "gap_b": float(tb.loc[k, "gap"]), "lo_b": float(tb.loc[k, "gap_lo"]),
                      "hi_b": float(tb.loc[k, "gap_hi"]), "p_b": float(tb.loc[k, "p"]),
                      "gap_f": float(tbl.loc[k, "gap"]), "p_f": float(tbl.loc[k, "p"])}
                     for k in SHOW_M]
    h["curve"] = {"h": list(st.HORIZONS_M),
                  "gap_b": tb["gap"].tolist(), "lo_b": tb["gap_lo"].tolist(),
                  "hi_b": tb["gap_hi"].tolist(), "gap_f": tbl["gap"].tolist(),
                  "lo_f": tbl["gap_lo"].tolist(), "hi_f": tbl["gap_hi"].tolist()}
    print(tb.loc[list(SHOW_M)].round(4).to_string())
    print(f"  break-even vs bonds {fm(h['be'])}  95% CI [{fm(h['be_lo'])}, {fm(h['be_hi'])}]"
          f"  finite in {h['be_share_finite']:.0%} of resamples")
    print(f"  break-even vs bills {fm(h['be_bills'])}  95% CI [{fm(h['be_bills_lo'])}, "
          f"{fm(h['be_bills_hi'])}]")

    est = []
    for meth in ("overlapping", "nonoverlapping", "iid"):
        for label, df in (("1926-2018", X), ("1926-1990 (B&T window)",
                                             X[X.index < "1991-01-01"])):
            Pm = st.pt_curve(df, st.HORIZONS_M, method=meth)
            est.append({"method": meth, "window": label,
                        "be_bonds": st.break_even(st.HORIZONS_M,
                                                  (Pm["stock"] - Pm["bond"]).to_numpy()),
                        "be_bills": st.break_even(st.HORIZONS_M,
                                                  (Pm["stock"] - Pm["bill"]).to_numpy())})
    h["estimators"] = est
    h["be_iid"] = [e["be_bonds"] for e in est
                   if e["method"] == "iid" and e["window"] == "1926-2018"][0]
    h["be_bt"] = [e["be_bonds"] for e in est
                  if e["method"] == "overlapping" and e["window"].startswith("1926-1990")][0]
    h["be_bt_iid"] = [e["be_bonds"] for e in est
                      if e["method"] == "iid" and e["window"].startswith("1926-1990")][0]
    print(pd.DataFrame(est).to_string(index=False))
    vr = []
    for c in ("stock", "bond"):
        l1 = np.log1p(X[c].to_numpy()).var()
        vr.append({"asset": c, "vr": [float(np.log1p(st.horizon_returns(X[c].to_numpy(), k))
                                            .var() / (k * l1)) for k in (6, 12, 24, 60, 120)]})
    h["variance_ratios"] = vr
    print(vr)

    # daily end of the curve: stocks (price-only) vs bills
    Pd = st.pt_curve(d, st.HORIZONS_D, cols=["stock", "bill"])
    sd_ = st.summarise_gap(Pd, bd, "stock", "bill")
    h["daily_gap"] = [{"h": int(k), "gap": float(sd_["table"].loc[k, "gap"]),
                       "lo": float(sd_["table"].loc[k, "gap_lo"]),
                       "hi": float(sd_["table"].loc[k, "gap_hi"]),
                       "p": float(sd_["table"].loc[k, "p"])} for k in st.HORIZONS_D]
    print(sd_["table"].round(4).to_string())

    # ------------------------------------------------------------------ 3
    print("\n=== 3. robustness ===")
    h["lambda_12"] = st.break_even_lambda(X, 12)
    h["lambda_12_bills"] = st.break_even_lambda(X, 12, b="bill")
    h["lambda_1"] = st.break_even_lambda(X, 1)
    print(f"  λ for indifference at 12 months: {h['lambda_12']:.2f} (bonds), "
          f"{h['lambda_12_bills']:.2f} (bills); at 1 month {h['lambda_1']:.2f}")
    sw_w = st.param_sweep(X, weighting=True)
    sw_n = st.param_sweep(X, weighting=False)
    h["sweep_w"] = {"alphas": [float(a) for a in sw_w.columns],
                    "rows": [{"lambda": float(l), "be": [float(v) for v in sw_w.loc[l]]}
                             for l in sw_w.index]}
    h["sweep_n"] = {"alphas": [float(a) for a in sw_n.columns],
                    "rows": [{"lambda": float(l), "be": [float(v) for v in sw_n.loc[l]]}
                             for l in sw_n.index]}
    h["lam_effect"] = float(sw_w.loc[3.0, 0.88] - sw_w.loc[2.0, 0.88])
    h["alpha_effect"] = float(abs(sw_w.loc[2.25, 0.70] - sw_w.loc[2.25, 1.0]))
    print(sw_w.round(1).to_string())
    print(sw_n.round(1).to_string())

    regimes = []
    R = data.to_real(m)
    Xn57 = X[X.index >= R.index[0]]
    cases = (("pre-1970 (nominal)", X[X.index < data.SPLIT]),
             ("post-1970 (nominal)", X[X.index >= data.SPLIT]),
             ("1957-2018 nominal", Xn57),
             ("1957-2018 real (core CPI)", R))
    for label, df in cases:
        Pr = st.pt_curve(df, st.HORIZONS_M)
        br = st.bootstrap_curves(df, st.HORIZONS_M, COLS, n_boot=N_BOOT_ROB, block=st.BLOCK_M,
                                 seed=1026)
        a = st.summarise_gap(Pr, br, "stock", "bond")
        b = st.summarise_gap(Pr, br, "stock", "bill")
        Pi = st.pt_curve(df, st.HORIZONS_M, method="iid")
        regimes.append({"case": label, "n": int(len(df)),
                        "be_b": a["break_even"], "lo_b": a["be_lo"], "hi_b": a["be_hi"],
                        "be_f": b["break_even"], "lo_f": b["be_lo"], "hi_f": b["be_hi"],
                        "be_iid": st.break_even(st.HORIZONS_M,
                                                (Pi["stock"] - Pi["bond"]).to_numpy()),
                        "loss12": float(st.loss_probability_curve(df, [12]).loc[12, "stock"])})
        print(f"  {label:28s} bonds {fm(a['break_even'])} [{fm(a['be_lo'])}, "
              f"{fm(a['be_hi'])}]  bills {fm(b['break_even'])}")
    h["regimes"] = regimes
    h["be_pre"], h["be_post"] = regimes[0]["be_b"], regimes[1]["be_b"]
    h["be_real"] = regimes[3]["be_b"]

    mats = []
    for mat in (5.0, 10.0, 20.0, 30.0):
        mm = data.load_monthly(maturity=mat)[COLS]
        Pm = st.pt_curve(mm, st.HORIZONS_M)
        mats.append({"maturity": mat, "bond_vol": float(mm["bond"].std() * np.sqrt(12)),
                     "bond_cagr": float((1 + mm["bond"]).prod() ** (12 / len(mm)) - 1),
                     "be": st.break_even(st.HORIZONS_M, (Pm["stock"] - Pm["bond"]).to_numpy())})
    h["maturities"] = mats
    blocks = []
    for bl in (6, 12, 24, 60):
        bb = st.bootstrap_curves(X, st.HORIZONS_M, COLS, n_boot=N_BOOT_ROB, block=bl, seed=1027)
        s = st.summarise_gap(P, bb, "stock", "bond")
        blocks.append({"block": bl, "lo": s["be_lo"], "hi": s["be_hi"],
                       "share_finite": s["share_finite"]})
    h["blocks"] = blocks
    print(pd.DataFrame(mats).round(3).to_string(index=False))
    print(pd.DataFrame(blocks).round(2).to_string(index=False))

    # ------------------------------------------------------------------ 4
    print("\n=== 4. the discipline question: what does looking cost? ===")
    dd = st.discipline_table(d, ["D", "W", "M"], 252, cost_bps=5.0, n_boot=N_BOOT,
                             base_block=21)
    mmq = st.discipline_table(X, ["M", "Q", "Y"], 12, cost_bps=5.0, n_boot=N_BOOT,
                              base_block=12)
    m90 = st.discipline_table(X[X.index >= "1990-01-01"], ["M", "Q", "Y"], 12,
                              cost_bps=5.0, n_boot=N_BOOT, base_block=12)
    disc = [dict(freq=f, tape="daily S&P 500 price, 1990-2018", **dd.loc[f].to_dict())
            for f in ("D", "W")]
    disc += [dict(freq=f, tape="monthly FF total return, 1926-2018", **mmq.loc[f].to_dict())
             for f in ("M", "Q", "Y")]
    h["discipline"] = disc
    h["discipline_check"] = (
        [dict(freq="M", tape="daily S&P 500 price, 1990-2018", **dd.loc["M"].to_dict())]
        + [dict(freq=f, tape="monthly FF total return, 1990-2018", **m90.loc[f].to_dict())
           for f in ("M", "Q", "Y")])
    cols = ["bh_cagr", "my_cagr_gross", "my_cagr_net", "d_cagr_net", "t_hac", "p_ret",
            "bh_sharpe", "my_sharpe_net", "d_sharpe", "p_sharpe", "time_invested",
            "switches_per_year"]
    print(pd.DataFrame(disc).set_index("freq")[cols].round(3).to_string())
    print(pd.DataFrame(h["discipline_check"]).set_index("freq")[cols].round(3).to_string())
    cs_d = st.cost_sweep(d, "D", 252)
    cs_m = st.cost_sweep(X, "M", 12)
    h["cost_daily"] = cs_d.reset_index().to_dict("records")
    h["cost_monthly"] = cs_m.reset_index().to_dict("records")
    print(cs_d.round(4).to_string())

    # ------------------------------------------------------------------ 5
    print("\n=== 5. synthetic control (machinery proof, not market evidence) ===")
    syn = []
    for s in (1.0, 0.5, 0.0):
        dfs, _ = data.synthetic_monthly(n_months=6000, signal_strength=s, seed=1025)
        Ps = st.pt_curve(dfs, st.HORIZONS_M)
        syn.append({"signal_strength": s,
                    "be_bonds": st.break_even(st.HORIZONS_M,
                                              (Ps["stock"] - Ps["bond"]).to_numpy()),
                    "be_bills": st.break_even(st.HORIZONS_M,
                                              (Ps["stock"] - Ps["bill"]).to_numpy())})
    h["synthetic"] = syn
    fin = {1.0: [], 0.0: []}
    for s in (1.0, 0.0):
        for seed in range(40):
            dfs, _ = data.synthetic_monthly(n_months=len(X), signal_strength=s, seed=seed)
            Ps = st.pt_curve(dfs, st.HORIZONS_M, cols=["stock", "bill"])
            fin[s].append(st.break_even(st.HORIZONS_M, (Ps["stock"] - Ps["bill"]).to_numpy()))
    h["syn_92y_planted_finite"] = float(np.mean(np.isfinite(fin[1.0])))
    h["syn_92y_null_finite"] = float(np.mean(np.isfinite(fin[0.0])))
    h["syn_92y_planted_iqr"] = [float(np.nanpercentile(np.where(np.isfinite(fin[1.0]),
                                                                fin[1.0], np.nan), q))
                                for q in (25, 75)]
    print(pd.DataFrame(syn).to_string(index=False))
    print(f"  92-year planted worlds with a finite bills break-even: "
          f"{h['syn_92y_planted_finite']:.0%}; null worlds: {h['syn_92y_null_finite']:.0%}")

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = time.time() - t0
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  runtime {h['runtime_s']:.0f}s")
    return h


def pct(x, nd=1):
    return f"{x:.{nd}%}"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prov = "\n".join(f"| {p['leg']} | {p['package']} | `{p['dataset']}` | `{p['sha256']}…` |"
                     for p in h["provenance"])
    loss = "\n".join(
        f"| {r['h']} | {pct(r['stock'])} [{pct(r['stock_lo'])}, {pct(r['stock_hi'])}] | "
        f"{pct(r['bond'])} [{pct(r['bond_lo'])}, {pct(r['bond_hi'])}] | "
        f"{pct(r['bill'])} [{pct(r['bill_lo'])}, {pct(r['bill_hi'])}] |"
        for r in h["loss_monthly"])
    lday = {1: "1 day", 2: "2 days", 5: "1 week", 10: "2 weeks", 21: "1 month",
            63: "3 months", 126: "6 months", 252: "1 year"}
    lossd = "\n".join(f"| {lday[r['h']]} | {pct(r['stock'])} [{pct(r['lo'])}, {pct(r['hi'])}] |"
                      for r in h["loss_daily"])
    gaps = "\n".join(
        f"| {r['h']} | {r['pt_stock']:+.4f} | {r['pt_bond']:+.4f} | {r['pt_bill']:+.4f} | "
        f"**{r['gap_b']:+.4f}** [{r['lo_b']:+.4f}, {r['hi_b']:+.4f}] | {r['p_b']:.3f} | "
        f"{r['gap_f']:+.4f} | {r['p_f']:.3f} |" for r in h["gap_rows"])
    est = "\n".join(f"| {e['method']} | {e['window']} | {fm(e['be_bonds'])} | "
                    f"{fm(e['be_bills'])} |" for e in h["estimators"])
    dgap = "\n".join(f"| {lday[r['h']]} | {r['gap']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}] | "
                     f"{r['p']:.3f} |" for r in h["daily_gap"])
    alph = h["sweep_w"]["alphas"]
    sw_hdr = " | ".join(f"α={a:.2f}" for a in alph)
    sw_w = "\n".join(f"| {r['lambda']:.2f} | " + " | ".join(fm(x) for x in r["be"]) + " |"
                     for r in h["sweep_w"]["rows"])
    sw_n = "\n".join(f"| {r['lambda']:.2f} | " + " | ".join(fm(x) for x in r["be"]) + " |"
                     for r in h["sweep_n"]["rows"])
    reg = "\n".join(
        f"| {r['case']} | {r['n']} | {pct(r['loss12'])} | **{fm(r['be_b'])}** "
        f"[{fm(r['lo_b'])}, {fm(r['hi_b'])}] | {fm(r['be_f'])} [{fm(r['lo_f'])}, "
        f"{fm(r['hi_f'])}] | {fm(r['be_iid'])} |" for r in h["regimes"])
    mats = "\n".join(f"| {r['maturity']:.0f}y | {pct(r['bond_vol'])} | {pct(r['bond_cagr'], 2)} "
                     f"| {fm(r['be'])} |" for r in h["maturities"])
    blks = "\n".join(f"| {r['block']} | [{fm(r['lo'])}, {fm(r['hi'])}] | "
                     f"{pct(r['share_finite'], 0)} |" for r in h["blocks"])

    def drow(r):
        return (f"| {st.FREQ_LABEL[r['freq']]} | {r['tape']} | {pct(r['bh_cagr'], 2)} | "
                f"{pct(r['my_cagr_gross'], 2)} | {pct(r['my_cagr_net'], 2)} | "
                f"**{r['d_cagr_net']:+.2%}** | {r['t_hac']:.2f} | {r['p_ret']:.3f} | "
                f"{r['bh_sharpe']:.2f} | {r['my_sharpe_net']:.2f} | {r['p_sharpe']:.3f} | "
                f"{pct(r['time_invested'], 0)} | {r['switches_per_year']:.1f} |")
    disc = "\n".join(drow(r) for r in h["discipline"])
    discc = "\n".join(drow(r) for r in h["discipline_check"])
    csd = "\n".join(f"| {r['cost_bps']:.0f} | {pct(r['my_cagr'], 2)} | {r['d_cagr']:+.2%} | "
                    f"{r['my_sharpe']:.2f} |" for r in h["cost_daily"])
    csm = "\n".join(f"| {r['cost_bps']:.0f} | {pct(r['my_cagr'], 2)} | {r['d_cagr']:+.2%} | "
                    f"{r['my_sharpe']:.2f} |" for r in h["cost_monthly"])
    vrs = "\n".join(f"| {r['asset']} | " + " | ".join(f"{x:.2f}" for x in r["vr"]) + " |"
                     for r in h["variance_ratios"])
    syn = "\n".join(f"| {r['signal_strength']:.1f} | {fm(r['be_bonds'])} | {fm(r['be_bills'])} |"
                    for r in h["synthetic"])
    return f"""# Results — Study 1025 (Don't Look) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py). As-of **{h['as_of']}** (last full
month of the Fama-French and core-CPI tapes; every leg is cut there). Monthly panel
{h['first_month']} → {h['as_of']}, {h['n_months']} months, fingerprint `{h['fp_monthly']}`;
daily panel {h['first_day']} → {h['as_of']}, {h['n_days']} sessions, fingerprint
`{h['fp_daily']}`. Circular block bootstrap of the joint monthly return vector, block
{st.BLOCK_M} months, {N_BOOT} resamples (headline) / {N_BOOT_ROB} (robustness); daily block
{st.BLOCK_D} sessions. Prospect theory: Tversky-Kahneman (1992), α = {st.ALPHA}, λ =
{st.LAMBDA}, probability weights γ = {st.GAMMA_GAIN} (gains), δ = {st.GAMMA_LOSS} (losses),
reference point a zero nominal return — Benartzi & Thaler's (1995) calibration.*

## 0. Data provenance

| Leg | Package | Dataset | Pinned SHA-256 |
|---|---|---|---|
{prov}

- **Stocks** — Fama-French `Mkt-RF + RF`: CRSP value-weighted market, **total return**, nominal.
- **Bills** — Fama-French `RF`, one-month T-bill.
- **Bonds** — **constructed**: Moody's seasoned **AAA corporate** yield (monthly average) turned
  into the monthly total return of a constant-maturity **20-year par bond** by the
  duration-convexity approximation `r ≈ y₋₁/12 − D·Δy + ½·C·Δy²` (exact par-bond repricing
  agrees to within 0.08 pp a month; see `tests/test_data.py`).
- **Real terms** — deflated by **core** CPI (`CPILFESL`, ex food & energy), 1957-02 onward.
- **Daily stocks** — S&P 500 **price index** (no dividends) from `skfolio`; bills spread from the
  monthly `RF`. Used only for the daily/weekly ends of the curve and the daily/weekly switcher.

## 1. How often do you *see* a loss?

Share of overlapping windows with a negative nominal return, with the 95% block-bootstrap band.

| Horizon (months) | Stocks | Bonds (AAA, 20y) | Bills |
|--:|--:|--:|--:|
{loss}

Daily tape (S&P 500 price index, 1990-2018):

| Horizon | Share of windows with a loss |
|---|--:|
{lossd}

The chance of seeing red on stocks falls from **{pct(h['loss_1d'], 0)} of days** to
**{pct(h['loss_1m'], 0)} of months**, **{pct(h['loss_12m'], 0)} of years** and
**{pct(h['loss_120m'], 0)} of decades**. This half of the story is mechanical — any asset whose
drift is positive and whose noise grows like √h shows fewer losses the less often it is
looked at.

## 2. Prospect-theory value by evaluation horizon

CPT value of holding each asset for one evaluation period (overlapping windows), and the gap
stocks − bonds with its 95% bootstrap band and two-sided bootstrap p.

| Horizon (months) | Stocks | Bonds | Bills | Gap stocks − bonds [95% CI] | p | Gap stocks − bills | p |
|--:|--:|--:|--:|--:|--:|--:|--:|
{gaps}

**Break-even ("equilibrium") horizon, stocks vs bonds: {fm(h['be'])}** — 95% bootstrap CI
**[{fm(h['be_lo'])}, {fm(h['be_hi'])}]**, bootstrap median {fm(h['be_median'])}; a finite
break-even exists in {pct(h['be_share_finite'], 0)} of resamples and lands between 6 and 24
months in only {pct(h['be_share_6_24'], 0)} of them.
Stocks vs bills: **{fm(h['be_bills'])}** [{fm(h['be_bills_lo'])}, {fm(h['be_bills_hi'])}].

### Three estimators, two windows

| Estimator | Window | Break-even vs bonds | vs bills |
|---|---|--:|--:|
{est}

The **i.i.d.** row draws months independently and compounds them — a world with no serial
dependence. It lands at about a year, just where Benartzi and Thaler did. The real *sequence* of
returns (overlapping windows) puts the break-even later. Variance ratios (variance of h-month
log returns over h × the monthly variance; 1 under independence) show why:

| Asset | 6 mo | 12 mo | 24 mo | 60 mo | 120 mo |
|---|--:|--:|--:|--:|--:|
{vrs}

Stocks are *more* volatile than independence implies at one to two years (short-run runs such
as 1929-32) and less at ten (mean reversion); i.i.d. resampling erases both, and the first is
exactly where the break-even lives. The constructed bond leg's ratios are inflated partly by
construction — a monthly-*average* yield smooths and autocorrelates the bond return — which is
one more reason the bond-side number is soft.

### The daily end — stocks (price only) vs bills

| Horizon | CPT gap stocks − bills [95% CI] | p |
|---|--:|--:|
{dgap}

A daily or weekly evaluator dislikes stocks decisively (p < 0.001 at every horizon up to three
months); on 29 years of price-only data the gap is still negative, though no longer
significant, at six months and a year.

## 3. Robustness

**Implied loss aversion.** The λ that makes an *annual* evaluator exactly indifferent between
stocks and bonds is **{h['lambda_12']:.2f}** ({h['lambda_12_bills']:.2f} vs bills); a *monthly*
evaluator needs only λ = {h['lambda_1']:.2f}.

**λ × α, with TK92 probability weighting** (break-even vs bonds):

| λ \\ α | {sw_hdr} |
|--:|{'--:|' * len(alph)}
{sw_w}

**λ × α, without probability weighting:**

| λ \\ α | {sw_hdr} |
|--:|{'--:|' * len(alph)}
{sw_n}

With λ = 1 (no loss aversion) the break-even collapses to the first grid point: the premium is
then attractive at any horizon. Loss aversion does the work: at α = 0.88, moving λ from 2 to 3 moves
the break-even by {h['lam_effect']:.0f} months; at λ = 2.25, moving α from 0.70 to 1.00 moves it
by {h['alpha_effect']:.0f}. Probability weighting **lengthens** the break-even here — it overweights
the rare deep stock losses.

**Sub-periods, real vs nominal** (bootstrap {N_BOOT_ROB} resamples each):

| Sample | Months | P(loss) 1y | Break-even vs bonds [95% CI] | vs bills [95% CI] | i.i.d. vs bonds |
|---|--:|--:|--:|--:|--:|
{reg}

**Bond maturity** (the bond leg is constructed; shorter bonds are less volatile, so stocks
need a longer horizon to beat them):

| Maturity | Bond vol | Bond CAGR | Break-even vs bonds |
|---|--:|--:|--:|
{mats}

**Block length** of the bootstrap (headline point estimate fixed):

| Block (months) | 95% CI of break-even vs bonds | Resamples with a finite break-even |
|--:|--:|--:|
{blks}

## 4. The discipline question — what does looking actually cost?

The rule, fixed before the run: at the end of each evaluation period the investor sees the
stock market's return over that period; after a **loss** they hold **bills** for the next
period, after a **gain** they hold **stocks**. One lag (the period's sign, known at its last
close, sets the position from the next observation). A switch turns over 1× NAV one-way and
pays **5 bp**. Gap = buy-and-hold minus switcher, net. Sharpe = excess of bills for both.

| Checking clock | Tape | B&H CAGR | Switcher CAGR gross | Switcher CAGR net | Gap (net) | HAC t | Boot p | B&H Sharpe | Switcher Sharpe (net) | Boot p (ΔSharpe) | Time in stocks | Switches / yr |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{disc}

Cross-checks on the common 1990-2018 window (not in the verdict):

| Checking clock | Tape | B&H CAGR | Switcher CAGR gross | Switcher CAGR net | Gap (net) | HAC t | Boot p | B&H Sharpe | Switcher Sharpe (net) | Boot p (ΔSharpe) | Time in stocks | Switches / yr |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{discc}

**Cost sweep** — daily checker (S&P price, 1990-2018):

| One-way cost (bp) | Switcher CAGR | Gap vs B&H | Switcher Sharpe |
|--:|--:|--:|--:|
{csd}

**Cost sweep** — monthly checker (FF total return, 1926-2018):

| One-way cost (bp) | Switcher CAGR | Gap vs B&H | Switcher Sharpe |
|--:|--:|--:|--:|
{csm}

## 5. Synthetic control — machinery proof, not market evidence

i.i.d. lognormal world, 6,000 months, planted arithmetic equity premium = `signal_strength` ×
6.6% a year (bills 3.6%, stock vol ≈ 19%):

| signal_strength | Break-even vs bonds | Break-even vs bills |
|--:|--:|--:|
{syn}

At zero premium no evaluation horizon on the grid makes stocks attractive against bills, as it
must. With the planted premium the break-even lands near a year. On **92-year** samples the
same planted world produces a finite stocks-vs-bills break-even in
{pct(h['syn_92y_planted_finite'], 0)} of 40 draws (interquartile range
{fm(h['syn_92y_planted_iqr'][0])} to {fm(h['syn_92y_planted_iqr'][1])}), and the null in
{pct(h['syn_92y_null_finite'], 0)} — the sampling noise in section 2's interval is a property
of ninety years of data, not of the method.

## Caveats

- **The bond leg is constructed**, from a *monthly-average* AAA corporate yield with a
  duration-convexity approximation at constant 20-year maturity. Averaging smooths monthly bond
  returns; the AAA spread adds a little credit premium. Benartzi and Thaler used a Treasury
  bond series that is not in the bundle. Section 3's maturity sweep shows the direction of the
  dependence.
- **Daily tape is price-only.** It omits ~2% a year of dividends: negligible at a one-day
  horizon, and in the switcher it *flatters* the switcher (buy-and-hold misses the dividends,
  the switcher earns bills while out), so the daily/weekly cost of myopia is understated.
- **Core CPI**, not headline, deflates the real-terms check, and only from 1957.
- **Ninety-two years hold nine independent decades.** Every long-horizon number rests on
  overlapping windows; inference resamples the monthly path in {st.BLOCK_M}-month blocks,
  which keeps short-run dependence but breaks multi-year mean reversion — section 3 shows the
  interval under other block lengths.
- **The CPT investor is narrow-framed by construction** — one asset, one evaluation period, a
  zero reference point. That is the model being tested, not an endorsement of it; Benartzi and
  Thaler's own equilibrium compares *portfolios*, which this study does not optimise.
- **The switcher is one rule**, fixed before the run. A different reaction (a one-period
  time-out, a partial de-risking) would cost a different amount; the point is the order of
  magnitude by checking clock, not a forecast for any particular investor.
- **5 bp one-way** is an index-fund / futures cost today; pre-1970 costs were higher, which
  would only raise the cost of myopia.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1025-myopic-loss-aversion](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    keep = {k: h[k] for k in (
        "as_of", "fp_monthly", "fp_daily", "be", "be_lo", "be_hi", "be_bills", "be_bills_lo",
        "be_bills_hi", "be_iid", "be_bt", "be_pre", "be_post", "be_real", "gap_short",
        "p_short", "gap_long", "p_long", "lambda_12", "loss_1d", "loss_1m", "loss_12m",
        "loss_120m", "runtime_s")}
    keep["discipline"] = [{k: r[k] for k in ("freq", "d_cagr_net", "t_hac", "p_ret",
                                             "d_sharpe", "p_sharpe")}
                          for r in h["discipline"]]
    keep["signal"] = h["_verdict"]["signal"]
    keep["trad"] = h["_verdict"]["trad"]
    print("##HEADLINE## " + json.dumps(keep, default=float))


if __name__ == "__main__":
    main()
