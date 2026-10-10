"""Real-tape verification — Study 1027 (Oil Is Equity in a Mirror). Regenerates docs/results.md.

Runs the mirror test on the frozen tapes inside the ``arch`` wheel: GJR-GARCH and EGARCH
asymmetry on daily WTI spot (1986-2018) and the S&P 500 price index (1999-2018) with robust
standard errors; a model-free forward-volatility regression split by the sign of today's
move (Newey-West) and a block-bootstrapped sign correlation; a pre-registered two-era split
(1986-2007 supply-shock era vs 2008-2018) plus a rolling three-year GJR; return skewness;
a monthly-average Brent cross-check; and the consequence for a 21-day vol-target overlay on
both assets, one execution lag, costs one-way × traded NAV. A synthetic GJR world with a
dialled asymmetry proves the estimators recover sign and size.

    python studies/1027-oil-inverse-leverage/examples/verify.py

Offline. Finishes in well under a minute.
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

from mirrorlev import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

# Pre-registered split (from the claim itself: the supply-shock era vs the era of the
# demand-driven crashes, 2008 and 2014-16). Fixed before the run; the three-era table and
# the rolling window are descriptive only.
ERAS = {"1986-2007": ("1986-01-01", "2007-12-31"),
        "2008-2018": ("2008-01-01", "2018-12-31")}
ERAS3 = {"1986-1999": ("1986-01-01", "1999-12-31"),
         "2000-2007": ("2000-01-01", "2007-12-31"),
         "2008-2018": ("2008-01-01", "2018-12-31")}
COST_WTI_BPS = 10.0   # one-way, a front-month futures proxy incl. roll slippage
COST_SP_BPS = 5.0     # one-way, index futures / ETF
N_BOOT = 1000


def report() -> dict:
    t0 = time.time()
    wti, sp = data.load_wti(), data.load_sp500()
    crude = data.load_crude_monthly()
    rw, rs = data.log_returns(wti), data.log_returns(sp)
    rw99 = rw[rw.index >= sp.index[0]]
    h: dict = {"as_of": data.AS_OF, "as_of_monthly": data.AS_OF_MONTHLY,
               "provenance": data.provenance()}
    print(f"as-of {data.AS_OF} (daily) / {data.AS_OF_MONTHLY} (monthly)")
    for p in h["provenance"]:
        print(f"  {p['tape']:10s} {p['rows']:6d} rows {p['first']} -> {p['last']}  "
              f"sha256 {p['sha256'][:12]}  fp {p['fingerprint']}")

    print("\n=== 1. parametric asymmetry (robust SEs) ===")
    fits = []
    for lab, win, r in (("WTI spot", "1986-2018", rw), ("WTI spot", "1999-2018", rw99),
                        ("S&P 500", "1999-2018", rs)):
        for mdl in ("gjr", "egarch"):
            f = st.fit_asymmetry(r, mdl)
            fits.append({"tape": lab, "window": win, **f})
            print(f"  {lab:9s} {win} {mdl:6s} gamma {f['gamma']:+.4f} (se {f['gamma_se']:.4f})"
                  f"  inverse t {f['inv_t']:+6.2f}  down/up {f['down_up_ratio']:.2f}"
                  f"  conv {f['converged']}")
    h["fits"] = fits
    get = {(f["tape"], f["window"], f["model"]): f for f in fits}
    h["wti_gjr_inv_t"] = get[("WTI spot", "1986-2018", "gjr")]["inv_t"]
    h["wti_egarch_inv_t"] = get[("WTI spot", "1986-2018", "egarch")]["inv_t"]
    h["sp_gjr_inv_t"] = get[("S&P 500", "1999-2018", "gjr")]["inv_t"]
    h["wti_gjr_ratio"] = get[("WTI spot", "1986-2018", "gjr")]["down_up_ratio"]

    print("\n=== 2. model-free: forward vol by the sign of today's move ===")
    mf = []
    for lab, win, r in (("WTI spot", "1986-2018", rw), ("WTI spot", "1999-2018", rw99),
                        ("S&P 500", "1999-2018", rs)):
        for hz in (5, 21):
            m = st.forward_vol_regression(r, hz)
            c = st.sign_correlation(r, hz, n_boot=N_BOOT)
            mf.append({"tape": lab, "window": win, **m,
                       "corr": c["corr"], "corr_lo": c["lo"], "corr_hi": c["hi"],
                       "corr_p": c["p"], "matched": c["matched_up_over_down"]})
            print(f"  {lab:9s} {win} h={hz:2d}  b_up {m['b_up']:+.2f}  b_dn {m['b_dn']:+.2f}"
                  f"  diff t {m['diff_t']:+6.2f} | corr {c['corr']:+.3f} "
                  f"[{c['lo']:+.3f}, {c['hi']:+.3f}]  matched up/down {c['matched_up_over_down']:.3f}")
    h["mf"] = mf
    gm = {(m["tape"], m["window"], m["h"]): m for m in mf}
    h["wti_mf_t"] = gm[("WTI spot", "1986-2018", 21)]["diff_t"]
    h["wti_mf5_t"] = gm[("WTI spot", "1986-2018", 5)]["diff_t"]
    h["wti_corr21"] = gm[("WTI spot", "1986-2018", 21)]["corr"]
    h["sp_mf_t"] = gm[("S&P 500", "1999-2018", 21)]["diff_t"]

    print("\n=== 3. regime or property? ===")
    e2 = st.era_table(rw, ERAS)
    e3 = st.era_table(rw, ERAS3)
    d = st.era_difference(e2, "1986-2007", "2008-2018")
    print(e2.round(3).to_string())
    print(e3.round(3).to_string())
    print(f"  GJR gamma 2008-2018 minus 1986-2007: {d['diff']:+.4f} (z {d['z']:+.2f})")
    h["eras"] = e2.reset_index().to_dict("records")
    h["eras3"] = e3.reset_index().to_dict("records")
    h["era_labels"] = list(e2.index)
    h["era_inv_t"] = [float(x) for x in e2["gjr_inv_t"]]
    h["era_inv"] = [float(-x) for x in e2["gjr_gamma"]]
    h["era_diff"] = d
    h["era_diff_z"] = d["z"]
    sp_e = st.era_table(rs, {"1999-2007": ("1999-01-01", "2007-12-31"),
                             "2008-2018": ("2008-01-01", "2018-12-31")})
    h["sp_eras"] = sp_e.reset_index().to_dict("records")
    print(sp_e.round(3).to_string())
    roll = st.rolling_asymmetry(rw, window=756, step=63)
    roll_sp = st.rolling_asymmetry(rs, window=756, step=63)
    h["roll_n"] = int(len(roll))
    h["roll_share_inverse"] = float((roll["inv_t"] >= 2).mean())
    h["roll_share_equity"] = float((roll["inv_t"] <= -2).mean())
    h["roll_max_t"] = float(roll["inv_t"].max())
    h["roll_max_date"] = str(roll["inv_t"].idxmax().date())
    h["roll_min_t"] = float(roll["inv_t"].min())
    h["roll_min_date"] = str(roll["inv_t"].idxmin().date())
    h["roll_pre08_share_inverse"] = float((roll.loc[:"2007-12-31", "inv_t"] >= 2).mean())
    h["roll_post08_share_equity"] = float(
        (roll.loc[pd.Timestamp("2010-12-31"):, "inv_t"] <= -2).mean())
    h["roll_sp_share_equity"] = float((roll_sp["inv_t"] <= -2).mean())
    h["roll_inverse_ends"] = [str(d.date()) for d in roll.index[roll["inv_t"] >= 2]]
    h["roll_last_inverse"] = h["roll_inverse_ends"][-1] if h["roll_inverse_ends"] else "never"
    print(f"  rolling 3y GJR on WTI ({len(roll)} windows): inverse t>=2 in "
          f"{h['roll_share_inverse']:.0%}, equity t<=-2 in {h['roll_share_equity']:.0%}; "
          f"max {h['roll_max_t']:+.2f} ({h['roll_max_date']}), min {h['roll_min_t']:+.2f} "
          f"({h['roll_min_date']})")
    print(f"  S&P rolling: equity t<=-2 in {h['roll_sp_share_equity']:.0%} of windows")

    print("\n=== 4. skewness ===")
    sk = []
    wm = data.log_returns(wti.resample("ME").last())
    sm_ = data.log_returns(sp.resample("ME").last())
    bm = data.log_returns(crude["brent"])
    for lab, r, blk in (("WTI spot daily 1986-2018", rw, 21), ("WTI spot daily 1999-2018", rw99, 21),
                        ("S&P 500 daily 1999-2018", rs, 21),
                        ("WTI spot month-end 1986-2018", wm, 6),
                        ("S&P 500 month-end 1999-2018", sm_, 6),
                        ("Brent monthly average 1987-2019", bm, 6)):
        s = st.skew_ci(r, n_boot=N_BOOT, block=blk)
        sk.append({"series": lab, **s})
        print(f"  {lab:32s} skew {s['skew']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]  "
              f"quantile skew {s['quantile_skew']:+.3f}")
    h["skew"] = sk
    sd = st.skew_difference(rw99, rs, n_boot=N_BOOT)
    h["skew_diff"] = sd
    print(f"  WTI minus S&P daily skew, common dates: {sd['diff']:+.2f} "
          f"[{sd['lo']:+.2f}, {sd['hi']:+.2f}]  p {sd['p']:.3f}")

    print("\n=== 5. monthly-average cross-check ===")
    mo = {}
    for col in ("brent", "wti"):
        m = st.monthly_asymmetry(crude[col])
        mo[col] = m
        print(f"  {col:5s} b_up {m['b_up']:+.3f}  b_dn {m['b_dn']:+.3f}  diff t {m['diff_t']:+.2f}"
              f"  (n {m['n']})")
    h["monthly"] = mo
    h["brent_mf_t"] = mo["brent"]["diff_t"]

    print("\n=== 6. the consequence: a vol-target overlay ===")
    rf = data.load_rf_daily(sp.index)
    sw, ss = data.simple_returns(wti), data.simple_returns(sp)
    ovs = {}
    for key, r, rfx, c in (("wti", sw, None, COST_WTI_BPS),
                           ("wti99", sw[sw.index >= sp.index[0]], None, COST_WTI_BPS),
                           ("sp", ss, rf, COST_SP_BPS)):
        o = st.vol_target_overlay(r, rfx, cost_bps=c, n_boot=N_BOOT)
        o.pop("series")
        ovs[key] = o
        print(f"  {key:5s} {o['start']}->{o['end']}  SR B&H {o['sharpe_bh']:.3f}  gross "
              f"{o['sharpe_gross']:.3f}  net {o['sharpe_net']:.3f}  gain {o['gain_net']:+.3f} "
              f"[{o['gain_lo']:+.3f}, {o['gain_hi']:+.3f}]  MDD {o['mdd_bh']:.0%} -> "
              f"{o['mdd_net']:.0%}  turnover {o['turnover_ann']:.1f}x  react {o['react_corr']:+.3f}"
              f"  breakeven {o['breakeven_bps']:.1f}bp")
    h["overlay"] = ovs
    for k, v in ovs["wti"].items():
        h[f"ov_wti_{k}"] = v
    h["ov_wti_react"] = ovs["wti"]["react_corr"]
    h["ov_sp_gain_net"] = ovs["sp"]["gain_net"]
    h["ov_sp_gain_lo"] = ovs["sp"]["gain_lo"]
    h["ov_sp_gain_hi"] = ovs["sp"]["gain_hi"]
    h["ov_sp_react"] = ovs["sp"]["react_corr"]
    cs_w = st.cost_sweep(sw, None)
    cs_s = st.cost_sweep(ss, rf)
    h["cost_sweep"] = {"wti": cs_w.reset_index().to_dict("records"),
                       "sp": cs_s.reset_index().to_dict("records")}
    print(pd.concat({"WTI gain": cs_w["gain_net"], "S&P gain": cs_s["gain_net"]}, axis=1)
          .round(3).to_string())

    print("\n=== 7. synthetic calibration (machinery proof, not evidence) ===")
    cal = []
    for k in (1.0, 0.5, 0.0, -0.5, -1.0):
        px, tr = data.synthetic_gjr(n_years=20, signal_strength=k, seed=1027)
        r = data.log_returns(px)
        g = st.fit_asymmetry(r, "gjr")
        e = st.fit_asymmetry(r, "egarch")
        m = st.forward_vol_regression(r, 21)
        o = st.vol_target_overlay(data.simple_returns(px), None, n_boot=2)
        cal.append({"k": k, "gamma_true": tr["gamma"], "gamma_hat": g["gamma"],
                    "gjr_inv_t": g["inv_t"], "egarch_inv_t": e["inv_t"], "mf_t": m["diff_t"],
                    "react": o["react_corr"]})
        print(f"  k {k:+.1f}  gamma true {tr['gamma']:+.3f} hat {g['gamma']:+.3f}  "
              f"inverse t GJR {g['inv_t']:+6.2f} EGARCH {e['inv_t']:+6.2f} MF {m['diff_t']:+6.2f}"
              f"  thermostat react {o['react_corr']:+.3f}")
    h["calibration"] = cal

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = round(time.time() - t0, 1)
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    print(f"  runtime {h['runtime_s']}s")
    return h


def _ratio(x: float) -> str:
    """Down/up impact ratio; '∞' when the fitted up-shock impact is zero or negative."""
    return "∞ (up shocks add ~nothing)" if (not np.isfinite(x) or x < 0) else f"{x:.2f}"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prov = "\n".join(
        f"| `{p['tape']}` | {p['label']} | {p['package']} | `{p['file']}` | "
        f"`{p['sha256'][:16]}…` | {p['first']} → {p['last']} | {p['rows']:,} | "
        f"`{p['fingerprint']}` |" for p in h["provenance"])
    fit_rows = "\n".join(
        f"| {f['tape']} | {f['window']} | {'GJR' if f['model'] == 'gjr' else 'EGARCH'} | "
        f"{f['gamma']:+.4f} | {f['gamma_se']:.4f} | {f['gamma_t']:+.2f} | **{f['inv_t']:+.2f}** | "
        f"{_ratio(f['down_up_ratio'])} | {f['persistence']:.3f} |" for f in h["fits"])
    mf_rows = "\n".join(
        f"| {m['tape']} | {m['window']} | {m['h']} | {m['b_up']:+.2f} | {m['b_dn']:+.2f} | "
        f"**{m['diff_t']:+.2f}** | {m['corr']:+.3f} [{m['corr_lo']:+.3f}, {m['corr_hi']:+.3f}] | "
        f"{m['matched']:.3f} |" for m in h["mf"])
    era_rows = "\n".join(
        f"| {e['era']} | {e['n']:,} | {e['ann_vol']:.0%} | {e['gjr_gamma']:+.4f} | "
        f"**{e['gjr_inv_t']:+.2f}** | {e['egarch_inv_t']:+.2f} | {e['mf_t']:+.2f} |"
        for e in h["eras"])
    era3_rows = "\n".join(
        f"| {e['era']} | {e['n']:,} | {e['ann_vol']:.0%} | {e['gjr_gamma']:+.4f} | "
        f"{e['gjr_inv_t']:+.2f} | {e['egarch_inv_t']:+.2f} | {e['mf_t']:+.2f} |"
        for e in h["eras3"])
    sp_era_rows = "\n".join(
        f"| S&P 500 {e['era']} | {e['n']:,} | {e['ann_vol']:.0%} | {e['gjr_gamma']:+.4f} | "
        f"{e['gjr_inv_t']:+.2f} | {e['egarch_inv_t']:+.2f} | {e['mf_t']:+.2f} |"
        for e in h["sp_eras"])
    sk_rows = "\n".join(
        f"| {s['series']} | {s['n']:,} | {s['skew']:+.2f} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | "
        f"{s['quantile_skew']:+.3f} |" for s in h["skew"])
    names = {"wti": "WTI spot (proxy) 1987-2018", "wti99": "WTI spot (proxy) 1999-2018",
             "sp": "S&P 500 price index − T-bill"}
    ov_rows = "\n".join(
        f"| {names[k]} | {o['cost_bps']:.0f} | {o['sharpe_bh']:.3f} | {o['sharpe_gross']:.3f} | "
        f"{o['sharpe_net']:.3f} | **{o['gain_net']:+.3f}** | [{o['gain_lo']:+.3f}, "
        f"{o['gain_hi']:+.3f}] | {o['mdd_bh']:.0%} → {o['mdd_net']:.0%} | "
        f"{o['turnover_ann']:.1f}× | {o['mean_w']:.2f} | **{o['react_corr']:+.3f}** |"
        for k, o in h["overlay"].items())
    cs_rows = "\n".join(
        f"| {a['cost_bps']:.0f} | {a['gain_net']:+.3f} | {b['gain_net']:+.3f} |"
        for a, b in zip(h["cost_sweep"]["wti"], h["cost_sweep"]["sp"]))
    cal_rows = "\n".join(
        f"| {c['k']:+.1f} | {c['gamma_true']:+.3f} | {c['gamma_hat']:+.3f} | {c['gjr_inv_t']:+.2f} | "
        f"{c['egarch_inv_t']:+.2f} | {c['mf_t']:+.2f} | {c['react']:+.3f} |"
        for c in h["calibration"])
    d = h["era_diff"]
    mo = h["monthly"]
    sd = h["skew_diff"]
    return f"""# Results — Study 1027 (Oil Is Equity in a Mirror) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py) in {h['runtime_s']} s. Daily as-of
**{h['as_of']}**, monthly as-of **{h['as_of_monthly']}**. Every number below is from the real
tapes except §7, which is the synthetic machinery check and is labelled as such.*

**Inverse score convention used throughout:** positive = volatility rises more after a
**rally** than after an equal fall (the oil claim); negative = the textbook equity leverage
effect.

## 0. Data provenance

All tapes ship inside the `arch` wheel and are read through `quantlab.bundled`, which refuses
any file whose SHA-256 differs from the pin.

| Tape | What it is | Package | File | SHA-256 pin | Span used | Rows | Fingerprint |
|---|---|---|---|---|---|--:|---|
{prov}

- **WTI is a spot price** (Cushing, FRED `DCOILWTICO`), not an investable futures return. The
  overlay in §6 treats the daily spot change as a proxy for the excess return of a
  fully-collateralised front-month position, ignoring roll yield. That caps Tradability.
- **The S&P 500 is a price index** (no dividends); its excess return subtracts the
  Fama-French T-bill rate, which ends 2018-11, so the S&P overlay ends there.
- **The monthly Brent/WTI tape holds monthly *averages*** (its WTI column equals the
  calendar-month mean of the daily tape to the cent), so it is used for direction only.
- WTI's three January-2019 prints are a partial month and are dropped; the monthly tape's
  January-2020 row is dropped likewise.

## 1. Parametric asymmetry — GJR-GARCH(1,1,1) and EGARCH(1,1,1)

Student-t innovations, constant mean, robust (Bollerslev-Wooldridge sandwich) standard errors,
returns in percent. GJR `gamma` > 0 is equity-type; EGARCH `gamma` < 0 is equity-type; the
"inverse t" column maps both onto one scale.

| Tape | Window | Model | gamma | robust SE | t(gamma) | Inverse t | Down/up impact | Persistence |
|---|---|---|--:|--:|--:|--:|--:|--:|
{fit_rows}

The S&P 500 shows the textbook leverage effect at overwhelming strength. **WTI shows the same
sign, not the mirror**: over 1986-2018 a down shock carries {h['wti_gjr_ratio']:.2f}× the
variance impact of an equal up shock in the GJR fit (inverse t = {h['wti_gjr_inv_t']:+.2f},
EGARCH {h['wti_egarch_inv_t']:+.2f}). The effect is several times weaker than in equities, and
on the common 1999-2018 window it is stronger than over the full sample — the first hint that it
lives in the later years.

## 2. Model-free — forward realised volatility split by the sign of today's move

`RV(t+1..t+h) = a + b_up·r⁺_t + b_dn·|r⁻_t| + c·RV21(t)`, Newey-West with *h* lags (the forward
windows overlap). The test is on `b_up − b_dn`. The correlation column is corr(r_t,
RV(t+1..t+h)) with a circular block-bootstrap 95% interval (blocks of max(h, 21) days); the last
column is forward vol after up days ÷ after down days, matched within |r| quintiles.

| Tape | Window | h | b_up | b_dn | t(b_up − b_dn) | corr(r, fwd RV) [95% CI] | Matched up/down |
|---|---|--:|--:|--:|--:|--:|--:|
{mf_rows}

No lens finds rallies followed by *more* volatility than falls. Over the full WTI sample the
model-free difference is equity-signed but short of the bar (t = {h['wti_mf_t']:+.2f} at 21 days,
{h['wti_mf5_t']:+.2f} at 5); on 1999-2018 it is equity-signed and significant. The |r|-matched
ratio sits below 1 everywhere — after an up day of a given size, oil is *calmer* than after a
down day of the same size.

## 3. A property of oil, or a regime?

**Pre-registered split** (fixed before the run, taken from the claim itself — the supply-shock
era against the decade of demand-driven crashes, 2008 and 2014-16):

| Era | Days | Ann. vol | GJR gamma | **GJR inverse t** | EGARCH inverse t | Model-free t (h=21) |
|---|--:|--:|--:|--:|--:|--:|
{era_rows}

GJR gamma, 2008-2018 minus 1986-2007: **{d['diff']:+.4f}** (SE {d['se']:.4f}, **z = {d['z']:+.2f}**;
the two eras are disjoint, so the fits are close to independent).

Descriptive three-era cut and the S&P mirror by era:

| Era | Days | Ann. vol | GJR gamma | GJR inverse t | EGARCH inverse t | Model-free t |
|---|--:|--:|--:|--:|--:|--:|
{era3_rows}
{sp_era_rows}

**Rolling three-year GJR** ({h['roll_n']} windows, 756 days, stepped 63): the inverse score
clears +2 in **{h['roll_share_inverse']:.0%}** of windows (pre-2008:
{h['roll_pre08_share_inverse']:.0%}) and falls below −2 in **{h['roll_share_equity']:.0%}**
(windows ending 2011 or later: {h['roll_post08_share_equity']:.0%}). Its peak is
{h['roll_max_t']:+.2f} (window ending {h['roll_max_date']}), its trough {h['roll_min_t']:+.2f}
(ending {h['roll_min_date']}). On the S&P, {h['roll_sp_share_equity']:.0%} of three-year windows
are equity-signed at t ≤ −2.

So the regime story is half right, and backwards. Something did change in 2008 — oil went
from **symmetric on average** to **equity-like** — but in the direction *away* from the mirror.
The mirror itself shows up only in a handful of three-year windows: those ending
{', '.join(h['roll_inverse_ends'])}, i.e. windows holding the 1990-91 Gulf War spike and the
tight-inventory run-up of 1996. None ends after {h['roll_last_inverse'][:4]}. Those windows are
overlapping and were found by looking, so they are a description, not a test — and in the
pre-registered eras the inverse sign never clears the bar.

## 4. Skewness

Block-bootstrap 95% intervals; the quantile skew `(q95 + q05 − 2·q50)/(q95 − q05)` is immune to
a single extreme day.

| Series | n | Skewness | 95% CI | Quantile skew |
|---|--:|--:|--:|--:|
{sk_rows}

WTI minus S&P daily skewness on common dates: **{sd['diff']:+.2f}** [{sd['lo']:+.2f},
{sd['hi']:+.2f}], p = {sd['p']:.3f} — indistinguishable. An asset whose storms arrive in
rallies should be *positively* skewed, or at least clearly less negatively skewed than equities.
Daily WTI is neither: its full-sample moment skew is negative, dominated by a few crash days
(17 January 1991 alone is −33% in log terms), and its quantile skew is close to the S&P's. Only
at the monthly horizon is oil visibly less left-skewed than the S&P (on different windows, and
with overlapping intervals) — consistent with a *weaker* leverage effect, not with a reversed one.

## 5. Monthly-average cross-check

`|r_(m+1)| = a + b_up·r⁺_m + b_dn·|r⁻_m| + c·mean|r|(3m)`, Newey-West(3):

| Series | b_up | b_dn | t(b_up − b_dn) | n |
|---|--:|--:|--:|--:|
| Brent (monthly average) | {mo['brent']['b_up']:+.3f} | {mo['brent']['b_dn']:+.3f} | {mo['brent']['diff_t']:+.2f} | {mo['brent']['n']} |
| WTI (monthly average) | {mo['wti']['b_up']:+.3f} | {mo['wti']['b_dn']:+.3f} | {mo['wti']['diff_t']:+.2f} | {mo['wti']['n']} |

Same sign as the daily tape, short of the bar: big down months, not big up months, are followed
by the bigger moves.

## 6. The consequence — a vol-target overlay

`w_t = min(2, target_t / RV21_t)`, `target_t` the expanding mean of RV21 (past-only). The
weight set at the close of *t* earns day *t+1* — **one lag**. Costs one-way × |Δw| × NAV.
Excess-vs-excess on both legs: the S&P subtracts the T-bill; the WTI spot change is treated
as an already-excess futures-proxy return. "Reacts" = corr(trailing 21-day return, weight):
positive means the thermostat de-risks after **falls**, negative after **rallies**.

| Book | Cost (bp) | SR buy & hold | SR gross | SR net | Net gain | 95% CI (block bootstrap) | Max DD | Turnover/yr | Mean w | Reacts |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{ov_rows}

Net Sharpe gain across one-way costs:

| Cost (bp) | WTI 1987-2018 | S&P 1999-2018 |
|--:|--:|--:|
{cs_rows}

On the S&P the thermostat behaves as the leverage effect predicts — it cuts after falls
(reacts {h['ov_sp_react']:+.2f}). On WTI over the full sample it is nearly **sign-blind**
({h['ov_wti_react']:+.2f}): oil's volatility answers the *size* of a move far more than its
direction. On 1999-2018 it leans equity-like ({h['overlay']['wti99']['react_corr']:+.2f}). In no
window does the thermostat de-risk after rallies, which is what the mirror would require. Its
value-add on the WTI spot proxy is negative even before costs; on the S&P price index it is
small and positive before costs, and gone by ~{h['overlay']['sp']['breakeven_bps']:.0f} bp one-way.
Neither bootstrap interval excludes zero.

## 7. Synthetic calibration — machinery proof, NOT market evidence

A GJR-GARCH(1,1,1) with Student-t(8) shocks, 20 years, 30% vol, persistence 0.97; the knob
`k` sets the planted gamma = 0.10·k (k = +1 equity-type, −1 the oil mirror, 0 symmetric).

| k | gamma planted | gamma estimated | GJR inverse t | EGARCH inverse t | Model-free t | Thermostat reacts |
|--:|--:|--:|--:|--:|--:|--:|
{cal_rows}

The estimators recover the sign and size of the asymmetry, read zero on the symmetric null,
and the thermostat's reaction flips sign with the mirror — so had oil carried the inverse
effect, this harness would have shown it.

## Caveats

- **Spot is not investable.** The overlay numbers are a spot proxy; a real futures book adds
  roll yield (large and regime-dependent in oil) and pays roll costs this tape cannot see.
- **The tape ends in 2018.** It holds 2008 and 2014-16 but not April 2020, when the front-month
  WTI future settled below zero — the most extreme demand-driven crash in the history of the
  contract. If anything that would push the late era further towards the equity sign.
- **One commodity.** Gold and other markets are not tested here; the published inverse
  asymmetry in gold is outside this study's evidence.
- **The era split is one pre-registered cut.** The three-era table and the rolling window are
  descriptive; their many windows are not independent tests and are not used for the stamp.
- **Moment skewness is fragile** on a tape with a −33% day; the quantile skew is the robust
  read.
- **GARCH asymmetry is a parametric summary.** The model-free regression is there because a
  mis-specified variance equation can manufacture or hide asymmetry; where they disagree in
  strength (full-sample GJR vs model-free) the model-free number is the conservative one.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1027-oil-inverse-leverage](../README.md). Not investment advice.*
"""


def _jsonable(h: dict) -> dict:
    out = {}
    for k, val in h.items():
        if k in ("provenance", "fits", "mf", "eras", "eras3", "sp_eras", "skew", "calibration",
                 "cost_sweep", "overlay", "monthly", "era_diff", "skew_diff"):
            continue
        out[k] = val
    return out


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(_jsonable(h), default=float))


if __name__ == "__main__":
    main()
