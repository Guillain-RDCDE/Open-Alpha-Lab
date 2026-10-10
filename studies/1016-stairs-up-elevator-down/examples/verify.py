"""Real-tape verification — Study 1016 (Stairs Up, Elevator Down). Regenerates docs/results.md.

Times the first X% of every drawup and drawdown leg on three frozen tapes, tests the paths for
time reversibility (Ramsey-Rothman, Newey-West and block-bootstrap errors), measures skewness by
horizon with block-bootstrap intervals, fits GARCH / GJR / EGARCH and asks how much of the
asymmetry each reproduces, then races a pre-registered fast-exit / slow-entry rule against its
symmetric twins and buy-and-hold after costs with one execution lag.

    python studies/1016-stairs-up-elevator-down/examples/verify.py

Offline: every tape comes from ``quantlab.bundled`` (SHA-256 pinned). Runs in about a minute.
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

from stairs import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

N_NULL = 200          # Monte Carlo replications for the leg nulls
N_BOOT = 1000         # block-bootstrap replications (statistics)
N_BOOT_RACE = 500     # block-bootstrap replications (Sharpe differences)
N_PATHS = 100         # simulated paths per model in the mechanism check
COST_DAILY = 0.0005   # 5 bp one-way: index futures / a large ETF
COST_MONTHLY = 0.0010  # 10 bp one-way on the 1926+ monthly tape (older, dearer markets)
TRADE_END = data.FF_AS_OF   # cash rate (Ken French T-bill) ends here


def _rows(df: pd.DataFrame) -> list:
    return df.reset_index().to_dict("records")


def report() -> dict:
    t0 = time.time()
    sp = data.load_sp500()
    nq = data.load_nasdaq()
    ff = data.load_ff_market()
    fi = data.ff_index(ff)
    h: dict = {"as_of": data.AS_OF, "sp_as_of": data.SP500_AS_OF,
               "nq_as_of": data.NASDAQ_AS_OF, "ff_as_of": data.FF_AS_OF,
               "fp_sp": data.fingerprint(sp), "fp_nq": data.fingerprint(nq),
               "fp_ff": data.fingerprint(ff),
               "sp_span": f"{sp.index[0].date()} → {sp.index[-1].date()}",
               "nq_span": f"{nq.index[0].date()} → {nq.index[-1].date()}",
               "ff_span": f"{ff.index[0].strftime('%Y-%m')} → {ff.index[-1].strftime('%Y-%m')}",
               "n_sp": int(len(sp)), "n_nq": int(len(nq)), "n_ff": int(len(ff)),
               "provenance": data.PROVENANCE}
    print(f"as-of {data.AS_OF}")
    print(f"  S&P 500 price index {h['sp_span']}  n={len(sp)}  fp {h['fp_sp']}")
    print(f"  Nasdaq price index  {h['nq_span']}  n={len(nq)}  fp {h['fp_nq']}")
    print(f"  FF market TR        {h['ff_span']}  n={len(ff)}  fp {h['fp_ff']}")

    r_sp = st.log_returns(sp).to_numpy()
    r_nq = st.log_returns(nq).to_numpy()
    r_ff = np.log(fi).diff().dropna().to_numpy()

    # ------------------------------------------------------------------ 1. legs
    print("\n=== 1. time to gain X% vs time to lose X% ===")
    legs = {}
    for name, r in (("S&P 500 (daily)", r_sp), ("Nasdaq (daily)", r_nq),
                    ("FF market (monthly)", r_ff)):
        sf = st.leg_null_test(r, kind="signflip", n_rep=N_NULL, seed=1016)
        pm = st.leg_null_test(r, kind="permutation", n_rep=N_NULL, seed=1017)
        sf["p_perm"] = pm["p"]
        sf["perm_median"] = pm["null_median"]
        legs[name] = sf
        print(f"  {name}")
        print(sf[["n_up", "n_down", "gm_up", "gm_down", "elevator_ratio", "null_median",
                  "p", "p_reverse", "p_perm", "speed_ratio", "speed_p"]].round(3).to_string())
    h["legs"] = {k: _rows(v) for k, v in legs.items()}
    L = legs["S&P 500 (daily)"]
    for x, tag in ((0.05, "05"), (0.10, "10"), (0.20, "20")):
        h[f"elev{tag}"] = float(L.loc[x, "elevator_ratio"])
        h[f"speed{tag}"] = float(L.loc[x, "speed_ratio"])
        h[f"speed{tag}_p"] = float(L.loc[x, "speed_p"])
    h["gm_up10"] = float(L.loc[0.10, "gm_up"])
    h["gm_down10"] = float(L.loc[0.10, "gm_down"])
    h["elev10_p_signflip"] = float(L.loc[0.10, "p"])
    h["elev10_p_reverse"] = float(L.loc[0.10, "p_reverse"])
    h["elev10_p_perm"] = float(L.loc[0.10, "p_perm"])
    h["nq_elev10"] = float(legs["Nasdaq (daily)"].loc[0.10, "elevator_ratio"])
    h["ff_elev10"] = float(legs["FF market (monthly)"].loc[0.10, "elevator_ratio"])
    # a plotting/notebook aid: the S&P 500's 10% legs
    lg = st.legs(np.concatenate([[0.0], np.cumsum(r_sp)]), np.log1p(0.10))
    h["sp_legs10"] = lg.to_dict("records")

    # ------------------------------------------------------------------ 2. reversibility
    print("\n=== 2. time reversibility (Ramsey-Rothman) ===")
    tr_sp = st.tr_stats(r_sp, K=10, n_boot=N_BOOT, block=20, seed=1016)
    tr_spw = st.tr_stats(r_sp, K=10, n_boot=200, block=20, seed=1016, winsor=0.005)
    tr_nq = st.tr_stats(r_nq, K=10, n_boot=N_BOOT, block=20, seed=1016)
    wk = st.aggregate(np.log(sp), "W-FRI").to_numpy()
    tr_wk = st.tr_stats(wk, K=8, n_boot=N_BOOT, block=8, seed=1016)
    tr_ff = st.tr_stats(r_ff, K=6, n_boot=N_BOOT, block=6, seed=1016)
    tr_tbl = pd.DataFrame([
        {"tape": "S&P 500 daily", "K": 10, "n": tr_sp["n"], "tr_sum": tr_sp["tr_sum"],
         "t_nw": tr_sp["t_nw"], "t_boot": tr_sp["t_boot"], "reversed": tr_sp["tr_sum_reversed"]},
        {"tape": "S&P 500 daily, 0.5% tails clipped", "K": 10, "n": tr_spw["n"],
         "tr_sum": tr_spw["tr_sum"], "t_nw": tr_spw["t_nw"], "t_boot": tr_spw["t_boot"],
         "reversed": tr_spw["tr_sum_reversed"]},
        {"tape": "S&P 500 weekly", "K": 8, "n": tr_wk["n"], "tr_sum": tr_wk["tr_sum"],
         "t_nw": tr_wk["t_nw"], "t_boot": tr_wk["t_boot"], "reversed": tr_wk["tr_sum_reversed"]},
        {"tape": "Nasdaq daily", "K": 10, "n": tr_nq["n"], "tr_sum": tr_nq["tr_sum"],
         "t_nw": tr_nq["t_nw"], "t_boot": tr_nq["t_boot"], "reversed": tr_nq["tr_sum_reversed"]},
        {"tape": "FF market monthly (TR)", "K": 6, "n": tr_ff["n"], "tr_sum": tr_ff["tr_sum"],
         "t_nw": tr_ff["t_nw"], "t_boot": tr_ff["t_boot"], "reversed": tr_ff["tr_sum_reversed"]},
    ]).set_index("tape")
    print(tr_tbl.round(3).to_string())
    print("  S&P 500 per lag:")
    print(tr_sp["per_lag"].round(3).T.to_string())
    h["tr_table"] = _rows(tr_tbl)
    h["tr_per_lag"] = _rows(tr_sp["per_lag"])
    h["tr_per_lag_nq"] = _rows(tr_nq["per_lag"])
    h["tr_per_lag_ff"] = _rows(tr_ff["per_lag"])
    h["tr_sum"], h["tr_t_nw"], h["tr_t_boot"] = tr_sp["tr_sum"], tr_sp["t_nw"], tr_sp["t_boot"]
    h["tr_t_nw_wins"] = tr_spw["t_nw"]
    h["tr_reversed"] = tr_sp["tr_sum_reversed"]
    h["nq_tr_t_nw"], h["nq_tr_t_boot"] = tr_nq["t_nw"], tr_nq["t_boot"]
    h["ff_tr_t_nw"], h["ff_tr_t_boot"] = tr_ff["t_nw"], tr_ff["t_boot"]
    h["wk_tr_t_nw"] = tr_wk["t_nw"]

    # ------------------------------------------------------------------ 3. skew by horizon
    print("\n=== 3. skewness by horizon ===")
    lsp, lnq, lff = np.log(sp), np.log(nq), np.log(fi)
    series = {
        "S&P 500 daily": r_sp, "S&P 500 weekly": st.aggregate(lsp, "W-FRI").to_numpy(),
        "S&P 500 monthly": st.aggregate(lsp, "ME").to_numpy(),
        "Nasdaq daily": r_nq, "Nasdaq weekly": st.aggregate(lnq, "W-FRI").to_numpy(),
        "Nasdaq monthly": st.aggregate(lnq, "ME").to_numpy(),
        "FF monthly": r_ff, "FF quarterly": st.aggregate(lff, "QE").to_numpy(),
        "FF annual": st.aggregate(lff, "YE").to_numpy(),
    }
    blocks = {"S&P 500 daily": 21, "S&P 500 weekly": 8, "S&P 500 monthly": 6,
              "Nasdaq daily": 21, "Nasdaq weekly": 8, "Nasdaq monthly": 6,
              "FF monthly": 6, "FF quarterly": 4, "FF annual": 2}
    sk = st.skew_table(series, blocks, n_boot=N_BOOT, seed=1016)
    print(sk.round(3).to_string())
    h["skew_table"] = _rows(sk)
    h["skew_d"], h["skew_d_lo"], h["skew_d_hi"] = (float(sk.loc["S&P 500 daily", c])
                                                   for c in ("skew", "skew_lo", "skew_hi"))
    h["skew_m"], h["skew_m_lo"], h["skew_m_hi"] = (float(sk.loc["S&P 500 monthly", c])
                                                   for c in ("skew", "skew_lo", "skew_hi"))
    h["kelly_d"] = float(sk.loc["S&P 500 daily", "kelly"])
    h["kelly_m"] = float(sk.loc["S&P 500 monthly", "kelly"])

    # ------------------------------------------------------------------ 4. mechanism
    print("\n=== 4. mechanism: who reproduces the asymmetry? ===")
    fits = st.fit_models(r_sp)
    ft = pd.DataFrame(fits).T
    print(ft.round(4).to_string())
    h["fits"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in fits.items()}
    mc = st.mechanism_check(r_sp, fits, n_paths=N_PATHS, seed=1016)
    print(mc.round(3).to_string())
    h["mechanism"] = _rows(mc)
    key = {"gjr": "GJR-GARCH (leverage)", "egarch": "EGARCH (leverage)",
           "garch": "GARCH (symmetric)", "iid": "i.i.d. Student-t",
           "perm": "i.i.d. shuffle of the tape"}
    for k, w in key.items():
        h[f"share_tr_{k}"] = float(mc.loc[w, "share_tr"])
        h[f"share_elev_{k}"] = float(mc.loc[w, "share_elevator"])
        h[f"share_skm_{k}"] = float(mc.loc[w, "share_skew_m"])

    # ------------------------------------------------------------------ 5. trading
    print("\n=== 5. could you trade it? (pre-registered rules, one lag, net) ===")
    sp_t = sp[sp.index <= pd.Timestamp(TRADE_END)]
    nq_t = nq[nq.index <= pd.Timestamp(TRADE_END)]
    rf_sp, rf_nq = data.daily_rf(sp_t.index, ff), data.daily_rf(nq_t.index, ff)
    rf_m = ff["rf"].reindex(fi.index).fillna(0.0)
    races = {
        "S&P 500 daily 1990-2018": st.race(sp_t, rf_sp, COST_DAILY, 1, 252, 20, N_BOOT_RACE),
        "Nasdaq daily 1999-2018": st.race(nq_t, rf_nq, COST_DAILY, 1, 252, 20, N_BOOT_RACE),
        "FF monthly TR 1926-2018": st.race(fi, rf_m, COST_MONTHLY, 1, 12, 6, N_BOOT_RACE),
    }
    fi45 = fi[fi.index >= pd.Timestamp("1945-12-31")]
    races["FF monthly TR 1946-2018"] = st.race(fi45, rf_m, COST_MONTHLY, 1, 12, 6, N_BOOT_RACE)
    races["S&P 500 daily, 2-day lag"] = st.race(sp_t, rf_sp, COST_DAILY, 2, 252, 20, N_BOOT_RACE)
    for k, v in races.items():
        print(f"  {k}")
        print(v[["sharpe_gross", "sharpe_net", "sharpe_bh", "d_vs_bh", "p_vs_bh",
                 "time_in", "trades_per_year", "maxdd_net", "maxdd_bh"]].round(3).to_string())
    h["races"] = {k: _rows(v) for k, v in races.items()}
    P = st.PRIMARY_RULE
    h["trade_d_vs_bh"] = {
        "sp500": float(races["S&P 500 daily 1990-2018"].loc[P, "d_vs_bh"]),
        "nasdaq": float(races["Nasdaq daily 1999-2018"].loc[P, "d_vs_bh"]),
        "ff": float(races["FF monthly TR 1926-2018"].loc[P, "d_vs_bh"])}
    ffr = races["FF monthly TR 1926-2018"]
    h["trade_ff_p_vs_bh"] = float(ffr.loc[P, "p_vs_bh"])
    h["trade_ff_beats_sym"] = bool(ffr.loc["symmetric (5% / 5%)", "d_primary_minus_this"] > 0)
    h["trade_ff_post45_d"] = float(races["FF monthly TR 1946-2018"].loc[P, "d_vs_bh"])
    h["trade_ff_post45_p"] = float(races["FF monthly TR 1946-2018"].loc[P, "p_vs_bh"])
    h["trade_ff_sym10_d"] = float(ffr.loc["symmetric (10% / 10%)", "d_vs_bh"])
    h["trade_sp_sym10_d"] = float(races["S&P 500 daily 1990-2018"].loc[
        "symmetric (10% / 10%)", "d_vs_bh"])
    h["cost_daily_bps"] = COST_DAILY * 1e4
    h["cost_monthly_bps"] = COST_MONTHLY * 1e4
    h["sp_trade_sharpe_net"] = float(races["S&P 500 daily 1990-2018"].loc[P, "sharpe_net"])
    h["sp_trade_sharpe_bh"] = float(races["S&P 500 daily 1990-2018"].loc[P, "sharpe_bh"])
    h["sp_trade_p_vs_bh"] = float(races["S&P 500 daily 1990-2018"].loc[P, "p_vs_bh"])
    h["sp_trade_time_in"] = float(races["S&P 500 daily 1990-2018"].loc[P, "time_in"])

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = round(time.time() - t0, 1)
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    print(f"  runtime {h['runtime_s']} s")
    return h


def _f(v, fmt):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prov = h["provenance"]
    leg_rows = []
    for tape, rows in h["legs"].items():
        unit = "months" if "monthly" in tape else "sessions"
        for r in rows:
            leg_rows.append(
                f"| {tape} | {r['x']:.0%} | {r['n_up']} / {r['n_down']} | "
                f"{r['gm_up']:.1f} | {r['gm_down']:.1f} | **{r['elevator_ratio']:.2f}×** | "
                f"{r['null_median']:.2f}× | {r['p']:.3f} | {r['p_reverse']:.3f} | "
                f"{r['p_perm']:.3f} | {_f(r['speed_ratio'], '.2f')}× | "
                f"{_f(r['speed_p'], '.3f')} | {unit} |")
    leg_md = "\n".join(leg_rows)
    tr_md = "\n".join(
        f"| {r['tape']} | {r['K']} | {r['n']:,} | **{r['tr_sum']:+.3f}** | "
        f"**{r['t_nw']:+.2f}** | {r['t_boot']:+.2f} | {r['reversed']:+.3f} |"
        for r in h["tr_table"])
    lag_md = "\n".join(
        f"| {r['k']} | {r['tr']:+.3f} | {r['t_nw']:+.2f} |" for r in h["tr_per_lag"])
    sk_md = "\n".join(
        f"| {r['series']} | {r['n']:,} | {r['skew']:+.2f} | [{r['skew_lo']:+.2f}, "
        f"{r['skew_hi']:+.2f}] | {r['kelly']:+.3f} | [{r['kelly_lo']:+.3f}, "
        f"{r['kelly_hi']:+.3f}] |" for r in h["skew_table"])
    fit_md = "\n".join(
        f"| {k.upper()} | {f['alpha']:.3f} | {f['gamma']:+.3f} | {_f(f['gamma_t'], '+.1f')} | "
        f"{f['beta']:.3f} | {f['nu']:.1f} | {f['loglik']:,.1f} |" for k, f in h["fits"].items())
    mc_md = "\n".join(
        f"| {r['world']} | {r['tr_sum']:+.3f} | {r['skew_daily']:+.2f} | "
        f"{r['skew_monthly']:+.2f} | {r['elevator_ratio']:.2f}× | **{r['share_tr']:.0%}** | "
        f"{r['share_skew_m']:.0%} | {r['share_elevator']:.0%} | "
        f"{_f(r['tr_p05'], '+.2f')} … {_f(r['tr_p95'], '+.2f')} |" for r in h["mechanism"])
    race_md = []
    for tape, rows in h["races"].items():
        for r in rows:
            race_md.append(
                f"| {tape} | {r['rule']} | {r['sharpe_gross']:.2f} | **{r['sharpe_net']:.2f}** | "
                f"{r['sharpe_bh']:.2f} | **{r['d_vs_bh']:+.2f}** [{r['d_vs_bh_lo']:+.2f}, "
                f"{r['d_vs_bh_hi']:+.2f}] | {r['p_vs_bh']:.3f} | {r['time_in']:.0%} | "
                f"{r['trades_per_year']:.1f} | {r['maxdd_net']:.0%} / {r['maxdd_bh']:.0%} |")
    race_md = "\n".join(race_md)
    mech = {r["world"]: r for r in h["mechanism"]}
    gjr = mech["GJR-GARCH (leverage)"]
    return f"""# Results — Study 1016 (Stairs Up, Elevator Down) on three real tapes

*Generated by [`examples/verify.py`](../examples/verify.py). Study as-of
**{h['as_of']}** (latest full month on any tape used). Every number below is from a real tape
except section 4's simulated worlds, which are labelled as such.*

## Data provenance

| Tape | Source package | Kind | Span (full periods only) | Obs | SHA-256 pin (file) | Fingerprint |
|---|---|---|---|--:|---|---|
| S&P 500 | {prov['sp500']['package']}, `{prov['sp500']['tape']}` | {prov['sp500']['kind']} | {h['sp_span']} | {h['n_sp']:,} | `{prov['sp500']['sha256'][:16]}…` | `{h['fp_sp']}` |
| Nasdaq Composite | {prov['nasdaq']['package']}, `{prov['nasdaq']['tape']}` (`Adj Close`) | {prov['nasdaq']['kind']} | {h['nq_span']} | {h['n_nq']:,} | `{prov['nasdaq']['sha256'][:16]}…` | `{h['fp_nq']}` |
| US market (Fama-French) | {prov['ff']['package']}, `{prov['ff']['tape']}` | {prov['ff']['kind']} | {h['ff_span']} | {h['n_ff']:,} | `{prov['ff']['sha256'][:16]}…` | `{h['fp_ff']}` |

Loaded through [`quantlab/bundled.py`](../../../quantlab/bundled.py), which refuses any file whose
bytes differ from the pin. The skfolio S&P 500 tape ends 2022-12-28, mid-month, so December 2022
is dropped. The two daily tapes are **price indices**: no dividends.

## 1. Time to gain X% against time to lose X%

Each path is cut into alternating drawup and drawdown legs at a log-symmetric reversal size
`h = log(1+X)`. For each leg we time the **first passage** across `h` from its starting extreme —
the days to gain X% off a trough, the days to lose X% from a peak. Elevator ratio = geometric-mean
up time ÷ geometric-mean down time; **above 1 means falls are faster**. *p* (elevator) and
*p* (reverse) are one-sided Monte Carlo p-values against {N_NULL} **sign-flip** paths (every
|return|, the drift and the volatility clustering kept; signs randomised); *p* (shuffle) uses a
permutation that keeps each day's size *and* sign but destroys the order. Speed ratio = median
|move| per day of completed down legs ÷ completed up legs, with its own sign-flip p.

| Tape | X | Legs up / down | GM time up | GM time down | Elevator ratio | Null median | *p* (elevator) | *p* (reverse) | *p* (shuffle) | Speed ratio | Speed *p* | Unit |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
{leg_md}

**The literal proverb runs backwards.** On the S&P 500, losing 10% from a peak took a
geometric-mean **{h['gm_down10']:.1f} sessions** and gaining 10% off a trough
**{h['gm_up10']:.1f}** — an elevator ratio of **{h['elev10']:.2f}×**. The same holds at 5%
({h['elev05']:.2f}×) and 20% ({h['elev20']:.2f}×), on the Nasdaq ({h['nq_elev10']:.2f}× at 10%)
and on the monthly century ({h['ff_elev10']:.2f}×). Against the sign-flip null the tape sits in
the *reverse* tail (p = {h['elev10_p_reverse']:.3f}): **the fastest X% on the tape is the rebound
off a low.** What the steelman can keep is the *completed* leg: whole declines are shorter and
steeper than whole advances (speed ratio {h['speed10']:.2f}× at 10%, sign-flip p =
{h['speed10_p']:.3f}; {h['speed05']:.2f}× at 5%, p = {h['speed05_p']:.3f}) — because bull legs
run for a long time, not because the first 10% of a fall is quick.

## 2. Is the path time-reversible?

Ramsey-Rothman (1996): with standardised returns *x*, `TR(k) = E[x_t² x_(t−k)] − E[x_t x_(t−k)²]`.
Every TR(k) is zero for a time-reversible process; a negative value means a fall is followed by
bigger squared moves than a rise is — the leverage signature. Summed over k = 1..K. Newey-West
t on the summed product series; block-bootstrap t from a circular block bootstrap of the
standardised returns ({N_BOOT} resamples). "Reversed" is the statistic on the time-reversed
series — the exact negative.

| Tape | K | n | TR sum | NW *t* | Block-bootstrap *t* | Reversed copy |
|---|--:|--:|--:|--:|--:|--:|
{tr_md}

S&P 500, lag by lag:

| k | TR(k) | NW *t* |
|--:|--:|--:|
{lag_md}

**The tape is time-irreversible, robustly**: t = {h['tr_t_nw']:+.2f} (NW) and
{h['tr_t_boot']:+.2f} (block bootstrap) on the S&P 500; {h['tr_t_nw_wins']:+.2f} with the 0.5%
tails clipped, so it is not three crash days; {h['nq_tr_t_nw']:+.2f} on the Nasdaq. On weekly
data it is {h['wk_tr_t_nw']:+.2f} and on a century of monthly total returns {h['ff_tr_t_nw']:+.2f}
(block bootstrap {h['ff_tr_t_boot']:+.2f}). The sign is the same at every horizon and on every
tape; only the monthly Newey-West t falls short of 2, on a seventh of the observations. The
largest single term is lag 1: volatility responds to the sign of yesterday's move.

## 3. Skewness by horizon

Moment skewness of non-overlapping log returns, with circular-block-bootstrap 95% intervals;
Kelly quantile skewness `(q90 + q10 − 2·q50)/(q90 − q10)` alongside, which a handful of crash days
cannot move.

| Series | n | Skewness | 95% CI | Kelly skew | 95% CI |
|---|--:|--:|---|--:|---|
{sk_md}

Moment skewness is negative at every horizon on the S&P 500, but its intervals are wide — a few
days carry it, and on the Nasdaq daily and the Fama-French monthly and quarterly series the
interval straddles zero. The quantile skew is small but negative too ({h['kelly_d']:+.3f} daily,
{h['kelly_m']:+.3f} monthly): the "many small steps up, a few big drops" picture would need a
*positive* quantile skew, and the tape does not show one. Skewness says the left tail is longer;
it says nothing about order. The order test of section 2 is the strong witness.

## 4. Mechanism — which world reproduces the asymmetry? *(simulated, {N_PATHS} paths each)*

Fitted on the S&P 500 daily log returns (percent), Student-t shocks, `arch`:

| Model | α | γ (leverage) | γ *t* | β | ν | Log-lik |
|---|--:|--:|--:|--:|--:|--:|
{fit_md}

Each model simulated at the tape's length ({h['n_sp']:,} days). Medians over paths; share =
simulated ÷ real (for the elevator ratio, on its log).

| World | TR sum | Daily skew | Monthly skew | Elevator ratio (10%) | Share of TR | Share of monthly skew | Share of log elevator | TR sum, 5-95% of paths |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{mc_md}

The leverage models do the work. The fitted GJR-GARCH reproduces **{gjr['share_tr']:.0%}** of the
reversibility statistic, {gjr['share_skew_m']:.0%} of the monthly skewness and
{gjr['share_elevator']:.0%} of the (reversed) elevator ratio — volatility peaks at the trough, so
the first leg up is fast. The symmetric GARCH and both i.i.d. worlds reproduce essentially none
of the order. The shuffle of the tape keeps its daily skewness and still produces no
irreversibility: **it is the order, driven by volatility, not the size of single days.**

## 5. Could you trade it?

Rules fixed before the run. *Fast-exit / slow-entry*: exit once the close is 5% below the peak
since entry; re-enter once it is 10% above the trough since exit. Symmetric twins: 5%/5% and
10%/10%. **One execution lag** (the signal at the close of *t* earns from *t+1*; a 2-day-lag row
is the robustness check). One-way cost on traded NAV: {h['cost_daily_bps']:.0f} bp daily,
{h['cost_monthly_bps']:.0f} bp monthly. Cash earns Ken French's T-bill, so the daily races end
{TRADE_END}. Sharpe ratios are **excess of cash, strategy vs buy-and-hold**; the difference has
a paired circular-block-bootstrap 95% interval and one-sided *p* (share of resamples ≤ 0).

| Tape | Rule | Sharpe gross | Sharpe net | Sharpe B&H | Net − B&H [95% CI] | *p* | Time in | Trades / yr | Max DD rule / B&H |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
{race_md}

On the two daily tapes the pre-registered rule loses to buy-and-hold by
**{h['trade_d_vs_bh']['sp500']:+.2f}** (S&P 500) and **{h['trade_d_vs_bh']['nasdaq']:+.2f}**
(Nasdaq) of Sharpe, net — and those tapes are price-only, which *flatters* a rule that spends
time in cash. On the 1926-2018 monthly total-return tape it wins by
**{h['trade_d_vs_bh']['ff']:+.2f}** (p = {h['trade_ff_p_vs_bh']:.3f}); after 1945 the edge is
{h['trade_ff_post45_d']:+.2f} (p = {h['trade_ff_post45_p']:.3f}). And the *symmetric* 10%/10% twin
does as well as the asymmetric rule ({h['trade_ff_sym10_d']:+.2f} on the monthly century,
{h['trade_sp_sym10_d']:+.2f} on the S&P 500), so whatever a drawdown rule earns, the stairs/elevator
asymmetry is not the source. The rule's genuine product is a shallower drawdown, which is
insurance, not alpha.

## Caveats

- **Price-only daily tapes.** The S&P 500 and Nasdaq series carry no dividends (≈ 2%/yr for the
  S&P 500 over the span). The leg and reversibility statistics barely notice a smooth 2% drift;
  the trading races do, and the omission favours the timing rules. The Fama-French tape is total
  return.
- **Few deep legs.** At 20% there are about ten legs per daily tape. The Monte Carlo nulls are
  built at the tape's own length, so the p-values account for it, but the 20% ratios are noisy.
- **Leg definitions are a choice.** First passage from a confirmed extreme measures "how fast is
  the first X%"; completed-leg speed measures "how steep is the whole move". They disagree on
  this tape, and the disagreement is the finding. Both are reported.
- **Sixth moments.** The Ramsey-Rothman statistic's variance involves sixth moments, which fat
  tails strain; hence the block bootstrap and the clipped-tail check, which agree.
- **Mechanism shares are medians** over {N_PATHS} simulated paths of one tape length; individual
  paths scatter widely (section 4's last column), so "reproduces 70%" means the real tape is a
  typical draw from the leverage world, not that 30% is left unexplained.
- **No options data.** The claim that the asymmetry is what index put skew prices is the
  literature's (Bates 2000; Bakshi, Kapadia & Madan 2003), not tested here.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1016-stairs-up-elevator-down](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    slim = {k: v for k, v in h.items() if k not in ("legs", "races", "sp_legs10",
                                                     "skew_table", "mechanism", "tr_table",
                                                     "tr_per_lag", "tr_per_lag_nq",
                                                     "tr_per_lag_ff", "provenance")}
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    print("##HEADLINE## " + json.dumps(clean(slim), default=float, allow_nan=False))


if __name__ == "__main__":
    main()
