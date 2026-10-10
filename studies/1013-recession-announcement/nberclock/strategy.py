"""It's Official — the measurement, the inference and the rules. Study 1013.

The claim has three moving parts and each gets its own instrument:

- **The market leads the committee.** ``lead_lag`` finds the market's own peak and trough
  around every NBER cycle and measures, in months, how far ahead of the official reference dates
  they fell. ``sign_test`` asks whether "the market turns first" happens more often than a coin
  would allow.
- **By the announcement, the damage is done.** ``announcement_table`` measures, at every official
  announcement, the drawdown already realised (peak announcements) or the rally already missed
  (trough announcements), and the forward returns that followed.
- **So the announcement is a buy signal.** ``rotation_test`` compares the forward return after
  the real announcement dates with the same statistic after *every circular shift* of the whole
  set of dates. Shifting the dates as a block keeps their spacing and clustering — two
  announcements six months apart stay six months apart, so their overlapping forward windows
  stay overlapping — and keeps the market's own autocorrelation intact. It is an exact,
  seed-free randomisation test. ``power_curve`` runs the same test on the synthetic world to
  show how large an effect twelve events could detect at all.

Then the trade: ``signal_avoid`` (bills from the peak announcement to the trough announcement,
the naive "it's official, get out" rule the claim predicts will lose) and ``signal_buy_after``
(equity for a year after each announcement, bills otherwise — the claim's own rule), booked by
``timing_backtest`` with **one execution lag** (an announcement made during bar *t* is acted on
at the close of *t* and earns bar *t+1* onward — one ``shift``, applied once, here) and costs
charged one-way × traded NAV.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from quantlab.analytics import mean_tstat_hac  # noqa: E402

from . import data  # noqa: E402


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Whole calendar months from ``a`` to ``b`` (positive when ``b`` is later)."""
    return int((b.year - a.year) * 12 + (b.month - a.month))


def bar_position(index: pd.DatetimeIndex, date) -> int | None:
    """Position of the bar on which ``date`` is first known: the first bar dated >= ``date``.

    Daily tape: the announcement day itself. Monthly tape (month-end labels): the month that
    contains the announcement. The signal is acted on at that bar's close and earns from the
    *next* bar — the study's single execution lag. ``None`` if the date is past the tape.
    """
    d = pd.Timestamp(date)
    if len(index) == 0 or d < index[0] - pd.offsets.MonthBegin(1):
        return None
    pos = int(index.searchsorted(d, side="left"))
    if pos == 0 and d < index[0] - pd.Timedelta(days=7):
        return None
    return pos if pos < len(index) else None


def excess_index(monthly: pd.DataFrame) -> pd.Series:
    """Wealth of the market relative to rolling one-month bills (``∏(1+mkt)/∏(1+rf)``)."""
    return ((1.0 + monthly["mkt"]).cumprod() / (1.0 + monthly["rf"]).cumprod()).rename("xs")


def sign_test(x) -> dict:
    """Two-sided exact binomial test that positive and negative values are equally likely.

    Ties (zeros) are dropped, as in the textbook sign test.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    n = pos + neg
    p = float(sps.binomtest(pos, n, 0.5).pvalue) if n > 0 else np.nan
    return {"n_pos": pos, "n_neg": neg, "n_zero": int((x == 0).sum()), "p": p}


# --------------------------------------------------------------------------- #
# 1. Lead / lag: where are the market's own turning points?
# --------------------------------------------------------------------------- #
def lead_lag(level: pd.Series, cycles: pd.DataFrame, pre_months: int = 24,
             peak_slack: int = 3, post_months: int = 12) -> pd.DataFrame:
    """The market's own peak and trough around each NBER cycle, and how far they lead.

    ``level`` is a month-end wealth index. For each cycle with NBER peak *P* and trough *T*:

    - **market peak** = the highest month-end in ``[P − pre_months, P + peak_slack]`` (the
      market may top a little *after* the NBER peak, as in September 1929);
    - **market trough** = the lowest month-end in ``[market peak, T + post_months]``.

    Both windows are fixed in advance and symmetric in spirit: the market is allowed to lag
    the committee, and when it does the lead comes out negative rather than being hidden.

    ``lead_peak`` = months from the market peak to *P* (positive = the market topped first);
    ``lead_trough`` likewise. ``dd_at_nber_peak`` is the drawdown from the market peak already
    on the books when the NBER peak month closed, ``dd_at_nber_trough`` the same at the NBER
    trough, ``max_dd`` the full peak-to-trough fall. A cycle whose look-back window starts
    before the tape is flagged ``censored`` (only 1926 here).
    """
    rows = []
    idx = level.index
    for _, c in cycles.iterrows():
        P, T = pd.Timestamp(c["peak"]), pd.Timestamp(c["trough"])
        start = P - pd.DateOffset(months=pre_months)
        end = T + pd.DateOffset(months=post_months)
        if P < idx[0] or T > idx[-1]:
            continue
        w1 = level[(idx >= start) & (idx <= P + pd.DateOffset(months=peak_slack)
                                     + pd.offsets.MonthEnd(0))]
        mpk = w1.idxmax()
        w2 = level[(idx >= mpk) & (idx <= end + pd.offsets.MonthEnd(0))]
        mtr = w2.idxmin()
        hi = float(level.loc[mpk])
        lvl_p = float(level.asof(P))
        lvl_t = float(level.asof(T))
        rows.append({
            "peak": P, "trough": T, "nber_months": months_between(P, T),
            "mkt_peak": mpk, "mkt_trough": mtr,
            "lead_peak": months_between(mpk, P), "lead_trough": months_between(mtr, T),
            "mkt_bear_months": months_between(mpk, mtr),
            "dd_at_nber_peak": lvl_p / hi - 1.0,
            "dd_at_nber_trough": lvl_t / hi - 1.0,
            "max_dd": float(level.loc[mtr]) / hi - 1.0,
            "rally_by_nber_trough": lvl_t / float(level.loc[mtr]) - 1.0
            if mtr <= T else 0.0,
            "censored": bool(start < idx[0]),
        })
    return pd.DataFrame(rows)


def lead_lag_all(tr_index: pd.Series, daily_price: pd.Series,
                 cycles: pd.DataFrame) -> pd.DataFrame:
    """Every NBER cycle on the best tape that covers it, labelled.

    Cycles inside the Fama-French window (to 2018-11) are measured on the monthly
    **total-return** index; the 2020 cycle, which only the skfolio tape reaches, on month-end
    closes of the daily **price** index. The ``tape`` column says which.
    """
    a = lead_lag(tr_index, cycles)
    a["tape"] = "FF total return"
    later = cycles[cycles["trough"] > tr_index.index[-1]]
    if len(later) and daily_price is not None:
        b = lead_lag(data.daily_to_monthly(daily_price), later)
        b["tape"] = "S&P 500 price"
        a = pd.concat([a, b], ignore_index=True)
    return a


def lead_summary(ll: pd.DataFrame, min_fall: float | None = None) -> dict:
    """Sign tests and medians of the leads, on uncensored cycles (optionally only real bears)."""
    d = ll[~ll["censored"]]
    if min_fall is not None:
        d = d[d["max_dd"] <= -abs(min_fall)]
    st_t, st_p = sign_test(d["lead_trough"]), sign_test(d["lead_peak"])
    return {"n": int(len(d)),
            "trough_pos": st_t["n_pos"], "trough_neg": st_t["n_neg"],
            "trough_zero": st_t["n_zero"], "trough_p": st_t["p"],
            "trough_median": float(d["lead_trough"].median()),
            "trough_mean": float(d["lead_trough"].mean()),
            "peak_pos": st_p["n_pos"], "peak_neg": st_p["n_neg"], "peak_zero": st_p["n_zero"],
            "peak_p": st_p["p"], "peak_median": float(d["lead_peak"].median()),
            "peak_mean": float(d["lead_peak"].mean())}


def _turns_np(x: np.ndarray, p: int, t: int, pre: int, slack: int, post: int):
    """Positions of the market peak and trough for one cycle (numpy twin of ``lead_lag``)."""
    a0, a1 = max(p - pre, 0), min(p + slack, len(x) - 1)
    mpk = a0 + int(np.argmax(x[a0:a1 + 1]))
    b1 = min(t + post, len(x) - 1)
    mtr = mpk + int(np.argmin(x[mpk:b1 + 1]))
    return mpk, mtr


def lead_rotation_test(level: pd.Series, cycles: pd.DataFrame, pre_months: int = 24,
                       peak_slack: int = 3, post_months: int = 12) -> dict:
    """Is the measured lead bigger than the *estimator* produces on arbitrary dates?

    The windows in ``lead_lag`` are not neutral. The market trough is searched for *after* the
    market peak, and with an upward-drifting market the lowest point after a high tends to come
    soon after it — so even a calendar of random dates shows the "market trough" arriving before
    the "NBER trough" more often than half the time. (The synthetic null caught this; a sign
    test against 50% would have been too generous.)

    So the honest null is the same estimator on the same tape with the whole NBER calendar
    shifted by ``k`` months, for every ``k`` that keeps the shifted cycles inside the tape
    (positions wrap modulo the tape; a cycle whose shifted window would cross the wrap is
    dropped for that shift). Statistics: the share of cycles with the market trough first and
    the mean trough lead. One-sided p = share of shifts at least as extreme as the real calendar.
    """
    x = np.asarray(level, dtype=float)
    n = len(x)
    idx = level.index
    P = np.array([idx.searchsorted(c) for c in cycles["peak"]])
    T = np.array([idx.searchsorted(c) for c in cycles["trough"]])
    ok0 = (P - pre_months >= 0) & (T + post_months < n)
    P, T = P[ok0], T[ok0]
    share, mean_lead, n_used = [], [], []
    for k in range(n):
        ps, ts = (P + k) % n, (T + k) % n
        good = (ps - pre_months >= 0) & (ts + post_months < n) & (ts > ps)
        if good.sum() < max(3, len(P) // 2):
            continue
        leads = []
        for p, t in zip(ps[good], ts[good]):
            _, mtr = _turns_np(x, int(p), int(t), pre_months, peak_slack, post_months)
            leads.append(int(t) - mtr)
        leads = np.array(leads)
        share.append(float(np.mean(leads > 0)))
        mean_lead.append(float(np.mean(leads)))
        n_used.append(int(good.sum()))
    share, mean_lead = np.array(share), np.array(mean_lead)
    return {"observed_share": float(share[0]), "observed_mean_lead": float(mean_lead[0]),
            "null_share_mean": float(share.mean()), "null_mean_lead": float(mean_lead.mean()),
            "p_share": float(np.mean(share >= share[0] - 1e-12)),
            "p_mean_lead": float(np.mean(mean_lead >= mean_lead[0] - 1e-12)),
            "n_shifts": int(len(share)), "n_cycles": int(len(P)),
            "perm_share": share, "perm_mean_lead": mean_lead}


# --------------------------------------------------------------------------- #
# 2. The announcements
# --------------------------------------------------------------------------- #
def announcement_table(level: pd.Series, anns: pd.DataFrame, horizons=(3, 6, 12, 24),
                       bars_per_month: float = 1.0, pre_months: int = 24,
                       trough_pre_months: int = 12, post_months: int = 18) -> pd.DataFrame:
    """Per announcement: what was already done, and what came next.

    ``level`` is any wealth index (monthly excess-of-bills, monthly total return or daily
    price). ``horizons`` are in **months**, converted to bars by ``bars_per_month`` (1 on the
    monthly tape, 21 on the daily). The announcement is known at the close of its bar (see
    ``bar_position``); forward returns run from that close.

    Peak announcements: ``already`` = drawdown from the highest level in the ``pre_months``
    before the reference month up to the announcement bar (the fall already realised);
    ``still_to_come`` = further fall from the announcement to the lowest level in the
    following ``post_months``; ``share_done`` = the fraction of the cycle's total fall that was
    already on the books.

    Trough announcements: ``already`` = rally from the lowest level in the window
    ``[reference − trough_pre_months, announcement]`` (the rebound already missed) and
    ``months_since_extreme`` = how long ago that low was.

    An event whose look-back window starts before the tape is kept for its forward returns but
    flagged ``censored`` and its ``already`` set to NaN — a truncated history would understate
    the fall.
    """
    idx = level.index
    rows = []
    for _, a in anns.iterrows():
        pos = bar_position(idx, a["date"])
        if pos is None:
            continue
        ref = pd.Timestamp(a["reference"])
        look = pre_months if a["kind"] == "peak" else trough_pre_months
        start = ref - pd.DateOffset(months=look)
        censored = bool(start < idx[0] - pd.DateOffset(months=1))
        d0 = idx[pos]
        x0 = float(level.iloc[pos])
        r = {"kind": a["kind"], "reference": ref, "date": pd.Timestamp(a["date"]), "bar": d0,
             "lag_months": months_between(ref, pd.Timestamp(a["date"])), "censored": censored}
        hist = level[(idx >= start) & (idx <= d0)]
        if a["kind"] == "peak":
            hi_d = hist.idxmax()
            hi = float(hist.max())
            fut = level[(idx >= d0) & (idx <= d0 + pd.DateOffset(months=post_months))]
            lo = float(fut.min())
            r["already"] = x0 / hi - 1.0
            r["still_to_come"] = lo / x0 - 1.0
            total = lo / hi - 1.0
            r["total_fall"] = total
            # a "share of the fall" is meaningless when there was no real fall (< 2%)
            r["share_done"] = (r["already"] / total) if total < -0.02 else np.nan
            r["months_since_extreme"] = months_between(hi_d, d0)
        else:
            lo_d = hist.idxmin()
            lo = float(hist.min())
            r["already"] = x0 / lo - 1.0
            r["still_to_come"] = np.nan
            r["total_fall"] = np.nan
            r["share_done"] = np.nan
            r["months_since_extreme"] = months_between(lo_d, d0)
        if censored:
            for k in ("already", "still_to_come", "total_fall", "share_done",
                      "months_since_extreme"):
                r[k] = np.nan
        for hm in horizons:
            hb = int(round(hm * bars_per_month))
            r[f"fwd_{hm}m"] = (float(level.iloc[pos + hb]) / x0 - 1.0
                               if pos + hb < len(level) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def best_available(at_monthly: pd.DataFrame, at_daily: pd.DataFrame) -> pd.DataFrame:
    """One row per announcement: the monthly total-return row when it exists and is uncensored,
    otherwise the daily price row. ``tape`` says which — the 2020 pair is price-only."""
    m = at_monthly.copy()
    m["tape"] = "FF total return"
    d = at_daily.copy()
    d["tape"] = "S&P 500 price"
    keep = m[~m["censored"]]
    rest = d[~d["date"].isin(keep["date"])]
    out = pd.concat([keep, rest], ignore_index=True).sort_values("date")
    return out.reset_index(drop=True)


def forward_returns(level: pd.Series, h_bars: int) -> np.ndarray:
    """``level[i + h] / level[i] − 1`` for every bar with a full forward window."""
    x = np.asarray(level, dtype=float)
    return x[h_bars:] / x[:-h_bars] - 1.0


def rotation_test(level: pd.Series, positions, h_bars: int) -> dict:
    """Exact circular-shift randomisation test of the mean forward return after events.

    The statistic is the mean ``h_bars``-ahead return from the event bars. The null
    distribution is that statistic after **every** circular shift ``k`` of the whole set of
    event positions over the ``M = N − h`` bars that have a full forward window
    (``(p + k) mod M``). This is the "random announcement dates" null, made block-aware: the
    events keep their spacing (and so their overlap), the market keeps its own serial
    dependence, and the test needs no seed. One-sided p = share of shifts whose statistic is at
    least the observed one (the observed shift included, so p ≥ 1/M).
    """
    f = forward_returns(level, h_bars)
    M = len(f)
    pos = np.array([p for p in positions if p is not None and p < M], dtype=int)
    if len(pos) == 0:
        return {}
    shifts = np.arange(M)
    perm = f[(pos[None, :] + shifts[:, None]) % M].mean(axis=1)
    obs = float(perm[0])
    null_mean = float(perm.mean())
    null_sd = float(perm.std(ddof=1))
    return {"n_events": int(len(pos)), "h_bars": int(h_bars), "observed": obs,
            "null_mean": null_mean, "null_sd": null_sd,
            "excess_vs_random": obs - null_mean,
            "z": (obs - null_mean) / null_sd if null_sd > 0 else np.nan,
            "p_one_sided": float(np.mean(perm >= obs - 1e-15)),
            "p_lower": float(np.mean(perm <= obs + 1e-15)),
            "n_shifts": int(M), "event_returns": f[pos].tolist(), "perm": perm}


def event_positions(index: pd.DatetimeIndex, anns: pd.DataFrame, kinds=("peak", "trough")):
    """Bar positions of the announcements of the given ``kinds`` that fall inside ``index``."""
    out = []
    for _, a in anns.iterrows():
        if a["kind"] in kinds:
            p = bar_position(index, a["date"])
            if p is not None:
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
# 3. Power: how big an effect can twelve events see?
# --------------------------------------------------------------------------- #
def power_curve(effects=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30), n_sims: int = 200,
                n_cycles: int = 5, n_years: int = 92, vol: float = 0.18,
                h_bars: int = 12, alpha: float = 0.05, seed: int = 1013) -> pd.DataFrame:
    """Rejection rate of ``rotation_test`` on synthetic worlds with a known planted effect.

    Each world is i.i.d. monthly returns at volatility ``vol`` with ``n_cycles`` NBER-style
    cycles (so ``2 × n_cycles`` announcements) and an abnormal log return of size ``effect``
    over the 12 months after every announcement — and nothing else (no planted lead, so the
    test is isolated). The row at ``effect = 0`` is the test's size; the other rows are its
    power. Matching ``vol``, ``n_years`` and the event count to the real tape makes the curve
    an honest answer to "could this study have seen it?".
    """
    rows = []
    for e in effects:
        rej, obs = 0, []
        for i in range(n_sims):
            w = data.synthetic_world(n_years=n_years, n_cycles=n_cycles, signal_strength=1.0,
                                     lead_months=0, bear_depth=0.0, announce_premium=e,
                                     vol=vol, seed=seed + 7919 * i)
            lvl = excess_index(w["returns"])
            pos = event_positions(lvl.index, w["announcements"])
            r = rotation_test(lvl, pos, h_bars)
            if not r:
                continue
            obs.append(r["excess_vs_random"])
            rej += int(r["p_one_sided"] < alpha)
        rows.append({"effect": float(e), "reject_rate": rej / max(n_sims, 1),
                     "mean_measured_excess": float(np.mean(obs)) if obs else np.nan,
                     "n_events": 2 * n_cycles, "n_sims": n_sims})
    return pd.DataFrame(rows).set_index("effect")


def minimum_detectable_effect(curve: pd.DataFrame, power: float = 0.8) -> float:
    """Smallest planted effect whose rejection rate reaches ``power`` (linear interpolation)."""
    x = curve.index.to_numpy(dtype=float)
    y = curve["reject_rate"].to_numpy(dtype=float)
    above = np.where(y >= power)[0]
    if len(above) == 0:
        return float("nan")
    j = int(above[0])
    if j == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    return float(x0 + (power - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)


# --------------------------------------------------------------------------- #
# 4. Rules: what a believer would actually do
# --------------------------------------------------------------------------- #
def signal_avoid(index: pd.DatetimeIndex, anns: pd.DataFrame) -> pd.Series:
    """1 = equity, 0 = bills from each peak announcement until the next trough announcement.

    The value at bar *t* is the position decided at *t*'s close; ``timing_backtest`` applies it
    to bar *t+1*. A trough announcement past the tape keeps the rule in bills to the end.
    """
    sig = pd.Series(1.0, index=index)
    a = anns.sort_values("date").reset_index(drop=True)
    for i, row in a.iterrows():
        if row["kind"] != "peak":
            continue
        p = bar_position(index, row["date"])
        if p is None:
            continue
        nxt = a[(a.index > i) & (a["kind"] == "trough")]
        q = bar_position(index, nxt["date"].iloc[0]) if len(nxt) else None
        q = len(index) if q is None else q
        sig.iloc[p:q] = 0.0
    return sig


def signal_buy_after(index: pd.DatetimeIndex, anns: pd.DataFrame, hold_bars: int,
                     kinds=("peak", "trough")) -> pd.Series:
    """1 = equity for ``hold_bars`` bars after each announcement of ``kinds``, else 0 (bills)."""
    sig = pd.Series(0.0, index=index)
    for p in event_positions(index, anns, kinds):
        sig.iloc[p:p + hold_bars] = 1.0
    return sig


def timing_backtest(ret: pd.Series, rf: pd.Series, signal: pd.Series, cost_bps: float = 10.0,
                    periods_per_year: int = 12) -> dict:
    """Book a long/bills switch with ONE execution lag and one-way costs on traded NAV.

    ``w_t = signal_{t−1}``: the position decided at the close of *t−1* earns bar *t* — the only
    shift in the study's pipeline. Gross return ``w·ret + (1−w)·rf``. Every switch trades 100%
    of NAV once (sell equity → bills, or back) and is charged ``cost_bps`` one-way on that NAV.
    The book starts in its first position at no cost. Sharpe is on returns **in excess of
    bills**, so a rule that sits in cash is never flattered by comparing its raw Sharpe with
    buy-and-hold's excess Sharpe.
    """
    w = signal.shift(1)
    w.iloc[0] = signal.iloc[0]
    w = w.reindex(ret.index).astype(float)
    gross = w * ret + (1.0 - w) * rf
    turnover = w.diff().abs().fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    out = {}
    for lbl, r in (("gross", gross), ("net", net)):
        xs = r - rf
        mu = float(xs.mean() * periods_per_year)
        sd = float(xs.std(ddof=1) * np.sqrt(periods_per_year))
        wealth = (1.0 + r).cumprod()
        out[lbl] = {"ann_excess": mu, "vol": sd,
                    "sharpe": mu / sd if sd > 0 else np.nan,
                    "cagr": float(wealth.iloc[-1] ** (periods_per_year / len(r)) - 1.0),
                    "max_dd": float((wealth / wealth.cummax() - 1.0).min())}
    out["exposure"] = float(w.mean())
    out["switches"] = int((turnover > 0).sum())
    out["turnover_per_year"] = float(turnover.sum() * periods_per_year / len(turnover))
    out["net_series"] = net
    out["gross_series"] = gross
    return out


def compare_rules(ret: pd.Series, rf: pd.Series, signals: dict, cost_bps: float = 10.0,
                  periods_per_year: int = 12, hac_lags: int = 12) -> pd.DataFrame:
    """Buy-and-hold against each rule: gross and net excess Sharpe, gain, active-return HAC t."""
    bh = timing_backtest(ret, rf, pd.Series(1.0, index=ret.index), cost_bps, periods_per_year)
    rows = [{"rule": "buy-and-hold", "exposure": 1.0, "switches": 0,
             "ann_excess_gross": bh["gross"]["ann_excess"],
             "ann_excess_net": bh["net"]["ann_excess"],
             "sharpe_gross": bh["gross"]["sharpe"], "sharpe_net": bh["net"]["sharpe"],
             "max_dd_net": bh["net"]["max_dd"], "cagr_net": bh["net"]["cagr"],
             "sharpe_gain_net": 0.0, "active_ann_net": 0.0, "active_hac_t": np.nan}]
    for name, sig in signals.items():
        b = timing_backtest(ret, rf, sig, cost_bps, periods_per_year)
        act = b["net_series"] - bh["net_series"]
        t = mean_tstat_hac(act, lags=hac_lags)["tstat"]
        rows.append({"rule": name, "exposure": b["exposure"], "switches": b["switches"],
                     "ann_excess_gross": b["gross"]["ann_excess"],
                     "ann_excess_net": b["net"]["ann_excess"],
                     "sharpe_gross": b["gross"]["sharpe"], "sharpe_net": b["net"]["sharpe"],
                     "max_dd_net": b["net"]["max_dd"], "cagr_net": b["net"]["cagr"],
                     "sharpe_gain_net": b["net"]["sharpe"] - bh["net"]["sharpe"],
                     "active_ann_net": float(act.mean() * periods_per_year),
                     "active_hac_t": float(t)})
    return pd.DataFrame(rows).set_index("rule")


def rule_rotation_test(ret: pd.Series, rf: pd.Series, signal: pd.Series,
                       cost_bps: float = 10.0, periods_per_year: int = 12,
                       step: int = 1) -> dict:
    """Net excess Sharpe of a rule against the same rule on circularly-shifted dates.

    "Does buying on the *official* dates beat buying on random dates with the same spacing?"
    Each shift rolls the whole signal by ``k`` bars (``np.roll``), so exposure, the number of
    switches and the cost bill are identical; only the timing moves. One-sided p = share of
    shifts with a Sharpe at least the observed one.
    """
    r = np.asarray(ret, dtype=float)
    f = np.asarray(rf, dtype=float)
    s = np.asarray(signal, dtype=float)
    n = len(r)
    shifts = np.arange(0, n, step)
    # Build all rolled signals at once: S[k, t] = s[(t - k) mod n]
    tt = np.arange(n)
    S = s[(tt[None, :] - shifts[:, None]) % n]
    W = np.concatenate([S[:, :1], S[:, :-1]], axis=1)   # one execution lag
    turn = np.abs(np.diff(W, axis=1, prepend=W[:, :1]))
    net = W * r[None, :] + (1.0 - W) * f[None, :] - turn * cost_bps / 1e4
    xs = net - f[None, :]
    sh = xs.mean(axis=1) / xs.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
    obs = float(sh[0])
    return {"observed_sharpe": obs, "null_mean": float(np.nanmean(sh)),
            "null_p95": float(np.nanpercentile(sh, 95)),
            "p_one_sided": float(np.mean(sh >= obs - 1e-12)), "n_shifts": int(len(shifts)),
            "perm": sh}


def cost_sweep(ret: pd.Series, rf: pd.Series, signal: pd.Series, costs=(0, 5, 10, 25, 50),
               periods_per_year: int = 12) -> pd.DataFrame:
    """Net excess Sharpe gain over buy-and-hold across one-way cost levels (bps)."""
    rows = []
    for c in costs:
        bh = timing_backtest(ret, rf, pd.Series(1.0, index=ret.index), c, periods_per_year)
        b = timing_backtest(ret, rf, signal, c, periods_per_year)
        rows.append({"cost_bps": c, "sharpe_net": b["net"]["sharpe"],
                     "bh_sharpe": bh["net"]["sharpe"],
                     "gain": b["net"]["sharpe"] - bh["net"]["sharpe"]})
    return pd.DataFrame(rows).set_index("cost_bps")


# --------------------------------------------------------------------------- #
# 5. Verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    The claim has two legs and they are stamped as legs:

    - **Lead leg** (uncensored cycles only) is *real* when the market trough precedes the NBER
      trough in significantly more cycles than a coin allows (exact sign test p < 0.05), the
      median lead is ≥ 1 month, **and** the share of cycles with the market first beats the
      same estimator run on every month-shift of the NBER calendar (``lead_rotation_test``,
      p < 0.05). The last clause was added after the synthetic null showed the estimator
      itself leans towards "market first" (see ``lead_rotation_test``); it only makes the bar
      higher, and it was fixed before the real tape was read with it.
    - **Buy-signal leg** is *real* when the mean 12-month forward excess return after the
      official announcements beats every-circular-shift random dates at one-sided p < 0.05 on
      the monthly total-return tape; *weak* when p < 0.20, or when either announcement kind
      alone clears 0.05.

    **Signal**: Real if both legs are real; **Mixed** if exactly one is (the split is spelled
    out); Weak if neither is real but the buy leg is weak; None otherwise.

    **Tradability**: **Investable** only if the buy leg is real, the claim's rule ("equity for a
    year after each announcement") beats buy-and-hold net of costs by ≥ 0.10 excess Sharpe with
    an active-return HAC t ≥ 2 and beats its own shifted-date versions at p < 0.05, and is ahead
    on the daily tape too. **Fragile** if the rule's net Sharpe gain is positive on BOTH tapes
    with HAC t ≥ 1. **Mirage** otherwise.
    """
    lead_real = ((h["lead_trough_sign_p"] < 0.05) and (h["lead_trough_median"] >= 1)
                 and (h["lead_rot_p_share"] < 0.05))
    buy_real = h["buy_p_pooled"] < 0.05
    buy_weak = (h["buy_p_pooled"] < 0.20) or (min(h["buy_p_peak"], h["buy_p_trough"]) < 0.05)
    if lead_real and buy_real:
        signal = "Real"
    elif lead_real != buy_real:
        signal = "Mixed"
    elif buy_weak:
        signal = "Weak"
    else:
        signal = "None"
    g_m, g_d, t_m = h["rule_gain_monthly"], h["rule_gain_daily"], h["rule_hac_t_monthly"]
    if (buy_real and g_m >= 0.10 and t_m >= 2 and h["rule_perm_p_monthly"] < 0.05
            and g_d > 0):
        trad = "Investable"
    elif g_m > 0 and g_d > 0 and t_m >= 1:
        trad = "Fragile"
    else:
        trad = "Mirage"
    buy_word = "real" if buy_real else ("weak" if buy_weak else "not there")
    lead_word = "real" if lead_real else "not significant"
    opener = ("Both halves of the claim hold. " if signal == "Real" else
              "The two halves of the claim split. " if signal == "Mixed" else
              "Neither half of the claim clears the bar. ")
    signal_why = (
        f"{opener}**The lead is {lead_word}:** across the {h['n_cycles_lead']} uncensored NBER "
        f"cycles since 1926 the market's own trough came before the official trough in "
        f"**{h['lead_trough_pos']} of {h['n_cycles_lead']}** (exact sign test "
        f"p = {h['lead_trough_sign_p']:.4f}), by a median **{h['lead_trough_median']:.0f} "
        f"months**; the market peak came first in {h['lead_peak_pos']} of {h['n_cycles_lead']} "
        f"(median lead {h['lead_peak_median']:.0f} months). That is not automatic: the same "
        f"turning-point rule run on every month-shift of the NBER calendar puts the market first "
        f"{h['lead_rot_null_share']:.0%} of the time, and the real calendar beats that at "
        f"p = {h['lead_rot_p_share']:.3f} — although the *size* of the lead "
        f"(mean {h['lead_rot_obs_mean']:.1f} months) is no bigger than on shifted dates "
        f"(p = {h['lead_rot_p_mean']:.2f}). By the time a peak was *announced*, "
        f"the median drawdown already on the books was **{h['dd_already_median']:.0%}**; by the "
        f"time a trough was announced, the market stood a median **{h['missed_median']:+.0%}** "
        f"above its low. **The buy signal is {buy_word}:** the 12-month excess return after the "
        f"{h['n_events_monthly']} announcements on the total-return tape averaged "
        f"{h['buy_obs_pooled']:+.1%} against {h['buy_null_pooled']:+.1%} after every circular "
        f"shift of the same dates (rotation p = {h['buy_p_pooled']:.2f}; peaks alone "
        f"p = {h['buy_p_peak']:.2f}, troughs alone p = {h['buy_p_trough']:.2f}; daily price "
        f"tape, {h['n_events_daily']} events, p = {h['buy_p_pooled_daily']:.2f}). The test is "
        f"close to blind by construction: with {h['n_events_monthly']} events it needs a planted "
        f"**{h['mde80']:.0%}** abnormal 12-month return to reach 80% power.")
    beat = "beat" if g_m > 0 else "trailed"
    trad_why = (
        f"The claim's own rule — equity for a year after each announcement, bills otherwise, one "
        f"execution lag, {h['cost_bps']:.0f} bp one-way — {beat} buy-and-hold: net excess "
        f"Sharpe {h['rule_sharpe_monthly']:.2f} vs {h['bh_sharpe_monthly']:.2f} on the monthly "
        f"total-return tape 1980–2018 (gain {g_m:+.2f}, active-return HAC t = {t_m:+.2f}; "
        f"against the same rule on shifted dates, rotation p = {h['rule_perm_p_monthly']:.2f}), "
        f"and {g_d:+.2f} on the daily price tape 1990–2022. It sits in equities only "
        f"{h['rule_exposure']:.0%} of the time, so it forgoes most of the premium buy-and-hold "
        f"collects. The opposite reflex — \"it's official, sit in bills until the recovery is "
        f"declared\" — changed net excess Sharpe by {h['avoid_gain_monthly']:+.2f} (monthly) and "
        f"{h['avoid_gain_daily']:+.2f} (daily). Costs are not what decides it: {h['cost_bps']:.0f} "
        f"bp on a handful of switches a decade is noise next to the timing.")
    if signal in ("Mixed", "Real") and lead_real:
        one = (f"The market really does turn before the NBER says so — a median "
               f"{h['lead_trough_median']:.0f}-month lead at the trough, and a median "
               f"{h['missed_median']:+.0%} rebound already banked by the day the recovery is "
               f"declared — but the announcement itself is no buy signal "
               f"(rotation p = {h['buy_p_pooled']:.2f}), and the claim's own trading rule "
               f"{'beat' if g_m > 0 else 'trailed'} buy-and-hold.")
    else:
        one = (f"The NBER's announcements carry no detectable timing information for stocks "
               f"(rotation p = {h['buy_p_pooled']:.2f}), and the rule built on them "
               f"{'beat' if g_m > 0 else 'trailed'} buy-and-hold.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one, "lead_real": lead_real, "buy_real": buy_real,
            "buy_weak": buy_weak}
