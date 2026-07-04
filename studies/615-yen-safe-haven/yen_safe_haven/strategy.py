"""Strategy + inference for Study 615 — Yen Safe Haven.

The claim: **when stocks crash, the yen rallies** — years of yen-funded carry trades
(borrow at ~0% in Tokyo, buy anything that yields) unwind violently in risk-off, and the
scramble to buy back borrowed yen makes JPY the "free" crash hedge. Same folk question as
study 69 (gold), different asset and a sharper mechanism: not a store-of-value story but a
**positioning** story — the hedge only pays if there is a carry stock to unwind.

We measure it four complementary ways:

    * **Conditional (downside) beta.** Piecewise daily regression of the yen's return on
      SPY split into down-moves and up-moves: ``jpy = a + b_dn min(spy,0) + b_up max(spy,0)``.
      The safe-haven claim is ``b_dn < 0`` (SPY down => JPY up) — HAC/Newey-West t (daily
      serial correlation + vol clustering).
    * **Quintile ladder.** Sort SPY days (and months) into quintiles; the yen's mean
      return per bucket. The claim lives in Q1 (worst SPY days): Welch t of Q1 vs Q5.
    * **Event studies.** Cumulative SPY vs JPY over the named crash windows (2008,
      Aug-2015, COVID's two legs, the 2022 bear, Aug-2024) — dates hardcoded in
      ``data.EVENTS``, returns always computed from the tape.
    * **Regime split.** The mechanism test: the downside beta re-estimated (a) by the
      **carry-stock proxy** — the US 13-week bill splits the sample into fat-differential
      (carry attractive, stock big) vs ZIRP (differential ~0, nothing to unwind) — and
      (b) among SPY **down-months** split by the sign of the 10-year yield move: growth
      scares (yields down) vs rate shocks (yields up, the 2022 configuration).

Tradability charges the honest carry: a US investor holding a JPY sleeve (FXY-style)
earns the yen's spot return **minus** the USD bill rate they gave up (Japanese short
rates ~0 all sample, so the US bill is the full negative carry). The hedge is a STATIC sleeve
held in advance — there is no timing signal, hence no signal lag to apply; the one
execution convention (documented) is that event/quintile returns use same-close daily
returns, never intraday foresight. Excess-vs-excess throughout; JPY spot is price-only
and labeled as such (a JPY deposit adds ~0 for almost the whole sample).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def to_returns(tape: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: jpy = the YEN's return vs USD (= -USDJPY pct change), spy.

    Also carries the bill yield (irx, %/yr) and the 10y yield level (tnx, %) forward-
    filled onto return days. Rows need both jpy and spy.
    """
    usdjpy = tape["usdjpy"].dropna()
    jpy = -usdjpy.pct_change()
    spy = tape["spy"].dropna().pct_change()
    out = pd.DataFrame({"jpy": jpy, "spy": spy}).dropna()
    out["irx"] = tape["irx"].reindex(out.index).ffill()
    out["tnx"] = tape["tnx"].reindex(out.index).ffill()
    return out


def monthly_returns(tape: pd.DataFrame) -> pd.DataFrame:
    """Month-end to month-end simple returns (complete months only — tape is as-of'd)."""
    m = tape[["usdjpy", "spy"]].resample("ME").last()
    out = pd.DataFrame({"jpy": -m["usdjpy"].pct_change(), "spy": m["spy"].pct_change()})
    out["tnx_chg"] = tape["tnx"].resample("ME").last().diff()
    out["irx"] = tape["irx"].resample("ME").last()
    return out.dropna(subset=["jpy", "spy"])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def nw_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS with Newey-West (HAC) standard errors. X excludes the constant (added here).

    Returns betas (const first) and their HAC t-stats. ``lags`` defaults to the standard
    ``floor(4 (n/100)^(2/9))`` plug-in.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    Z = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.inv(Z.T @ Z)
    beta = XtX_inv @ (Z.T @ y)
    u = y - Z @ beta
    Zu = Z * u[:, None]
    S = Zu.T @ Zu
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Zu[L:].T @ Zu[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    ss_res = float(u @ u)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "t": t, "n": n, "lags": lags,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b), unequal variances."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The conditional beta (headline)
# --------------------------------------------------------------------------- #
def downside_beta(rets: pd.DataFrame) -> dict:
    """Piecewise daily regression jpy ~ min(spy,0) + max(spy,0), HAC t on each leg.

    ``beta_dn < 0`` with a robust t is the safe-haven signal: on SPY down-moves the yen
    moves the OTHER way. Also reports the plain (unconditional) beta.
    """
    spy = rets["spy"].values
    jpy = rets["jpy"].values
    dn = np.minimum(spy, 0.0)
    up = np.maximum(spy, 0.0)
    pw = nw_ols(jpy, np.column_stack([dn, up]))
    plain = nw_ols(jpy, spy)
    return {
        "n": pw["n"], "lags": pw["lags"],
        "beta_dn": float(pw["beta"][1]), "t_dn": float(pw["t"][1]),
        "beta_up": float(pw["beta"][2]), "t_up": float(pw["t"][2]),
        "beta_all": float(plain["beta"][1]), "t_all": float(plain["t"][1]),
        "r2": pw["r2"],
    }


def quintile_ladder(rets: pd.DataFrame) -> dict:
    """Mean JPY return per SPY-return quintile (Q1 = worst SPY bucket), in bps.

    Welch t of Q1 vs Q5 (crash days vs melt-up days) and of Q1 vs the middle 60%.
    """
    q = pd.qcut(rets["spy"], 5, labels=False)
    means, ns = [], []
    for k in range(5):
        v = rets.loc[q == k, "jpy"].values
        means.append(float(v.mean() * 1e4))
        ns.append(len(v))
    q1 = rets.loc[q == 0, "jpy"].values
    q5 = rets.loc[q == 4, "jpy"].values
    mid = rets.loc[q.isin([1, 2, 3]), "jpy"].values
    spy_q1 = float(rets.loc[q == 0, "spy"].mean() * 1e4)
    return {"means_bps": means, "ns": ns, "spy_q1_bps": spy_q1,
            "t_q1_vs_q5": welch_t(q1, q5), "t_q1_vs_mid": welch_t(q1, mid),
            "q1_bps": means[0], "q5_bps": means[4]}


def down_month_split(m: pd.DataFrame, thresh: float = 0.0) -> dict:
    """JPY mean in SPY down-months vs up-months (Welch t), plus the worst-decile months."""
    dn = m.loc[m["spy"] < thresh, "jpy"].values
    up = m.loc[m["spy"] >= thresh, "jpy"].values
    dec = m.nsmallest(max(len(m) // 10, 5), "spy")
    return {
        "n_dn": len(dn), "n_up": len(up),
        "dn_bps": float(dn.mean() * 1e4), "up_bps": float(up.mean() * 1e4),
        "t_dn_vs_up": welch_t(dn, up),
        "dec_n": len(dec), "dec_spy_pct": float(dec["spy"].mean() * 100),
        "dec_jpy_pct": float(dec["jpy"].mean() * 100),
        "dec_hit": float((dec["jpy"] > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Event studies
# --------------------------------------------------------------------------- #
def event_table(tape: pd.DataFrame, events: list | None = None) -> list[dict]:
    """Cumulative SPY and JPY returns over each hardcoded crash window."""
    events = events if events is not None else _data.EVENTS
    usdjpy = tape["usdjpy"].dropna()
    spy = tape["spy"].dropna()
    out = []
    for label, start, end in events:
        s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
        w_spy = spy[(spy.index >= s0) & (spy.index <= s1)]
        w_fx = usdjpy[(usdjpy.index >= s0) & (usdjpy.index <= s1)]
        if len(w_spy) < 2 or len(w_fx) < 2:
            continue
        spy_ret = float(w_spy.iloc[-1] / w_spy.iloc[0] - 1.0)
        jpy_ret = float(w_fx.iloc[0] / w_fx.iloc[-1] - 1.0)   # yen return = inverse USDJPY
        out.append({"label": label, "start": start, "end": end,
                    "spy_pct": spy_ret * 100, "jpy_pct": jpy_ret * 100,
                    "hedged": jpy_ret > 0})
    return out


# --------------------------------------------------------------------------- #
# Regime splits — the mechanism test
# --------------------------------------------------------------------------- #
def carry_regime_split(rets: pd.DataFrame, cut: float = 2.0) -> dict:
    """Downside beta re-estimated by carry-stock regime.

    ``irx >= cut`` (%/yr): a fat USD-JPY rate differential — carry attractive, a large
    yen-funded stock exists to unwind. ``irx < cut``: US ZIRP — the differential is ~0
    and there is (mechanically) little carry to unwind. The claim's own mechanism says
    the hedge should be strongest in the FAT regime.
    """
    hi = rets[rets["irx"] >= cut]
    lo = rets[rets["irx"] < cut]
    return {"cut": cut,
            "hi": downside_beta(hi), "n_hi": len(hi),
            "lo": downside_beta(lo), "n_lo": len(lo),
            "share_hi": len(hi) / len(rets)}


def yield_direction_split(m: pd.DataFrame) -> dict:
    """Among SPY down-months: JPY mean when 10y yields FELL vs ROSE (Welch t of the gap).

    Growth scare (stocks down, yields down => carry differential compressing) vs rate
    shock (stocks down, yields UP => the differential widens all the way down — the 2022
    configuration). The safe haven should live in the first column only.
    """
    dn = m[m["spy"] < 0].dropna(subset=["tnx_chg"])
    fell = dn.loc[dn["tnx_chg"] < 0, "jpy"].values
    rose = dn.loc[dn["tnx_chg"] >= 0, "jpy"].values
    return {"n_fell": len(fell), "n_rose": len(rose),
            "fell_bps": float(fell.mean() * 1e4), "rose_bps": float(rose.mean() * 1e4),
            "t_gap": welch_t(fell, rose)}


# --------------------------------------------------------------------------- #
# Tradability — what the static hedge sleeve costs
# --------------------------------------------------------------------------- #
def hedge_cost(rets: pd.DataFrame, expense_bps: float = 40.0) -> dict:
    """Annualised cost of holding the JPY sleeve permanently (the honest carry bill).

    Excess-vs-excess: the sleeve's excess return = JPY spot return (price-only; a JPY
    deposit adds ~0 for nearly the whole sample) MINUS the US bill yield the dollars
    would otherwise earn. ``expense_bps`` is the FXY-style ETF expense ratio (0.40%/yr),
    charged on top. The hedge is static — held in advance, no signal, no timing lag; no
    round-trip trading costs accrue beyond one entry (ignored, <1 bp/yr amortised).
    """
    jpy = rets["jpy"].values
    bill_daily = rets["irx"].values / 100.0 / TRADING_DAYS
    spot_ann = float(np.expm1(np.log1p(jpy).mean() * TRADING_DAYS)) * 100
    ex = jpy - bill_daily
    ex_ann = float(np.expm1(np.log1p(ex).mean() * TRADING_DAYS)) * 100
    net_ann = ex_ann - expense_bps / 100.0
    years = len(jpy) / TRADING_DAYS
    return {"spot_ann_pct": spot_ann, "excess_ann_pct": ex_ann,
            "net_ann_pct": net_ann, "expense_bps": expense_bps, "years": years,
            "avg_bill_pct": float(rets["irx"].mean())}
