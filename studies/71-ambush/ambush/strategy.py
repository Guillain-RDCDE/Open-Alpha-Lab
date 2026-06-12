"""The ambush book — long SPY only on confluence days, under a hard 1%/day risk budget.

The accounting walks one honest day, indexed by the day the position is *held*:

- at the close of *t−1* the signals (``signals.confluence``) and the size
  (:func:`sizing`) are known; the target goes on at that close — **one execution
  lag, applied here and only here** (``shift(1)``);
- during day *t* the position rides ``Close_{t−1} → Close_t``, unless the low
  breaches the stop (−1% of NAV), in which case the day's underlying return is the
  stop threshold itself — or the *open*, if the day gaps straight through it — and
  the book is flat until the next close;
- **costs are one-way × |traded|** (bench rule): the intraday stop exit and the
  close-of-day rebalance each pay ``spread_bps``; a position carried overnight pays
  CFD financing ``(rf_t + financing_spread/252)`` per trading day held (weekend
  nights fold into the trading-day grid — stated, not hidden);
- the account's idle cash earns rf, so the book's daily return is quoted
  **excess-of-cash**: ``pos·r_under − pos·(rf + financing) − costs``. Raced against
  SPY buy-and-hold *excess-of-cash* — never raw vs excess (bench rule).

Sizing (frozen in the pre-registration): ``w = min(12%/σ̂_ann, 1%/(2·σ̂_d), 2.0)`` —
study 16's vol-target convention, then the **1%-of-NAV daily budget** as the binding
clamp (a 2σ down day loses at most 1% of the account). The stop is the same 1%
written as a price, not a second budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import signals

TRADING_DAYS = 252

TARGET_VOL_ANN = 0.12
MAX_LEVERAGE = 2.0
DAILY_RISK = 0.01
RISK_Z = 2.0
VOL_WINDOW = 20

SPREAD_BPS = 1.0          # one-way, per unit of notional traded (~0.5-pt RT on US500 @ ~6000)
FINANCING_SPREAD = 0.025  # annualised CFD financing spread over the cash rate


def sizing(
    returns: pd.Series,
    target_vol_ann: float = TARGET_VOL_ANN,
    max_leverage: float = MAX_LEVERAGE,
    daily_risk: float = DAILY_RISK,
    z: float = RISK_Z,
    window: int = VOL_WINDOW,
) -> pd.Series:
    """Past-only exposure for day *t+1*, known at the close of *t* (no lag applied here).

    ``min(vol-target, risk-budget, leverage cap)`` — with σ̂ the trailing ``window``-day
    std of close-to-close returns through *t*. Days without a full window read 0 (no
    trade before the estimator exists).
    """
    sig = returns.rolling(window).std(ddof=1)
    w_vol = (target_vol_ann / np.sqrt(TRADING_DAYS)) / sig
    w_risk = daily_risk / (z * sig)
    return np.minimum(w_vol, w_risk).clip(upper=max_leverage).fillna(0.0).rename("w")


def book(
    spy: pd.DataFrame,
    vix: pd.Series,
    rf: pd.Series,
    k: int = 3,
    spread_bps: float = SPREAD_BPS,
    financing_spread: float = FINANCING_SPREAD,
    use_stop: bool = True,
    **size_kw,
) -> pd.DataFrame:
    """Daily ledger of the K-confluence book. Columns:

    ``pos`` exposure held during the day · ``r_under`` underlying return credited
    (stop-aware) · ``gross`` position P&L · ``fin`` financing · ``cost`` spread paid ·
    ``net_excess`` the headline excess-of-cash return · ``stopped`` the stop fired.
    """
    conf = signals.confluence(spy, vix)
    r_cc = spy["Close"].pct_change()
    target = (conf["count"] >= k).astype(float) * sizing(r_cc, **size_kw)

    pos = target.shift(1).fillna(0.0)                       # the one lag
    prev_close = spy["Close"].shift(1)
    r_open = spy["Open"] / prev_close - 1.0
    r_low = spy["Low"] / prev_close - 1.0

    with np.errstate(divide="ignore"):
        thr = np.where(pos > 0, -DAILY_RISK / pos, -np.inf)
    stopped = use_stop & (pos > 0) & (r_low.values <= thr)
    r_under = pd.Series(
        np.where(stopped, np.where(r_open.values <= thr, r_open.values, thr), r_cc.values),
        index=spy.index,
        name="r_under",
    )

    pos_eod = pos.where(~stopped, 0.0)                      # what survives to the close
    rebalance = (target - pos_eod).abs()                    # trade at the close of t
    stop_exit = pos.where(stopped, 0.0)                     # trade at the stop, intraday
    cost = (spread_bps * 1e-4) * (rebalance + stop_exit)

    rf_d = rf.reindex(spy.index).ffill().fillna(0.0)
    fin = pos * (rf_d + financing_spread / TRADING_DAYS)

    gross = pos * r_under
    out = pd.DataFrame(
        {
            "count": conf["count"],
            "pos": pos,
            "r_under": r_under,
            "gross": gross,
            "fin": fin,
            "cost": cost,
            "net_excess": gross - fin - cost,
            "stopped": stopped,
        }
    )
    return out.iloc[1:]  # day 0 has no prior close


def bh_excess(spy: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """SPY buy-and-hold, excess-of-cash — the fair benchmark for an excess book."""
    r = spy["Close"].pct_change().iloc[1:]
    rf_d = rf.reindex(r.index).ffill().fillna(0.0)
    return (r - rf_d).rename("bh_excess")


def summary(excess: pd.Series, ledger: pd.DataFrame | None = None) -> dict:
    """Annualised excess Sharpe, excess return, drawdown of the excess equity line,
    plus (when the ledger is given) time-in-market and round-trips per year."""
    r = pd.Series(excess).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "ann_excess", "max_drawdown", "n")}
    sd = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    out = {
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "ann_excess": float(r.mean() * TRADING_DAYS),
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
        "n": int(len(r)),
    }
    if ledger is not None:
        led = ledger.loc[r.index]
        held = led["pos"] > 0
        entries = held & ~held.shift(1, fill_value=False)
        years = len(led) / TRADING_DAYS
        out["time_in_market"] = float(held.mean())
        out["trades_per_year"] = float(entries.sum() / years)
        out["stops_per_year"] = float(led["stopped"].sum() / years)
    return out


def lift_table(spy: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """The signal exhibit, before any overlay: next-day raw SPY close-to-close return
    by confluence count at today's close — H₁ is a monotone climb of this column."""
    conf = signals.confluence(spy, vix)
    nxt = spy["Close"].pct_change().shift(-1)
    rows = []
    for c in range(5):
        sel = nxt[conf["count"] == c].dropna()
        rows.append(
            {
                "count": c,
                "next_bps": float(sel.mean() * 1e4) if len(sel) else np.nan,
                "n": int(len(sel)),
                "share": float(len(sel) / len(nxt.dropna())),
            }
        )
    return pd.DataFrame(rows).set_index("count")


def armed_stream(spy: pd.DataFrame, vix: pd.Series, k: int = 3) -> pd.Series:
    """Next-day raw SPY returns on the days the K-confluence is armed (gross, unlevered)
    — the series the pre-registered HAC *t* is computed on."""
    conf = signals.confluence(spy, vix)
    nxt = spy["Close"].pct_change().shift(-1)
    return nxt[conf["count"] >= k].dropna().rename(f"armed_k{k}")


def premium_change(spy: pd.DataFrame, vix: pd.Series, k: int = 3, split: str = "2015") -> dict:
    """Welch-style decay test (study 42's construction): did the armed-day premium
    over the rest shrink across ``split``?"""
    conf = signals.confluence(spy, vix)
    nxt = spy["Close"].pct_change().shift(-1)
    armed = conf["count"] >= k
    out: dict = {"split": split}
    d, se = {}, {}
    for name, sel in (
        ("pre", nxt.index < pd.Timestamp(split)),
        ("post", nxt.index >= pd.Timestamp(split)),
    ):
        a = nxt[sel & armed].dropna()
        rest = nxt[sel & ~armed].dropna()
        d[name] = a.mean() - rest.mean()
        se[name] = float(np.sqrt(a.var(ddof=1) / len(a) + rest.var(ddof=1) / len(rest)))
        out[f"premium_{name}_bp"] = float(d[name] * 1e4)
        out[f"welch_t_{name}"] = float(d[name] / se[name]) if se[name] > 0 else np.nan
        out[f"n_armed_{name}"] = int(len(a))
    denom = float(np.sqrt(se["pre"] ** 2 + se["post"] ** 2))
    out["t_change"] = float((d["pre"] - d["post"]) / denom) if denom > 0 else np.nan
    return out


def variant_panel(spy: pd.DataFrame, vix: pd.Series, rf: pd.Series, ks=(1, 2, 3, 4)) -> pd.DataFrame:
    """Net excess streams for every K in the announced family — the Reality-Check input."""
    return pd.DataFrame({f"K{k}": book(spy, vix, rf, k=k)["net_excess"] for k in ks})


def cost_sweep(
    spy: pd.DataFrame, vix: pd.Series, rf: pd.Series, k: int = 3, grid=(0, 1, 2, 3, 5, 7, 10)
) -> pd.DataFrame:
    """Net excess Sharpe as the one-way spread climbs — where does the edge die?"""
    rows = []
    for bps in grid:
        led = book(spy, vix, rf, k=k, spread_bps=float(bps))
        rows.append({"spread_bps": bps, "net_sharpe": summary(led["net_excess"])["sharpe"]})
    return pd.DataFrame(rows).set_index("spread_bps")
