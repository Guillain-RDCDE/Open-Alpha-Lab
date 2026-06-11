"""The omen, and the only fair way to score it.

The January Barometer makes a *directional* claim: sign(January) predicts sign(rest-of-year). The
honest test is not "is the rest-of-year up after an up January" (it almost always is — stocks rise
most years) but "does conditioning on January beat the **base rate** of just predicting up?" Plus a
tradable version (sit in cash for the rest of the year after a down January) measured against simply
holding the index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annual_table(monthly_ret: pd.Series) -> pd.DataFrame:
    """Collapse a monthly-return series into yearly ``(jan, roy)`` pairs.

    ``jan`` = the January return; ``roy`` = the compounded February-through-December return. Years
    without a January or with fewer than six months are dropped, as is the final partial year.
    """
    df = pd.DataFrame({"ret": pd.Series(monthly_ret).astype(float)})
    df["year"] = df.index.year
    df["month"] = df.index.month
    rows = []
    for y, g in df.groupby("year"):
        if 1 in set(g["month"]) and len(g) >= 6 and g["month"].max() >= 11:
            jan = g.loc[g["month"] == 1, "ret"].iloc[0]
            roy = (1.0 + g.loc[g["month"] >= 2, "ret"]).prod() - 1.0
            rows.append((y, jan, roy))
    return pd.DataFrame(rows, columns=["year", "jan", "roy"]).set_index("year")


def base_rate(tbl: pd.DataFrame) -> float:
    """The unconditional P(rest-of-year > 0) — the bar the omen has to clear."""
    return float((tbl["roy"] > 0).mean())


def barometer_accuracy(tbl: pd.DataFrame) -> float:
    """Directional accuracy: how often sign(January) matches sign(rest-of-year)."""
    return float((np.sign(tbl["jan"]) == np.sign(tbl["roy"])).mean())


def conditional_means(tbl: pd.DataFrame) -> pd.DataFrame:
    """Rest-of-year mean and P(up), split by the sign of January (the believers' headline split)."""
    out = {}
    for lab, mask in [("jan_up", tbl["jan"] > 0), ("jan_down", tbl["jan"] <= 0)]:
        s = tbl.loc[mask, "roy"]
        out[lab] = {"n": int(len(s)), "roy_mean": float(s.mean()), "p_up": float((s > 0).mean())}
    out["unconditional"] = {"n": int(len(tbl)), "roy_mean": float(tbl["roy"].mean()), "p_up": base_rate(tbl)}
    return pd.DataFrame(out).T


def barometer_returns(tbl: pd.DataFrame) -> pd.Series:
    """The tradable omen: capture the rest-of-year return only after an up January, else sit in cash."""
    return pd.Series(np.where(tbl["jan"] > 0, tbl["roy"], 0.0), index=tbl.index, name="barometer")


def buy_hold_roy(tbl: pd.DataFrame) -> pd.Series:
    """Benchmark: always hold the index for the rest of the year (the thing the omen must beat)."""
    return tbl["roy"].rename("buy_hold")


def summary(annual_returns: pd.Series) -> dict:
    """Annualised stats for a *yearly* return series (CAGR, annual Sharpe, max-drawdown, hit-rate)."""
    r = pd.Series(annual_returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("cagr", "sharpe", "vol", "max_drawdown", "hit_rate", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    cagr = eq.iloc[-1] ** (1.0 / len(r)) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "cagr": float(cagr),
        "sharpe": float(mean / std) if std > 0 else np.nan,  # yearly obs → already annual
        "vol": float(std),
        "max_drawdown": float(dd),
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }
