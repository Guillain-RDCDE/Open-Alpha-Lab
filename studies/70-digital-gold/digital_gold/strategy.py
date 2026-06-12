"""Is bitcoin "digital gold"? Test the thesis that BTC is a gold-like store of value / safe haven: its
correlation with gold and with stocks (and how that correlation has drifted), its behaviour in equity
crashes, and whether a small BTC sleeve helped a portfolio. The alternative hypothesis: BTC is just a
high-beta risk asset that falls harder than stocks when it matters.
"""
from __future__ import annotations
import numpy as np, pandas as pd

TRADING_DAYS = 252


def annualized(ret: pd.Series) -> dict:
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann", "vol", "sharpe", "max_drawdown", "n")}
    ann, vol = r.mean() * TRADING_DAYS, r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min()
    return {"ann": float(ann), "vol": float(vol), "sharpe": float(ann / vol) if vol > 0 else np.nan,
            "max_drawdown": float(dd), "n": int(len(r))}


def correlations(ret: pd.DataFrame, btc="BTC-USD", eq="SPY", gold="GLD") -> dict:
    """BTC's correlation with stocks and gold, plus first-half vs second-half BTC–stock correlation to
    show whether it's drifting toward (or away from) being a risk asset."""
    half = len(ret) // 2
    return {"btc_stock": float(ret[btc].corr(ret[eq])), "btc_gold": float(ret[btc].corr(ret[gold])),
            "stock_gold": float(ret[eq].corr(ret[gold])),
            "btc_stock_first": float(ret[btc].iloc[:half].corr(ret[eq].iloc[:half])),
            "btc_stock_second": float(ret[btc].iloc[half:].corr(ret[eq].iloc[half:]))}


def crisis_behavior(monthly: pd.DataFrame, crash: float = -0.08, btc="BTC-USD", eq="SPY", gold="GLD") -> dict:
    """In equity-crash months (stock monthly return < ``crash``): mean BTC / gold / stock return and the
    share of crashes where BTC rose. A 'digital gold' would hold or rise; a risk asset falls harder."""
    cr = monthly[monthly[eq] < crash]
    return {"btc_in_crash": float(cr[btc].mean()), "gold_in_crash": float(cr[gold].mean()),
            "eq_in_crash": float(cr[eq].mean()),
            "btc_up_share": float((cr[btc] > 0).mean()) if len(cr) else np.nan,
            "gold_up_share": float((cr[gold] > 0).mean()) if len(cr) else np.nan, "n_crash": int(len(cr))}


def sleeve_effect(eq_ret: pd.Series, btc_ret: pd.Series, weights=(0.0, 0.05, 0.10)) -> pd.DataFrame:
    """Stats for a stock portfolio with a small BTC sleeve rebalanced daily — does a little BTC help, and
    at what drawdown cost?"""
    df = pd.concat([eq_ret.rename("eq"), btc_ret.rename("btc")], axis=1).dropna()
    rows = {}
    for w in weights:
        port = (1.0 - w) * df["eq"] + w * df["btc"]
        rows[f"{int(w*100)}% BTC"] = annualized(port)
    return pd.DataFrame(rows).T
