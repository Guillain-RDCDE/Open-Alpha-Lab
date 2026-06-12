"""Gold the "safe haven" — two folk claims, tested. (1) Inflation hedge: does gold track inflation, and
does it earn more when inflation is high? (2) Crisis hedge: does gold protect when equities crash? We
measure the inflation correlation, the high- vs low-inflation gold return, the stock correlation, and the
gold return in equity-crash months.
"""
from __future__ import annotations
import numpy as np, pandas as pd

MONTHS = 12


def yoy(monthly_level: pd.Series) -> pd.Series:
    return monthly_level.pct_change(MONTHS)


def inflation_hedge(gold_m: pd.Series, cpi_m: pd.Series, hi_thresh: float = 0.04) -> dict:
    """Correlation of year-on-year gold returns with YoY inflation, and gold's YoY return in
    high-inflation (>``hi_thresh``) vs low-inflation months. A real inflation hedge has positive
    correlation and a positive high-minus-low gap."""
    g, infl = yoy(gold_m), yoy(cpi_m)
    df = pd.concat([g.rename("gold"), infl.rename("infl")], axis=1).dropna()
    hi, lo = df[df["infl"] > hi_thresh], df[df["infl"] <= hi_thresh]
    return {"corr": float(df["gold"].corr(df["infl"])),
            "gold_yoy_mean": float(df["gold"].mean()), "infl_yoy_mean": float(df["infl"].mean()),
            "gold_hi_infl": float(hi["gold"].mean()), "gold_lo_infl": float(lo["gold"].mean()),
            "hi_minus_lo": float(hi["gold"].mean() - lo["gold"].mean()),
            "n": int(len(df)), "n_hi": int(len(hi))}


def crisis_ballast(gold_ret: pd.Series, eq_ret: pd.Series, crash: float = -0.08) -> dict:
    """Gold's behaviour in equity-crash months (equity monthly return < ``crash``): mean gold return,
    mean equity return, and the share of crash months where gold rose. A reliable haven *rises*; a mere
    ballast just doesn't fall."""
    df = pd.concat([gold_ret.rename("gold"), eq_ret.rename("eq")], axis=1).dropna()
    cr = df[df["eq"] < crash]
    return {"stock_corr": float(df["gold"].corr(df["eq"])),
            "gold_in_crash": float(cr["gold"].mean()), "eq_in_crash": float(cr["eq"].mean()),
            "gold_up_share": float((cr["gold"] > 0).mean()) if len(cr) else np.nan,
            "n_crash": int(len(cr)), "n": int(len(df))}


def summary(monthly_ret: pd.Series) -> dict:
    r = pd.Series(monthly_ret).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann", "vol", "sharpe", "n")}
    ann, vol = r.mean() * MONTHS, r.std(ddof=1) * np.sqrt(MONTHS)
    return {"ann": float(ann), "vol": float(vol),
            "sharpe": float(ann / vol) if vol > 0 else np.nan, "n": int(len(r))}
