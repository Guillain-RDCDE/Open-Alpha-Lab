"""The hedging-pressure signal and the engine: **does net hedger short-positioning predict returns?**

Hedging pressure ``HP_i = (CommShort − CommLong) / (CommShort + CommLong)`` is positive when commercial
hedgers are net short — i.e. producers are paying speculators to take the long side. The Cootner /
Gorton-Hayashi-Rouwenhorst thesis is that this *predicts* a positive return for the long speculator. The
engine measures it directly: each week, relate the (past-only) hedging-pressure signal to the next
week's futures return, pooled across the basket.

We work with the signal *demeaned per commodity* (each commodity has its own structural hedging level,
so what matters is its deviation from its own norm), and always lag it one week so it's tradable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def hp_signal(hp: pd.DataFrame, window: int = 156) -> pd.DataFrame:
    """Trailing z-score of each commodity's hedging pressure (past-only, lagged) — the tradable signal.

    Each commodity is demeaned/scaled by its own trailing ``window`` (≈3 years) so the signal is "how
    net-short are the hedgers *for this commodity, relative to its own history*". Lagged one week.
    """
    m = hp.rolling(window, min_periods=26).mean()
    s = hp.rolling(window, min_periods=26).std(ddof=1)
    return ((hp - m) / s).shift(1)


def hedging_premium(returns: pd.DataFrame, hp: pd.DataFrame, window: int = 156) -> dict:
    """Does high hedging pressure predict a higher next-week return? The pooled information coefficient.

    Aligns the (lagged) hedging-pressure signal with the contemporaneous next-week return, pools across
    the basket and over time, and reports the mean cross-sectional IC and its t-stat. A positive IC is
    the hedging-pressure premium; ~0 on a null. Also reports the top-minus-bottom tercile spread.
    """
    sig = hp_signal(hp, window)
    common = returns.index.intersection(sig.index)
    r, g = returns.loc[common], sig.loc[common]
    ics, top, bot = [], [], []
    for dt in common:
        gg = g.loc[dt].dropna()
        rr = r.loc[dt].reindex(gg.index).dropna()
        gg = gg.reindex(rr.index)
        if len(rr) < 4 or gg.std() == 0 or rr.std() == 0:
            continue
        ics.append(np.corrcoef(gg, rr)[0, 1])
        k = max(1, len(gg) // 3)
        order = gg.sort_values()
        bot.append(rr[order.index[:k]].mean()); top.append(rr[order.index[-k:]].mean())
    ics = np.array(ics)
    hi, lo = float(np.nanmean(top)), float(np.nanmean(bot))
    return {
        "mean_ic": float(ics.mean()) if len(ics) else np.nan,
        "ic_t": float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))) if len(ics) > 2 and ics.std(ddof=1) > 0 else np.nan,
        "top_minus_bottom_ann_pct": float((hi - lo) * WEEKS_PER_YEAR * 100.0),
        "premium_present": bool(np.nanmean(ics) > 0) if len(ics) else False,
        "n_weeks": int(len(ics)),
    }
