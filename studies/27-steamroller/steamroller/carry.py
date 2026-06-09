"""The carry signal and engine — and the load-bearing question: **does a higher rate predict a higher return?**

The carry signal is simply the **interest rate**: rank currencies by their short rate and the claim is
that the high-rate ones earn more, because uncovered interest-rate parity (UIRP) fails — the high-rate
currency does not depreciate enough to offset its yield advantage. The engine measures that directly:
sort by rate, read the average excess return per bucket. If high-rate currencies earned *less* (UIRP
held), there'd be no carry.

For real data the monthly **excess return** of holding currency ``i`` (funded in USD) is the rate
differential plus the spot appreciation: ``xret_i ≈ (r_i − r_usd)/12 + Δlog(fx_i)``. The synthetic
generator returns those excess returns directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def excess_returns(rates: pd.DataFrame, fx: pd.DataFrame, base: str = "USD") -> pd.DataFrame:
    """Monthly carry excess returns from rates and USD FX: ``(r_i − r_base)/12 + Δlog(fx_i)``.

    ``rates`` are annual %; ``fx`` is USD per foreign unit (so ``Δlog(fx)`` is foreign appreciation vs
    USD). The base currency (USD) has zero FX move and serves as the funding leg. Returns a
    ``months × currency`` frame of excess returns over the base.
    """
    r = rates / 100.0 / MONTHS_PER_YEAR
    base_r = r[base] if base in r.columns else r.mean(axis=1)
    fx_ret = np.log(fx).diff()
    cols = [c for c in fx.columns if c in r.columns]
    out = {}
    for c in cols:
        out[c] = (r[c] - base_r) + fx_ret[c]
    return pd.DataFrame(out).dropna(how="all")


def carry_premium_by_bucket(xret: pd.DataFrame, rates: pd.DataFrame, n_buckets: int = 3) -> dict:
    """Sort currencies by rate into ``n_buckets`` each month; read the average excess return per bucket.

    A **rising** profile (high-rate buckets out-earn low-rate ones) is the carry premium; the
    high-minus-low spread (annualised) is its size. Under full UIRP (the null) the profile is flat.
    """
    common = xret.index.intersection(rates.index)
    xr, rt = xret.loc[common], rates.loc[common]
    bucket_ret = {b: [] for b in range(n_buckets)}
    for dt in common:
        row_r = rt.loc[dt].dropna()
        row_x = xr.loc[dt].reindex(row_r.index).dropna()
        row_r = row_r.reindex(row_x.index)
        if len(row_x) < n_buckets:
            continue
        b = pd.qcut(row_r.rank(method="first"), n_buckets, labels=False)
        for k in range(n_buckets):
            v = row_x[b == k]
            if len(v):
                bucket_ret[k].append(v.mean())
    table = pd.Series({b: np.mean(v) if v else np.nan for b, v in bucket_ret.items()})
    ann = table * MONTHS_PER_YEAR * 100.0
    lo, hi = float(table.iloc[0]), float(table.iloc[-1])
    return {
        "bucket_ann_pct": ann.rename("ann_pct"),
        "hml_ann_pct": float((hi - lo) * MONTHS_PER_YEAR * 100.0),
        "premium_present": bool(hi > lo),
        "n_months": int(len(common)),
    }
