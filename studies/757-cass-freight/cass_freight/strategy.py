"""Strategy + inference for Study 757 — Cass-Freight.

The claim, operationalised on a monthly frame of ``cass`` (the Cass Freight Index proxy —
a level, or already-a-YoY on the synthetic control), ``spy`` and ``iyt`` (real month-end
closes):

    Freight moves the physical economy, so the Cass Freight Index *leads* the cycle: when
    freight is expanding be long equities (especially transports); when it rolls over, get
    out ahead of the slowdown the market hasn't priced.

We test it three ways:

  * **Conditional vs unconditional forward returns.** Mean H-month forward return of SPY /
    IYT when freight is expanding (YoY > 0, known after the publication lag) vs the
    unconditional base rate, with a Welch *t* and a placebo null.
  * **A lead-lag cross-correlation — the myth-check.** Correlate freight YoY with equity
    returns at a range of leads and lags. A genuine *leading* indicator peaks at a positive
    lead (freight today ⇒ stocks later). We ask whether freight leads, or whether equities
    lead *it*.
  * **A timing overlay, net of costs.** Hold SPY (or IYT) when freight is expanding, with
    the documented publication+execution lag and a one-way cost per turn, raced against
    buy-and-hold on a Sharpe basis.

The decisive object is the lead-lag structure: a "real-economy" dashboard that the market
has already discounted is a coincident-to-lagging read, not a nowcast you can trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DEFAULT_LAG

ANN = 12  # months per year


# --------------------------------------------------------------------------- #
# The freight signal
# --------------------------------------------------------------------------- #
def freight_yoy(frame: pd.DataFrame) -> pd.Series:
    """Year-over-year growth of the Cass proxy (the expansion/contraction signal).

    On the real frame ``cass`` is a *level*, so YoY = level / level.shift(12) − 1. On the
    synthetic control ``frame.attrs['cass_is_yoy']`` is set and ``cass`` already IS the YoY,
    returned as-is. No look-ahead: YoY at month ``t`` uses only levels up to ``t``.
    """
    if frame.attrs.get("cass_is_yoy"):
        return frame["cass"].astype(float)
    return (frame["cass"] / frame["cass"].shift(12) - 1.0)


def forward_returns(price: pd.Series, horizon: int) -> pd.Series:
    """``horizon``-month forward simple return for every month-end (NaN near the tail)."""
    return price.shift(-horizon) / price - 1.0


def expanding_months(frame: pd.DataFrame, thr: float = 0.0,
                     lag: int = DEFAULT_LAG) -> pd.Series:
    """Boolean Series: was freight 'expanding' (YoY > ``thr``) as *known* ``lag`` months
    earlier? The freight reference month ``t`` is public by ``t+1`` and acted on from
    ``t+2`` (``lag`` = publication + execution), so we compare ``freight_yoy.shift(lag)``.
    """
    return (freight_yoy(frame).shift(lag) > thr)


def conditional_returns(frame: pd.DataFrame, price_col: str, horizon: int,
                        thr: float = 0.0, lag: int = DEFAULT_LAG) -> np.ndarray:
    """Forward ``horizon``-month returns of ``price_col`` over months entered when the
    lagged freight signal says 'expanding'."""
    fwd = forward_returns(frame[price_col], horizon)
    sig = expanding_months(frame, thr=thr, lag=lag)
    sel = fwd[sig].dropna()
    return np.asarray(sel.values, dtype=float)


def unconditional_returns(frame: pd.DataFrame, price_col: str, horizon: int) -> np.ndarray:
    """All overlapping ``horizon``-month forward returns of ``price_col`` (the base rate)."""
    return np.asarray(forward_returns(frame[price_col], horizon).dropna().values,
                      dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2."""
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(frame: pd.DataFrame, price_col: str, horizon: int, thr: float = 0.0,
                   lag: int = DEFAULT_LAG, n_draws: int = 20_000, seed: int = 757) -> dict:
    """Small-sample placebo null: draw ``k`` random entry months (k = number of expanding
    months) many times and ask how often a random draw's mean forward return matches/beats
    the conditional set. Returns the conditional mean, placebo mean, and p = P[random >= obs].
    """
    obs = conditional_returns(frame, price_col, horizon, thr=thr, lag=lag)
    k = len(obs)
    fwd = forward_returns(frame[price_col], horizon).dropna().values
    n = len(fwd)
    if k == 0 or n == 0:
        return {"k": k, "cond_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = fwd[rng.integers(0, n, size=k)].mean()
    obs_mean = float(obs.mean())
    return {"k": k, "cond_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs_mean).mean())}


def summarize(frame: pd.DataFrame, price_col: str = "spy", horizon: int = 12,
              thr: float = 0.0, lag: int = DEFAULT_LAG) -> dict:
    """Headline stats for one horizon and one tape: n expanding months, conditional
    mean/win-rate, the unconditional base rate, Welch t, and the placebo p-value."""
    cond = conditional_returns(frame, price_col, horizon, thr=thr, lag=lag)
    base = unconditional_returns(frame, price_col, horizon)
    pl = placebo_pvalue(frame, price_col, horizon, thr=thr, lag=lag)
    return {
        "price_col": price_col, "horizon": horizon,
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()) if len(base) else float("nan"),
        "base_win": float((base > 0).mean()) if len(base) else float("nan"),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# Lead-lag cross-correlation — the "does freight LEAD?" myth-check
# --------------------------------------------------------------------------- #
def monthly_returns(frame: pd.DataFrame, price_col: str = "spy") -> pd.Series:
    """One-month simple returns of ``price_col`` on the month-end tape."""
    return (frame[price_col] / frame[price_col].shift(1) - 1.0)


def lead_lag_corr(frame: pd.DataFrame, price_col: str = "spy",
                  max_lag: int = 12) -> dict:
    """Cross-correlation of freight YoY (change) with equity monthly returns across leads.

    For each integer ``k`` in ``[-max_lag, +max_lag]`` we correlate the monthly *change* in
    freight YoY at month ``t`` with the equity return at month ``t+k``:

      * ``k > 0`` ⇒ freight change **leads** the equity move (freight today, stocks later) —
        what a genuine leading indicator needs.
      * ``k < 0`` ⇒ freight change **lags** the equity move (stocks today, freight later) —
        a coincident/lagging read of a market that already turned.

    Returns ``{"lags": [...], "corr": [...], "peak_lag": k*, "peak_corr": r*}`` where the
    peak is the ``k`` of maximum |corr|. We use the freight *change* (∆YoY), a stationary
    series, so the correlation is not a spurious trend-on-trend artefact.
    """
    dfreight = freight_yoy(frame).diff()
    eq = monthly_returns(frame, price_col)
    lags = list(range(-max_lag, max_lag + 1))
    corr = []
    for k in lags:
        joined = pd.concat([dfreight, eq.shift(-k)], axis=1).dropna()
        if len(joined) < 12:
            corr.append(float("nan"))
        else:
            corr.append(float(joined.iloc[:, 0].corr(joined.iloc[:, 1])))
    arr = np.array(corr)
    if np.all(np.isnan(arr)):
        return {"lags": lags, "corr": corr, "peak_lag": 0, "peak_corr": float("nan")}
    j = int(np.nanargmax(np.abs(arr)))
    return {"lags": lags, "corr": corr, "peak_lag": lags[j], "peak_corr": float(arr[j])}


# --------------------------------------------------------------------------- #
# Timing overlay, net of costs (the Tradability axis)
# --------------------------------------------------------------------------- #
def timing_backtest(frame: pd.DataFrame, price_col: str = "spy", thr: float = 0.0,
                    lag: int = DEFAULT_LAG, cost_bps: float = 10.0,
                    allow_short: bool = False) -> dict:
    """Long/flat (or long/short) timing overlay driven by the freight expansion sign.

    Position for month ``m`` is decided by the freight YoY known ``lag`` months earlier:
    +1 when YoY > ``thr``; 0 (long/flat) or −1 (long/short) otherwise. A one-way cost of
    ``cost_bps`` is charged on each change of position (turnover from the diff). Returns
    gross & net annualized return, vol and Sharpe for the rule and for buy-and-hold, both
    on the ``price_col`` tape (price-only, adjusted closes — labelled). The rule earns no
    cash yield while flat, which *flatters* it (a conservative comparison for the verdict).
    """
    ret = monthly_returns(frame, price_col).dropna()
    pos_raw = np.where(freight_yoy(frame).shift(lag) > thr, 1.0,
                       (-1.0 if allow_short else 0.0))
    pos = pd.Series(pos_raw, index=frame.index).reindex(ret.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    c = cost_bps / 1e4
    gross = pos * ret
    net = gross - turn * c

    def _stats(r: pd.Series) -> dict:
        mu, sd = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
        return {"ann_ret": float(mu), "ann_vol": float(sd),
                "sharpe": float(mu / sd) if sd > 0 else float("nan")}

    return {
        "price_col": price_col,
        "n_months": int(len(ret)),
        "n_turns": float(turn.sum()),
        "exposure": float((pos != 0).mean()),
        "gross": _stats(gross),
        "net": _stats(net),
        "buy_hold": _stats(ret),
        "cost_bps": cost_bps,
        "allow_short": allow_short,
    }
