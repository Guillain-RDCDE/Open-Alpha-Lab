"""The pre-earnings-runup strategy — buy N trading days before each scheduled announcement, exit the day
before the announcement (do NOT hold through the actual release).

The signal is simple: if a name has an earnings announcement scheduled T days from today (within [1, pre_days]
trading days), go long. The book is equal-weighted across all names currently in the pre-event window,
long-only (there is no obvious short — shorting names NOT near an announcement is a market bet, not an
earnings bet). We compare gross returns vs the market beta-adjusted excess return.

Implementation:
  1. For each trading day t, scan events to find names whose announcement date falls exactly
     ``k`` trading days ahead, for k in {1, 2, ..., pre_days}. Those names are IN the book today.
  2. Equal-weight across all in-book names; weight = 0 outside the window.
  3. Book return on day t+1 = mean return of the in-book names (lagged one day for tradeability).
  4. The position is entered at the OPEN of day t+1 (i.e. we act on the known upcoming earnings
     calendar, known by end of prior day — standard calendar scrape).

The claim from the literature (Lakonishok & Vermaelen 1990; Kim & Park 2005; So & Wang 2014): informed
investors and options traders position ahead of the announcement, creating a systematic pre-event drift
that averages +0.5–1% over the 5 days before the announcement. On a large-cap universe with tight spreads
and heavy institutional attention, this premium is expected to be much smaller and frequently arbitraged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _build_pre_event_mask(
    panel_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    pre_days: int = 5,
) -> pd.DataFrame:
    """Return a boolean ``dates × ticker`` frame that is True on the ``pre_days`` trading days *before*
    each announced event (the announcement day itself is NOT included — the strategy exits beforehand).

    Causal: on trading day t, the mask is True if the event falls 1..pre_days trading days *after* t,
    meaning the strategy enters on t and exits before the announcement.

    Vectorized implementation: for each ticker, build a Series of event positions and spread a window
    backward using rolling max on a binary indicator.
    """
    idx = panel_index
    n = len(idx)
    all_tickers = events["ticker"].unique().tolist()

    # Build a binary indicator frame: 1 on the day of each event, 0 elsewhere
    # Then rolling max backward over pre_days windows captures the window
    indicator = pd.DataFrame(0, index=idx, columns=all_tickers, dtype=np.int8)

    for tk, grp in events.groupby("ticker"):
        if tk not in indicator.columns:
            continue
        locs = idx.searchsorted(pd.DatetimeIndex(grp["date"]), side="left")
        locs = locs[locs < n]
        indicator.iloc[locs, indicator.columns.get_loc(tk)] = 1

    # A rolling window of size pre_days+1 shifted so the event day is NOT included:
    # roll forward: at position t, the window covers [t-pre_days, t]; we want 1 if any event
    # falls in [t+1, t+pre_days]. Equivalently, shift indicator by pre_days backward and roll.
    # Simpler: shift(-1) then rolling(pre_days).max() gives True on the pre_days days before the event.
    mask = (
        indicator.shift(-1)
        .rolling(window=pre_days, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )
    # shift(-1) moves the event indicator one step earlier; rolling max over pre_days then marks
    # the pre_days days before the event as True (event day itself gets False after the shift).
    return mask


def runup_weights(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    pre_days: int = 5,
) -> pd.DataFrame:
    """Equal-weight long-only pre-earnings book: each name in the pre-event window gets weight 1/N(t)
    where N(t) is the number of in-book names on day t. The weight frame is **lagged one day** so the
    book is strictly causal (the position is determined by yesterday's knowledge of the earnings calendar).

    Only names that appear in both the panel columns and events are included.
    """
    # Restrict to tickers present in the panel
    ev = events[events["ticker"].isin(panel.columns)].copy()
    mask = _build_pre_event_mask(panel.index, ev, pre_days=pre_days)

    # Align mask to panel columns — fill missing with False
    mask = mask.reindex(columns=panel.columns, fill_value=False)

    count = mask.sum(axis=1).replace(0, np.nan)
    w = mask.astype(float).div(count, axis=0).fillna(0.0)

    # Lag one day: weight on day t is based on information available before the open on t
    return w.shift(1).fillna(0.0)


def book_returns(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    pre_days: int = 5,
    cost_bps: float = 2.0,
) -> pd.Series:
    """Net daily return of the equal-weight pre-earnings book: weights × next-day returns minus turnover
    cost (``cost_bps`` per unit traded one-way). The book is long-only; returns are compared to market
    in the summary function, not here (raw book return, unhedged)."""
    w = runup_weights(panel, events, pre_days=pre_days)
    gross = (w * panel).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename("runup_book")


def market_return(panel: pd.DataFrame) -> pd.Series:
    """Simple equal-weight market return (across all names in the panel) as a benchmark."""
    return panel.mean(axis=1).rename("market")


def turnover(panel: pd.DataFrame, events: pd.DataFrame, pre_days: int = 5) -> float:
    """Average daily one-way turnover Σ|Δw| of the pre-earnings book."""
    w = runup_weights(panel, events, pre_days=pre_days)
    return float(w.diff().abs().sum(axis=1).mean())


def mean_pre_event_return(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    pre_days: int = 5,
) -> pd.Series:
    """Average raw return (unweighted) across all name×day observations in the pre-event window,
    by day-before-the-announcement (day -pre_days, ..., day -1). Returns a Series indexed by
    offset from the announcement (-pre_days = first day of the window, -1 = last day before announcement).

    This is the cleanest test of the drift: does the average return in the window, aligned to the
    announcement date, trend upward?
    """
    ev = events[events["ticker"].isin(panel.columns)].copy()
    idx = panel.index
    n = len(idx)

    # Precompute event positions and column indices once
    col_lookup = {c: i for i, c in enumerate(panel.columns)}
    arr = panel.values  # (n_dates, n_tickers)

    ev_locs = idx.searchsorted(pd.to_datetime(ev["date"]), side="left")
    ev_cols = np.array([col_lookup.get(tk, -1) for tk in ev["ticker"]])
    valid_events = ev_cols >= 0

    results = {}
    for offset in range(-pre_days, 0):
        target = ev_locs + offset
        ok = valid_events & (target >= 0) & (target < n)
        t_idx = target[ok]
        c_idx = ev_cols[ok]
        vals = arr[t_idx, c_idx].astype(float)
        finite_vals = vals[np.isfinite(vals)]
        results[offset] = float(np.mean(finite_vals)) if len(finite_vals) > 0 else float("nan")

    return pd.Series(
        {k: results[k] for k in sorted(results)},
        name="mean_return",
    )


def newey_west_t(returns: pd.Series, lags: int = 5) -> float:
    """HAC (Newey-West) *t*-statistic of the mean of a daily return series.

    The pre-earnings book holds each position for ~pre_days days, so serial correlation is mild but
    present. Newey-West with Bartlett weights up to ``lags`` days corrects for that."""
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    n = r.size
    if n < 3:
        return float("nan")
    mean = r.mean()
    u = r - mean
    var = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(u[k:] @ u[:-k]) / n
        var += 2.0 * w * gamma_k
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mean / se)


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew, n_days for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
        "skew": float(r.skew()),
        "n_days": int(len(r)),
    }


def pre_day_sweep(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    windows: list[int] | None = None,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Sweep over different pre-event windows (1, 2, 3, 5, 10 days before the announcement) and
    return a DataFrame with gross Sharpe, net Sharpe, and turnover for each window length."""
    if windows is None:
        windows = [1, 2, 3, 5, 10]
    rows = []
    for w in windows:
        g = book_returns(panel, events, pre_days=w, cost_bps=0.0)
        n = book_returns(panel, events, pre_days=w, cost_bps=cost_bps)
        sg = summary(g)
        sn = summary(n)
        to = turnover(panel, events, pre_days=w)
        rows.append({"pre_days": w, "gross_sharpe": sg["sharpe"], "net_sharpe": sn["sharpe"],
                     "turnover_per_day": to})
    return pd.DataFrame(rows).set_index("pre_days")


def breakeven_cost_bps(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    pre_days: int = 5,
) -> float:
    """The one-way cost in bps at which the book's gross-mean daily return is exactly consumed by turnover.
    gross_mean / turnover_per_day / 1e-4."""
    g = book_returns(panel, events, pre_days=pre_days, cost_bps=0.0)
    mean_daily = float(g.mean())
    to = turnover(panel, events, pre_days=pre_days)
    if to <= 0 or mean_daily <= 0:
        return 0.0
    return float(mean_daily / to / 1e-4)
