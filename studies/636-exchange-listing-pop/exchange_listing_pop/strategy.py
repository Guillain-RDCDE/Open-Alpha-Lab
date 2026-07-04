"""Event-study engine for Study 636 — Exchange Listing Pop (the "Coinbase effect").

Everything runs on a wide daily USD close panel (Binance USDT pairs + a ``BTC`` market
column) and an event table of Coinbase **listing days** (day 0 = the first day the
asset trades on Coinbase Exchange).

Core quantities
---------------
* **Abnormal return** — daily log return of the coin **minus** the daily log return of
  BTC (market-adjusted; removes the common crypto beta so a listing during a
  market-wide melt-up or crash is not mis-billed to the event).
* **CAR(a, b)** — the sum of abnormal log returns over event-day offsets ``a..b``
  (calendar days; crypto trades every day, so calendar = trading days).
* **Cluster-by-day** — Coinbase frequently announces several assets in one blog post
  (e.g. MATIC + SKL + SUSHI, 2021-03-09), so same-listing-day events are one draw of
  news, not independent observations. The **primary** inference averages same-day
  events into one cluster before the one-sample t.

Inference
---------
* One-sample *t* of the per-cluster CARs against zero (primary, deterministic).
* Welch *t* of event CARs vs same-coin random-date **placebo** CARs.
* An empirical placebo *p*: the share of random pseudo-event calendars (same coins,
  same count) whose mean CAR beats the observed one.

Tradability (exactly ONE execution lag, documented)
---------------------------------------------------
The listing goes live during day 0; the earliest close a follower can trade is the
**day-0 UTC close**. Every tradable rule here enters at the day-0 close and exits at
the close of day ``+K``. Costs are one-way × traded NAV on each leg; the short (fade)
variant also pays borrow/funding on the notional for the holding period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MKT = "BTC"


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def abnormal_returns(px: pd.DataFrame, mkt: str = MKT) -> pd.DataFrame:
    """Daily abnormal log returns: coin log return minus market (BTC) log return."""
    logret = np.log(px).diff()
    abn = logret.drop(columns=[mkt]).sub(logret[mkt], axis=0)
    return abn


def raw_log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff()


# --------------------------------------------------------------------------- #
# Event CARs
# --------------------------------------------------------------------------- #
def _event_pos(series_index: pd.DatetimeIndex, day0: pd.Timestamp) -> int | None:
    """Integer position of day 0 in the panel index (exact date required)."""
    if day0 in series_index:
        return int(series_index.get_loc(day0))
    return None


def event_car(abn: pd.DataFrame, events: pd.DataFrame, a: int, b: int,
              min_est: int = 30) -> pd.DataFrame:
    """Per-event CAR over offsets ``a..b`` (inclusive) around day 0.

    Requires the coin's abnormal-return series to cover the full window plus
    ``min_est`` pre-event days (so every event in every window uses real tape, no
    padding). Returns a frame with columns coin, day0, car (log), n_days.
    """
    rows = []
    for _, ev in events.iterrows():
        coin, day0 = ev["coin"], pd.Timestamp(ev["day0"])
        if coin not in abn.columns:
            continue
        col = abn[coin]
        pos = _event_pos(abn.index, day0)
        if pos is None:
            continue
        lo, hi = pos + a, pos + b
        if lo - min_est < 0 or hi >= len(abn.index):
            continue
        seg = col.iloc[lo:hi + 1]
        if seg.isna().any():
            continue
        rows.append(dict(coin=coin, day0=day0, car=float(seg.sum()),
                         n_days=int(len(seg))))
    return pd.DataFrame(rows)


def event_matrix(abn: pd.DataFrame, events: pd.DataFrame, a: int = -20, b: int = 30,
                 min_est: int = 30) -> pd.DataFrame:
    """Offsets × events matrix of abnormal log returns (for average-path charts).

    Same coverage rule as :func:`event_car`: an event enters only if its full
    ``a..b`` window (plus ``min_est`` pre-days) is on the tape with no gaps.
    """
    cols = {}
    for _, ev in events.iterrows():
        coin, day0 = ev["coin"], pd.Timestamp(ev["day0"])
        if coin not in abn.columns:
            continue
        pos = _event_pos(abn.index, day0)
        if pos is None:
            continue
        lo, hi = pos + a, pos + b
        if lo - min_est < 0 or hi >= len(abn.index):
            continue
        seg = abn[coin].iloc[lo:hi + 1]
        if seg.isna().any():
            continue
        cols[f"{coin}_{day0.date()}"] = seg.to_numpy()
    return pd.DataFrame(cols, index=range(a, b + 1))


def cluster_by_day(cars: pd.DataFrame) -> pd.Series:
    """Average same-listing-day events into one observation (one news draw)."""
    return cars.groupby("day0")["car"].mean()


def one_sample_t(x: np.ndarray) -> tuple[float, float, int]:
    """(mean, t, n) of a one-sample t-test against zero."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m / se) if se > 0 else float("nan"), n


def welch_t(x: np.ndarray, y: np.ndarray) -> float:
    """Welch two-sample t of mean(x) - mean(y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    return float((x.mean() - y.mean()) / np.sqrt(vx + vy))


def window_summary(abn: pd.DataFrame, events: pd.DataFrame, a: int, b: int) -> dict:
    """Mean CAR, cluster-level one-sample t, and counts for one event window."""
    cars = event_car(abn, events, a, b)
    cl = cluster_by_day(cars)
    m, t, n = one_sample_t(cl.to_numpy())
    m_ev = float(cars["car"].mean()) if len(cars) else float("nan")
    return dict(a=a, b=b, mean_car=m, t=t, n_clusters=n,
                n_events=len(cars), mean_car_events=m_ev,
                mean_simple_pct=float((np.exp(m) - 1) * 100) if np.isfinite(m) else float("nan"))


# --------------------------------------------------------------------------- #
# Placebo — random pseudo-listing dates on the same coins
# --------------------------------------------------------------------------- #
def placebo_cars(abn: pd.DataFrame, events: pd.DataFrame, a: int, b: int,
                 n_draws: int = 2000, seed: int = 636, buffer_days: int = 90,
                 min_est: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Placebo distribution for one window.

    For each draw: for every event coin pick a random date in that coin's Binance
    history at least ``buffer_days`` away from the true listing (and with the full
    window + estimation run available), compute the same CAR, take the mean across
    the pseudo-events. Returns ``(all pseudo CARs pooled, per-draw means)``.
    """
    rng = np.random.default_rng(seed)
    idx = abn.index
    per_event_pool: list[np.ndarray] = []
    csum = abn.cumsum()
    for _, ev in events.iterrows():
        coin, day0 = ev["coin"], pd.Timestamp(ev["day0"])
        if coin not in abn.columns:
            continue
        col = abn[coin]
        valid = col.notna()
        pos0 = _event_pos(idx, day0)
        if pos0 is None:
            continue
        ok = []
        first = int(valid.values.argmax())
        last = len(idx) - 1 - int(valid.values[::-1].argmax())
        for p in range(first + min_est - a if a < 0 else first + min_est, last - b):
            if abs(p - pos0) < buffer_days:
                continue
            lo, hi = p + a, p + b
            if lo < first or hi > last:
                continue
            ok.append(p)
        if len(ok) < 50:
            continue
        # CAR for every admissible position, vectorised via cumsum
        cs = csum[coin].to_numpy()
        ok = np.asarray(ok)
        car = cs[ok + b] - np.where(ok + a - 1 >= 0, cs[ok + a - 1], 0.0)
        car = car[~np.isnan(car)]
        if len(car) >= 50:
            per_event_pool.append(car)
    if not per_event_pool:
        return np.array([]), np.array([])
    draws = np.empty(n_draws)
    n_ev = len(per_event_pool)
    for d in range(n_draws):
        pick = [pool[rng.integers(0, len(pool))] for pool in per_event_pool]
        draws[d] = float(np.mean(pick))
    pooled = np.concatenate(per_event_pool)
    return pooled, draws


def placebo_pvalue(abn: pd.DataFrame, events: pd.DataFrame, a: int, b: int,
                   n_draws: int = 2000, seed: int = 636, side: str = "greater") -> dict:
    """Empirical placebo p + Welch t of event CARs vs the pooled placebo CARs."""
    cars = event_car(abn, events, a, b)
    obs = float(cars["car"].mean())
    pooled, draws = placebo_cars(abn, events, a, b, n_draws=n_draws, seed=seed)
    if side == "greater":
        p = float((draws >= obs).mean())
    else:
        p = float((draws <= obs).mean())
    wt = welch_t(cars["car"].to_numpy(), pooled) if len(pooled) else float("nan")
    return dict(obs=obs, p=p, welch_t=wt, draws=draws, n_pooled=int(len(pooled)))


# --------------------------------------------------------------------------- #
# Tradability — the follower's trade (enter at the day-0 close, ONE lag)
# --------------------------------------------------------------------------- #
def follower_trade(px: pd.DataFrame, events: pd.DataFrame, hold: int = 30,
                   cost_bps: float = 25.0, hedged: bool = True,
                   borrow_alt: float = 0.10, borrow_btc: float = 0.03,
                   side: str = "long") -> pd.DataFrame:
    """Net per-event return of the only trade a follower can actually place.

    Enter at the **day-0 UTC close** (the listing happened during day 0 — one
    execution lag), exit at the close of day ``+hold``. ``side='long'`` rides the pop;
    ``side='short'`` fades it. Any short leg pays borrow on the notional for the
    holding period: ``borrow_alt``/yr for the alt-coin leg, ``borrow_btc``/yr for the
    BTC hedge leg. ``hedged=True`` nets out BTC (long coin / short BTC or the
    reverse), charging costs on **both** legs. Costs are one-way ``cost_bps`` ×
    traded NAV per leg.
    """
    lr = raw_log_returns(px)
    rows = []
    for _, ev in events.iterrows():
        coin, day0 = ev["coin"], pd.Timestamp(ev["day0"])
        if coin not in px.columns:
            continue
        pos = _event_pos(px.index, day0)
        if pos is None or pos + hold >= len(px.index):
            continue
        seg = lr[coin].iloc[pos + 1: pos + hold + 1]
        seg_m = lr[MKT].iloc[pos + 1: pos + hold + 1]
        if seg.isna().any() or seg_m.isna().any():
            continue
        r_coin = float(np.exp(seg.sum()) - 1)
        r_mkt = float(np.exp(seg_m.sum()) - 1)
        gross = r_coin - r_mkt if hedged else r_coin
        if side == "short":
            gross = -gross
        # costs: round trip on the coin leg (+ round trip on the hedge leg if hedged)
        legs = 2 * (2 if hedged else 1)
        cost = cost_bps / 1e4 * legs
        # any short leg pays borrow for the holding period
        borrow = 0.0
        if side == "short":
            borrow += borrow_alt * hold / 365.0                 # short the alt
            if hedged:
                borrow += 0.0                                   # hedge = LONG BTC
        elif hedged:
            borrow += borrow_btc * hold / 365.0                 # long alt, SHORT BTC
        net = gross - cost - borrow
        rows.append(dict(coin=coin, day0=day0, gross=gross, net=net))
    return pd.DataFrame(rows)


def trade_summary(trades: pd.DataFrame) -> dict:
    """Cluster same-day trades, then mean/t of gross and net."""
    if trades.empty:
        return dict(n=0)
    g = trades.groupby("day0")[["gross", "net"]].mean()
    mg, tg, n = one_sample_t(g["gross"].to_numpy())
    mn, tn, _ = one_sample_t(g["net"].to_numpy())
    return dict(gross_pct=mg * 100, t_gross=tg, net_pct=mn * 100, t_net=tn,
                n=n, hit=float((g["net"] > 0).mean()))
