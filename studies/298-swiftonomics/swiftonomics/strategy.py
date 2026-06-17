"""Event-study and honest controls -- Study 298 (Swiftonomics).

The "Swiftonomics" claim, in the only form we can test on a clean tape: each
Eras Tour news event (announcement, ticketing meltdown, box-office record, film,
finale) moves **Live Nation (LYV)** by a measurable *abnormal* return -- i.e. a
return over and above what the market (^GSPC) explains. A tradable version would
be: at each tour event, go long LYV for a few days.

We test this with a textbook **market-model event study** (MacKinlay 1997):

1. **Estimation window.** For each event date, fit ``r_LYV = alpha + beta*r_mkt``
   by OLS on a clean window of ``est_window`` trading days that ends ``gap`` days
   *before* the event window -- no look-ahead.

2. **Abnormal return.** ``AR_t = r_LYV_t - (alpha + beta*r_mkt_t)`` inside the
   event window ``[-pre, +post]`` around the event day.

3. **CAR / CAAR.** The cumulative AR over the event window is the event's CAR;
   averaged across all events it is the CAAR (cumulative average abnormal return),
   the headline statistic.

4. **Inference.** A cross-event t-test on the per-event CARs (each event is one
   independent observation), plus a HAC/Newey-West t-stat on the daily AAR series
   to respect any serial correlation in the abnormal returns.

5. **Placebo dates.** Re-run the whole event study on random "placebo" dates that
   are *not* tour events. If the real CAAR is indistinguishable from the placebo
   distribution, there is no event effect.

**No look-ahead**: betas use only pre-event data (estimation window ends ``gap``
days before the window). A tradable position is entered at the *next* open after
the event (one trading-day execution lag).

**Costs**: a tradable long-LYV-around-events sleeve is round-tripped per event;
one-way cost x NAV is charged on entry and exit (gross vs net both reported).

**Survivorship / single-name**: this is a *single stock*. There is no
survivorship bias in the usual cross-sectional sense, but LYV is one name -- the
result is a statement about one company, not a diversified factor. Named on the
Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STOCK = "LYV"
MARKET = "^GSPC"


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns for the (LYV, ^GSPC) price panel."""
    return prices.pct_change().dropna(how="all")


def _market_model(stock_ret: np.ndarray, mkt_ret: np.ndarray) -> tuple[float, float]:
    """OLS alpha, beta for r_stock = alpha + beta * r_mkt."""
    x = np.column_stack([np.ones_like(mkt_ret), mkt_ret])
    coef, *_ = np.linalg.lstsq(x, stock_ret, rcond=None)
    return float(coef[0]), float(coef[1])


# ---------------------------------------------------------------------------
# Event-window abnormal returns
# ---------------------------------------------------------------------------
def map_events_to_trading_days(
    events: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each event date to the first trading day on/after it.

    Weekend/holiday news is therefore attributed to the next open. Events that
    fall after the end of the price sample are dropped. Returns a copy of
    ``events`` with an added ``event_pos`` (integer location in ``trading_index``)
    and ``event_day`` (the mapped trading date).
    """
    ti = pd.DatetimeIndex(trading_index)
    rows = []
    for _, ev in events.iterrows():
        d = pd.Timestamp(ev["date"])
        loc = ti.searchsorted(d, side="left")
        if loc >= len(ti):
            continue
        rec = dict(ev)
        rec["event_pos"] = int(loc)
        rec["event_day"] = ti[loc]
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out


def event_study(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre: int = 1,
    post: int = 3,
    est_window: int = 120,
    gap: int = 5,
    stock: str = STOCK,
    market: str = MARKET,
) -> dict:
    """Run a market-model event study around the supplied event dates.

    For each event with enough clean history, estimate (alpha, beta) on
    ``est_window`` days ending ``gap`` days before the event window, then compute
    the abnormal returns over ``[-pre, +post]`` trading days around the event.

    Returns a dict with:
      - ``car`` : np.ndarray of per-event CARs over the event window
      - ``ar_matrix`` : (n_events x (pre+post+1)) abnormal returns, rows = events
      - ``aar`` : average abnormal return per relative day (length pre+post+1)
      - ``caar`` : cumulative average abnormal return (running sum of aar)
      - ``betas`` : per-event estimated betas
      - ``offsets`` : the relative-day axis (e.g. [-1, 0, 1, 2, 3])
      - ``n_events`` : number of events actually used
    """
    rets = daily_returns(prices)
    ti = rets.index
    mapped = map_events_to_trading_days(events, ti)

    width = pre + post + 1
    offsets = np.arange(-pre, post + 1)

    car_list: list[float] = []
    beta_list: list[float] = []
    ar_rows: list[np.ndarray] = []

    s = rets[stock].to_numpy()
    m = rets[market].to_numpy()
    n = len(ti)

    for _, ev in mapped.iterrows():
        pos = int(ev["event_pos"])
        win_start = pos - pre
        win_end = pos + post
        est_end = win_start - gap
        est_start = est_end - est_window
        if est_start < 0 or win_end >= n:
            continue

        a, b = _market_model(s[est_start:est_end], m[est_start:est_end])
        ar = s[win_start:win_end + 1] - (a + b * m[win_start:win_end + 1])
        ar_rows.append(ar)
        car_list.append(float(ar.sum()))
        beta_list.append(b)

    if not ar_rows:
        return {
            "car": np.array([]), "ar_matrix": np.empty((0, width)),
            "aar": np.zeros(width), "caar": np.zeros(width),
            "betas": np.array([]), "offsets": offsets, "n_events": 0,
        }

    ar_matrix = np.vstack(ar_rows)
    aar = ar_matrix.mean(axis=0)
    caar = np.cumsum(aar)

    return {
        "car": np.array(car_list),
        "ar_matrix": ar_matrix,
        "aar": aar,
        "caar": caar,
        "betas": np.array(beta_list),
        "offsets": offsets,
        "n_events": len(car_list),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of a series (H0: mean = 0)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def car_inference(study: dict) -> dict:
    """Cross-event inference on the per-event CARs.

    Treats each event's CAR as one independent observation and runs a simple
    t-test (H0: mean CAR = 0). Also reports a HAC t-stat on the daily AAR series
    (respecting serial correlation within the event window) and the hit-rate
    (fraction of events with a positive CAR).
    """
    car = np.asarray(study["car"], dtype=float)
    n = car.size
    if n < 2:
        return {"n_events": int(n), "mean_car_bps": float("nan"), "car_t": float("nan"),
                "car_p": float("nan"), "hit_rate": float("nan"), "aar_hac_t": float("nan")}

    mean_car = float(car.mean())
    se = float(car.std(ddof=1) / np.sqrt(n))
    car_t = mean_car / se if se > 0 else float("nan")

    # Two-sided p from Student-t.
    from scipy import stats as scipy_stats
    car_p = float(2.0 * scipy_stats.t.sf(abs(car_t), df=n - 1)) if np.isfinite(car_t) else float("nan")

    aar_hac_t = _hac_tstat(study["aar"])

    return {
        "n_events": int(n),
        "mean_car_bps": round(mean_car * 1e4, 1),
        "car_t": round(float(car_t), 3),
        "car_p": round(float(car_p), 4),
        "hit_rate": round(float((car > 0).mean()), 3),
        "aar_hac_t": round(float(aar_hac_t), 3),
    }


def placebo_distribution(
    prices: pd.DataFrame,
    n_events: int,
    pre: int = 1,
    post: int = 3,
    est_window: int = 120,
    gap: int = 5,
    n_draws: int = 1000,
    seed: int = 298,
    stock: str = STOCK,
    market: str = MARKET,
) -> dict:
    """Random-date placebo: mean CAR from random event sets of the same size.

    Draws ``n_draws`` sets of ``n_events`` random trading days (each with a clean
    estimation window), runs the same market-model event study, and records the
    distribution of mean CARs. The placebo p-value for an observed mean CAR is the
    fraction of placebo means at least as extreme (two-sided).

    Returns ``{'mean_cars': np.ndarray, 'placebo_std_bps': float}``.
    """
    rets = daily_returns(prices)
    s = rets[stock].to_numpy()
    m = rets[market].to_numpy()
    n = len(rets)
    lo = est_window + gap + pre
    hi = n - post - 1
    rng = np.random.default_rng(seed)

    mean_cars = np.empty(n_draws)
    for d in range(n_draws):
        positions = rng.integers(lo, hi, size=n_events)
        cars = np.empty(n_events)
        for i, pos in enumerate(positions):
            win_start = pos - pre
            est_end = win_start - gap
            est_start = est_end - est_window
            a, b = _market_model(s[est_start:est_end], m[est_start:est_end])
            ar = s[win_start:pos + post + 1] - (a + b * m[win_start:pos + post + 1])
            cars[i] = ar.sum()
        mean_cars[d] = cars.mean()

    return {
        "mean_cars": mean_cars,
        "placebo_std_bps": round(float(mean_cars.std(ddof=1) * 1e4), 1),
    }


def placebo_pvalue(observed_mean_car: float, mean_cars: np.ndarray) -> float:
    """Two-sided placebo p-value: fraction of placebo means at least as extreme."""
    mc = np.asarray(mean_cars, dtype=float)
    centred = mc - mc.mean()
    return float((np.abs(centred) >= abs(observed_mean_car - mc.mean())).mean())


# ---------------------------------------------------------------------------
# Tradable sleeve -- long LYV around each event, with costs
# ---------------------------------------------------------------------------
def tradable_sleeve(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    hold: int = 3,
    one_way_bps: float = 10.0,
    lag: int = 1,
    stock: str = STOCK,
    market: str = MARKET,
) -> dict:
    """Long-LYV-around-events sleeve with a one-trading-day execution lag and costs.

    For each event, enter LYV at the open *after* the event day (``lag`` trading
    days) and hold ``hold`` trading days. The per-event raw return is the LYV
    return over the holding window; the market-hedged return subtracts the ^GSPC
    return over the same window (a long-LYV / short-market pair, which is the
    honest abnormal-return trade). One-way cost x NAV is charged on entry and exit
    (round trip = ``2 * one_way_bps``).

    Returns gross and net per-event statistics for both the raw and hedged trades.
    """
    rets = daily_returns(prices)
    ti = rets.index
    mapped = map_events_to_trading_days(events, ti)
    n = len(rets)
    rt_cost = 2.0 * one_way_bps * 1e-4

    raw_g, raw_n, hedge_g, hedge_n = [], [], [], []
    for _, ev in mapped.iterrows():
        entry = int(ev["event_pos"]) + lag
        exit_ = entry + hold
        if entry < 0 or exit_ > n:
            continue
        win = slice(entry, exit_)
        r_lyv = float((1.0 + rets[stock].iloc[win]).prod() - 1.0)
        r_mkt = float((1.0 + rets[market].iloc[win]).prod() - 1.0)
        raw_g.append(r_lyv)
        raw_n.append(r_lyv - rt_cost)
        hedge_g.append(r_lyv - r_mkt)
        hedge_n.append(r_lyv - r_mkt - 2.0 * rt_cost)  # both legs traded

    def _stat(v):
        a = np.asarray(v, dtype=float)
        if a.size < 2:
            return {"mean_bps": float("nan"), "t": float("nan"), "n": int(a.size)}
        se = a.std(ddof=1) / np.sqrt(a.size)
        return {"mean_bps": round(float(a.mean()) * 1e4, 1),
                "t": round(float(a.mean() / se), 3) if se > 0 else float("nan"),
                "n": int(a.size)}

    return {
        "raw_gross": _stat(raw_g),
        "raw_net": _stat(raw_n),
        "hedged_gross": _stat(hedge_g),
        "hedged_net": _stat(hedge_n),
        "one_way_bps": one_way_bps,
        "hold": hold,
    }


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre: int = 1,
    post: int = 3,
    n_placebo: int = 1000,
    seed: int = 298,
) -> dict:
    """One-stop summary dict for the real (or synthetic) price/event panel.

    Mirrors the R = dict(...) in build_notebooks.py and docs/results.md.
    """
    study = event_study(prices, events, pre=pre, post=post)
    inf = car_inference(study)
    plc = placebo_distribution(prices, n_events=max(study["n_events"], 1),
                               pre=pre, post=post, n_draws=n_placebo, seed=seed)
    plc_p = placebo_pvalue(float(np.asarray(study["car"]).mean()) if study["n_events"] else 0.0,
                           plc["mean_cars"]) if study["n_events"] else float("nan")
    sleeve = tradable_sleeve(prices, events)

    return {
        "n_events": inf["n_events"],
        "mean_car_bps": inf["mean_car_bps"],
        "car_t": inf["car_t"],
        "car_p": inf["car_p"],
        "hit_rate": inf["hit_rate"],
        "aar_hac_t": inf["aar_hac_t"],
        "median_beta": round(float(np.median(study["betas"])), 3) if study["n_events"] else float("nan"),
        "placebo_std_bps": plc["placebo_std_bps"],
        "placebo_p": round(float(plc_p), 4),
        "sleeve_hedged_gross_bps": sleeve["hedged_gross"]["mean_bps"],
        "sleeve_hedged_gross_t": sleeve["hedged_gross"]["t"],
        "sleeve_hedged_net_bps": sleeve["hedged_net"]["mean_bps"],
        "sleeve_hedged_net_t": sleeve["hedged_net"]["t"],
    }
