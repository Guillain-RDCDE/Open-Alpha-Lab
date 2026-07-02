"""Strategy + inference for Study 546 — Nobel-Announcement-Drift.

The folklore, stated cleanly: when a Nobel *science* prize is announced (early October), the
thematically related sector should catch a news-attention bid and **drift** upward in the days
that follow — pharma/biotech (XLV, IBB) after **Medicine**, tech/semis (XLK, SMH) after
**Physics** & **Chemistry**. We test it as a textbook event study.

The measurements:

1. **Abnormal returns (market model).** For each sector ETF we estimate a market-model beta on
   an *estimation window* strictly BEFORE the event (so no look-ahead), then the abnormal return
   on any day is ``r_sector - (alpha + beta * r_SPY)``. The drift we care about is *sector*
   movement net of the market — a Nobel prize that just happened to fall in a good week for the
   whole market is not a Nobel effect.

2. **The cumulative abnormal return (CAR).** For each (event, sector) pair, sum the abnormal
   returns over the post-event window (t0, t0+H]. t0 = the first trading session on/after the
   announcement (the one documented execution lag). We enter at that close and hold H days.

3. **The headline statistic.** The mean CAR across all mapped (prize, sector) events, with a
   one-sample t vs 0. Drift folklore predicts a *positive* mean CAR at t >= 2.

4. **The placebo null.** Re-run the identical CAR machinery on *random October dates* (same
   month, same sectors, same H) many times; the p-value is the share of random-October draws
   whose mean CAR >= the observed. "Could a random October week have looked this good?"

5. **Robustness.** Sweep the horizon H (3/5/10/21 days), the entry lag (t0 vs t0+1), and split
   pre/post-2012 — a drift that only exists at one horizon or one era is not a signal.

6. **Costs.** The tradable expression is: buy the mapped sector ETF at t0 close, sell at t0+H.
   Charge a round-trip one-way cost per event (large, liquid ETFs -> small bps). Long-only, so
   no borrow. Gross and net both reported.

7. **Synthetic positive control.** A deterministic world with a *planted* post-event drift; the
   engine must recover a positive mean CAR / t as the drift grows and stay flat at the null,
   averaged over >= 20 seeds (house rule).

The decisive number is whether the mean post-event CAR clears t >= 2 on the REAL tape AND beats
the random-October placebo AND is stable across horizons/eras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns + market model
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from an adjusted-close panel (rows dropped where all-NaN)."""
    return prices.pct_change().dropna(how="all")


def abnormal_returns(
    rets: pd.DataFrame, sector: str, benchmark: str, est_len: int = 120
) -> pd.Series:
    """Rolling-market-model abnormal-return series for ``sector`` net of ``benchmark``.

    For each day, (alpha, beta) is estimated on the ``est_len`` sessions ENDING strictly before
    that day (a trailing window that is fully public at the day's open — no look-ahead), and the
    abnormal return is ``r_sector - (alpha + beta * r_benchmark)``. Precomputing this once makes
    every event CAR (real or placebo) an O(1) rolling-sum lookup and keeps the estimation window
    honestly trailing. Days without a full estimation window are NaN.

    Vectorised via rolling covariance: ``beta = cov(sector, bench) / var(bench)`` and
    ``alpha = mean(sector) - beta * mean(bench)`` over the trailing window, both shifted one bar
    so day t uses only data through t-1.
    """
    pair = rets[[sector, benchmark]].dropna()
    s = pair[sector]
    m = pair[benchmark]
    roll_cov = s.rolling(est_len).cov(m)
    roll_var = m.rolling(est_len).var()
    roll_ms = s.rolling(est_len).mean()
    roll_mm = m.rolling(est_len).mean()
    beta = (roll_cov / roll_var).shift(1)     # strictly-before-t estimation window
    alpha = (roll_ms - (roll_cov / roll_var) * roll_mm).shift(1)
    ab = s - (alpha + beta * m)
    return ab.reindex(rets.index)


def event_car(
    ab: pd.Series,
    event_date: pd.Timestamp,
    horizon: int = 10,
    entry_lag: int = 0,
) -> float:
    """Cumulative abnormal return over the post-event window, from a precomputed AR series.

    t0 = the first trading session on/after ``event_date``, shifted by ``entry_lag`` (0 = enter
    at t0 close, 1 = enter one bar later). The CAR sums abnormal returns over the ``horizon``
    sessions *after* t0, i.e. (t0, t0+H]. Returns NaN if the window runs off the tape or has any
    missing abnormal return (incomplete estimation window).
    """
    idx = ab.index
    pos = idx.searchsorted(event_date, side="left")
    if pos >= len(idx):
        return float("nan")
    t0 = pos + entry_lag
    if t0 + horizon >= len(idx) or t0 < 0:
        return float("nan")
    win = ab.iloc[t0 + 1 : t0 + 1 + horizon]
    if win.isna().any() or len(win) < horizon:
        return float("nan")
    return float(win.sum())


def _ab_cache(rets: pd.DataFrame, prize_sector: dict, benchmark: str, est_len: int = 120) -> dict:
    """Precompute the abnormal-return series for every sector used in the mapping."""
    secs = sorted({s for v in prize_sector.values() for s in v if s in rets.columns})
    return {s: abnormal_returns(rets, s, benchmark, est_len=est_len) for s in secs}


def event_car_table(
    rets: pd.DataFrame,
    events: pd.DataFrame,
    prize_sector: dict,
    benchmark: str,
    horizon: int = 10,
    entry_lag: int = 0,
    est_len: int = 120,
    ab_cache: dict | None = None,
) -> pd.DataFrame:
    """CAR for every mapped (event, sector) pair.

    ``events`` has columns date/prize/year; ``prize_sector`` maps each prize to its sectors.
    Returns a long frame: date, prize, sector, year, car. NaN CARs (windows off-tape) dropped.
    ``ab_cache`` (from ``_ab_cache``) is reused across placebo draws for speed.
    """
    if ab_cache is None:
        ab_cache = _ab_cache(rets, prize_sector, benchmark, est_len=est_len)
    rows = []
    for ev in events.to_dict("records"):
        prize = ev["prize"]
        for sec in prize_sector.get(prize, []):
            if sec not in ab_cache:
                continue
            car = event_car(ab_cache[sec], ev["date"], horizon=horizon, entry_lag=entry_lag)
            rows.append(
                {"date": ev["date"], "prize": prize, "sector": sec,
                 "year": ev["year"], "car": car}
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.dropna(subset=["car"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def one_sample_t(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def car_summary(car_table: pd.DataFrame) -> dict:
    """Mean CAR across all mapped events, its one-sample t, and n."""
    if car_table.empty:
        return {"mean_car": float("nan"), "t": float("nan"), "n": 0}
    c = car_table["car"].to_numpy(dtype=float)
    return {"mean_car": float(np.nanmean(c)), "t": one_sample_t(c), "n": int(len(c))}


# --------------------------------------------------------------------------- #
# Placebo — random October dates
# --------------------------------------------------------------------------- #
def _october_business_days(rets: pd.DataFrame, years) -> dict[int, list[pd.Timestamp]]:
    """Trading days in October for each year present in the tape."""
    idx = rets.index
    oct_mask = idx.month == 10
    out: dict[int, list[pd.Timestamp]] = {}
    for y in years:
        days = idx[oct_mask & (idx.year == y)]
        if len(days) >= 5:
            out[int(y)] = list(days)
    return out


def placebo_pvalue(
    rets: pd.DataFrame,
    events: pd.DataFrame,
    prize_sector: dict,
    benchmark: str,
    horizon: int = 10,
    n_draws: int = 2000,
    seed: int = 546,
) -> dict:
    """Random-October placebo: relocate each event to a random October day of the same year.

    Each draw rebuilds the full CAR machinery with each announcement replaced by a random
    trading day in the SAME October (same sectors, same horizon), and records the mean CAR.
    p-value = share of random-October draws whose mean CAR >= the observed mean CAR. The honest
    "could a random October week have looked this good?" null.
    """
    ab_cache = _ab_cache(rets, prize_sector, benchmark)
    obs = car_summary(event_car_table(
        rets, events, prize_sector, benchmark, horizon=horizon, ab_cache=ab_cache))["mean_car"]
    if np.isnan(obs):
        return {"obs": float("nan"), "placebo_mean": float("nan"), "p_value": float("nan")}
    years = sorted(events["year"].unique())
    oct_days = _october_business_days(rets, years)
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    ev_list = events.to_dict("records")
    for i in range(n_draws):
        fake = []
        for ev in ev_list:
            days = oct_days.get(int(ev["year"]))
            if not days:
                continue
            d = days[rng.integers(len(days))]
            fake.append({"date": d, "prize": ev["prize"], "year": ev["year"]})
        fake_df = pd.DataFrame(fake)
        m = car_summary(event_car_table(
            rets, fake_df, prize_sector, benchmark, horizon=horizon, ab_cache=ab_cache))["mean_car"]
        means[i] = m
    p = float(np.nanmean(means >= obs - 1e-15))
    return {"obs": float(obs), "placebo_mean": float(np.nanmean(means)), "p_value": p}


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def horizon_sweep(
    rets: pd.DataFrame,
    events: pd.DataFrame,
    prize_sector: dict,
    benchmark: str,
    horizons=(3, 5, 10, 21),
    entry_lag: int = 0,
) -> pd.DataFrame:
    """Mean CAR + t at several post-event horizons."""
    rows = []
    for h in horizons:
        s = car_summary(event_car_table(
            rets, events, prize_sector, benchmark, horizon=h, entry_lag=entry_lag))
        rows.append({"horizon": h, "mean_car": s["mean_car"], "t": s["t"], "n": s["n"]})
    return pd.DataFrame(rows).set_index("horizon")


def era_split(
    rets: pd.DataFrame,
    events: pd.DataFrame,
    prize_sector: dict,
    benchmark: str,
    horizon: int = 10,
    split_year: int = 2013,
) -> dict:
    """Pre/post-split mean CAR (the stability-across-eras check)."""
    tab = event_car_table(rets, events, prize_sector, benchmark, horizon=horizon)
    pre = tab[tab["year"] < split_year]
    post = tab[tab["year"] >= split_year]
    return {
        "full": car_summary(tab),
        "pre": car_summary(pre),
        "post": car_summary(post),
        "split_year": split_year,
    }


# --------------------------------------------------------------------------- #
# Costs — the tradable expression
# --------------------------------------------------------------------------- #
def net_car(car_table: pd.DataFrame, cost_bps: float = 2.0) -> dict:
    """Mean CAR net of a round-trip one-way cost per event (buy at t0, sell at t0+H).

    Long-only sector ETFs (no borrow). Each event is one buy + one sell -> a round trip of
    ``2 * cost_bps``. Large, liquid ETFs, so the default is a small 2 bps one-way. Reports gross
    and net mean CAR and the net one-sample t.
    """
    if car_table.empty:
        return {"gross": float("nan"), "net": float("nan"), "net_t": float("nan")}
    round_trip = 2.0 * cost_bps * 1e-4
    c = car_table["car"].to_numpy(dtype=float)
    net = c - round_trip
    return {
        "gross": float(np.nanmean(c)),
        "net": float(np.nanmean(net)),
        "net_t": one_sample_t(net),
        "round_trip": float(round_trip),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (seed-robust)
# --------------------------------------------------------------------------- #
def synthetic_mean_t(
    data_mod, drift: float, horizon: int = 10, n_seeds: int = 25, base_seed: int = 546
) -> float:
    """Average the event-CAR one-sample t over ``n_seeds`` synthetic worlds for a planted drift.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky seed can manufacture significance. Returns the mean CAR-t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        px, events, _ = data_mod.synthetic_world(drift=drift, horizon=horizon, seed=s)
        rets = daily_returns(px)
        tab = event_car_table(rets, events, {"sector": ["SECTOR"]}, "SPY", horizon=horizon)
        ts.append(car_summary(tab)["t"])
    return float(np.nanmean(ts))
