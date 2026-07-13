"""Strategy + inference for Study 738 — Pollen-Season.

The claim, steelmanned: **the owners of the big allergy brands carry a repeatable spring
seasonal.** Hay-fever season (US tree/grass pollen, ~March→May) drives a predictable
demand spike for antihistamines and nasal sprays, so a basket of the listed brand owners
(Claritin/Bayer, Allegra/Sanofi, Zyrtec-Benadryl/Kenvue, Flonase/Haleon, plus the
private-label maker Perrigo) ought to *beat the market* through the pollen window, year
after year.

The unit of analysis is **one spring window per year** — an independent, non-overlapping
event (the 2019 window can't leak into the 2020 window). So the primary statistic is a
**one-sample t** of the per-year abnormal return (basket window return minus the market's
window return) across the ~30 sample years — NOT a daily panel, whose thousands of
autocorrelated rows would badly overstate the degrees of freedom.

The window is a pure **calendar-known** rule (fixed dates, ``data.SEASON_*_MMDD``):
enter at the last session of February, exit at the last session on/before May 31. Because
the dates are known years in advance, there is **no execution lag to apply** — you never
learn "spring is coming" from a print; the calendar tells you (the same free pass a
turn-of-month study gets).

Everything downstream is offline once the tape is cached: the abnormal-return table, the
Wilson hit rate, a random-window placebo (multi-seed), a costed long/short "trade it"
timer (both legs charged, the short pays borrow), and a synthetic positive control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

COST_BPS = 5.0        # one-way, per leg
BORROW_BPS_ANNUAL = 50.0  # annualised borrow on the short market leg (large-cap ETF)


# --------------------------------------------------------------------------- #
# Window resolution on a trading calendar (calendar-known -> no execution lag)
# --------------------------------------------------------------------------- #
def window_bounds(cal: pd.DatetimeIndex, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(entry, exit) sessions for ``year``'s pollen window on trading calendar ``cal``.

    ``entry`` = the last session strictly before Mar 1 (i.e. end of February — the season
    has not begun). ``exit`` = the last session on/before May 31 (the season is ending).
    Both are real trading dates on ``cal``; ``None`` if the calendar does not span them.
    """
    lo = pd.Timestamp(year, dt.SEASON_START_MMDD[0], dt.SEASON_START_MMDD[1])
    hi = pd.Timestamp(year, dt.SEASON_END_MMDD[0], dt.SEASON_END_MMDD[1])
    before = cal[cal < lo]
    upto = cal[cal <= hi]
    if len(before) == 0 or len(upto) == 0:
        return None
    entry, exit_ = before[-1], upto[-1]
    if exit_ <= entry:
        return None
    return entry, exit_


def _win_ret(close: pd.Series, entry: pd.Timestamp, exit_: pd.Timestamp) -> float:
    """Total-return over [entry, exit] if the series has both dates, else NaN."""
    if entry in close.index and exit_ in close.index:
        return float(close.loc[exit_] / close.loc[entry] - 1.0)
    return float("nan")


def basket_window_return(prices: dict[str, pd.Series], tickers, entry, exit_) -> tuple[float, int]:
    """Equal-weight window return over whichever ``tickers`` cover [entry, exit].

    Names not yet listed (a spin-off before its first listing, e.g. Kenvue for 2023) have
    no close at ``entry`` -> NaN -> dropped. Returns (basket_return, n_names_used); the
    count is the study's honest coverage flag, never silently back-filled.
    """
    rets = [_win_ret(prices[t], entry, exit_) for t in tickers]
    rets = [r for r in rets if np.isfinite(r)]
    if not rets:
        return float("nan"), 0
    return float(np.mean(rets)), len(rets)


# --------------------------------------------------------------------------- #
# The per-year abnormal-return table (one independent event per row)
# --------------------------------------------------------------------------- #
def build_spread_table(prices: dict[str, pd.Series], years, tickers=dt.ALLERGY_TICKERS,
                       market: str = dt.MARKET, staples: str = dt.STAPLES,
                       cost_bps: float = COST_BPS, borrow_bps: float = BORROW_BPS_ANNUAL,
                       min_names: int = 2) -> pd.DataFrame:
    """One row per spring window: basket vs market abnormal return + a costed L/S spread.

    Columns: ``year, entry, exit, k`` (window session count), ``n_names``, ``basket_ret``,
    ``mkt_ret``, ``abn`` (= basket_ret - mkt_ret, the headline), ``xlp_ret``, ``abn_xlp``
    (basket vs consumer-staples, the fairer-benchmark robustness), ``ls_gross`` (= abn),
    ``ls_net`` (abn minus 4 one-way costs and the short leg's borrow). A year is included
    only if the basket covers >= ``min_names`` names that spring.
    """
    cal = prices[market].index
    rows = []
    for y in years:
        wb = window_bounds(cal, y)
        if wb is None:
            continue
        entry, exit_ = wb
        b_ret, n_names = basket_window_return(prices, tickers, entry, exit_)
        if n_names < min_names or not np.isfinite(b_ret):
            continue
        m_ret = _win_ret(prices[market], entry, exit_)
        x_ret = _win_ret(prices[staples], entry, exit_)
        if not np.isfinite(m_ret):
            continue
        k = int(cal.get_loc(exit_) - cal.get_loc(entry))
        abn = b_ret - m_ret
        borrow = borrow_bps / 1e4 * (k / 252.0)     # short SPY financed over the window
        ls_net = abn - 4.0 * cost_bps / 1e4 - borrow
        rows.append(dict(
            year=y, entry=entry.date().isoformat(), exit=exit_.date().isoformat(),
            k=k, n_names=n_names, basket_ret=b_ret, mkt_ret=m_ret,
            abn=abn, xlp_ret=x_ret, abn_xlp=(b_ret - x_ret if np.isfinite(x_ret) else np.nan),
            ls_gross=abn, ls_net=ls_net,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives (events = years, independent & non-overlapping)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> dict:
    """Mean + one-sample t of ``x`` vs 0 — the correct unit for independent yearly events."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    sd = x.std(ddof=1)
    se = sd / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(sd),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a, b) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hit_rate(x) -> dict:
    """Share of years the basket beat the market, with a Wilson 95% interval."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-window placebo — is the observed spring seasonal inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_distribution(prices: dict[str, pd.Series], table: pd.DataFrame,
                         tickers=dt.ALLERGY_TICKERS, market: str = dt.MARKET,
                         n_seeds: int = 20, n_draws_per_seed: int = 250,
                         base_seed: int = 738) -> np.ndarray:
    """Null of the headline mean abnormal return, from random NON-spring windows.

    For each included year we redraw a same-length window anchored at a random session
    that does NOT overlap that year's real pollen window, recompute the basket-minus-market
    abnormal return over exactly the names that cover the random window, and average across
    the same years. Repeating ``n_seeds x n_draws_per_seed`` times traces out the
    distribution of "what a random calendar of same-size windows produces anyway". A real
    spring seasonal must sit in the right tail of this cloud.
    """
    cal = prices[market].index
    npos = len(cal)
    spring_pos = []
    specs = []
    for _, r in table.iterrows():
        entry = pd.Timestamp(r["entry"]); exit_ = pd.Timestamp(r["exit"])
        p0, p1 = cal.get_loc(entry), cal.get_loc(exit_)
        spring_pos.append((p0, p1))
        specs.append(int(r["k"]))
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = []
            for (p0, p1), k in zip(spring_pos, specs):
                lo, hi = k, npos - k - 1
                if hi <= lo:
                    continue
                # draw an anchor whose [a, a+k] window avoids this year's spring window
                for _try in range(8):
                    a = int(rng.integers(lo, hi))
                    if a + k < p0 - k or a > p1 + k:
                        break
                entry_d, exit_d = cal[a], cal[a + k]
                b, n_names = basket_window_return(prices, tickers, entry_d, exit_d)
                m = _win_ret(prices[market], entry_d, exit_d)
                if n_names >= 1 and np.isfinite(b) and np.isfinite(m):
                    vals.append(b - m)
            if vals:
                means.append(float(np.mean(vals)))
    return np.asarray(means)


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "right") -> float:
    """Empirical one-sided p-value of ``observed`` within the placebo draws."""
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "right":
        return float((placebo >= observed).mean())
    return float((placebo <= observed).mean())


def block_bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05,
                       seed: int = 738) -> tuple[float, float]:
    """Percentile CI on the mean abnormal return (years resampled with replacement)."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# --------------------------------------------------------------------------- #
# Calendar-month seasonality (the "is spring actually special?" chart)
# --------------------------------------------------------------------------- #
def basket_daily_returns(prices: dict[str, pd.Series], tickers=dt.ALLERGY_TICKERS) -> pd.Series:
    """Equal-weight daily return of whichever basket names trade each day (skipna)."""
    rets = pd.DataFrame({t: prices[t].pct_change() for t in tickers})
    basket = rets.mean(axis=1, skipna=True)
    basket[rets.notna().sum(axis=1) == 0] = np.nan
    return basket


def month_seasonality(prices: dict[str, pd.Series], tickers=dt.ALLERGY_TICKERS,
                      market: str = dt.MARKET) -> pd.DataFrame:
    """Mean daily abnormal return (basket - market), grouped by calendar month.

    A blunt cross-check on the window test: if the pollen story is real, the spring
    months should stand out above the other eleven. Returns a frame indexed 1..12 with
    ``mean_abn_bps`` and each month's one-sample t (over that month's daily abnormal
    returns — a coarse read, the window test is the primary).
    """
    b = basket_daily_returns(prices, tickers)
    m = prices[market].pct_change()
    common = b.index.intersection(m.index)
    abn = (b.reindex(common) - m.reindex(common)).dropna()
    rows = []
    for mo in range(1, 13):
        x = abn[abn.index.month == mo].to_numpy()
        st = one_sample_t(x)
        rows.append({"month": mo, "mean_abn_bps": st["mean"] * 1e4, "t": st["t"], "n": st["n"]})
    return pd.DataFrame(rows).set_index("month")


# --------------------------------------------------------------------------- #
# The tradable timer — long basket / short market over the pollen window
# --------------------------------------------------------------------------- #
def timer_stats(table: pd.DataFrame) -> dict:
    """Headline stats for the seasonal long/short spread, gross and net of costs+borrow."""
    g = one_sample_t(table["ls_gross"].to_numpy())
    n = one_sample_t(table["ls_net"].to_numpy())
    hr = hit_rate(table["ls_gross"].to_numpy())
    return {
        "n": g["n"],
        "gross_mean_bps": g["mean"] * 1e4, "gross_t": g["t"],
        "net_mean_bps": n["mean"] * 1e4, "net_t": n["t"],
        "hit_k": hr["k"], "hit_n": hr["n"], "hit_rate": hr["rate"],
        "hit_lo": hr["lo"], "hit_hi": hr["hi"],
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (machinery proof — never cited for the real stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int) -> dict:
    """Run the headline window one-sample-t on a synthetic tape with a planted bump."""
    basket, market = dt.synthetic_world(bump=bump, seed=seed)
    prices = {"BASKET": basket, dt.MARKET: market}
    years = sorted({d.year for d in basket.index})[1:-1]   # drop the ragged first/last year
    cal = market.index
    abn = []
    for y in years:
        wb = window_bounds(cal, y)
        if wb is None:
            continue
        entry, exit_ = wb
        b = _win_ret(basket, entry, exit_)
        m = _win_ret(market, entry, exit_)
        if np.isfinite(b) and np.isfinite(m):
            abn.append(b - m)
    return one_sample_t(np.asarray(abn))
