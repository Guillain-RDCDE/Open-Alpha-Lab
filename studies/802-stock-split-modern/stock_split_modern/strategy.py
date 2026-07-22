"""Strategy + inference for Study 802 — Stock-Split-Modern.

The claim: the classic post-split **drift** (stocks keep rising after a split; the
Ikenberry-Rankine-Stice 1996 anomaly) is alive in the **post-2020 mega-cap era** —
Tesla, Nvidia, Amazon, Alphabet all split and all soared afterwards.

The honest re-test. We measure **market-adjusted abnormal returns** (the stock's
total return minus SPY's over the identical window) at 1/3/6/12-month horizons after
the ex-split date, plus a tight [-1, +1] window *around* the ex-date. Market-adjustment
strips out the 2020-2024 bull run — the confound that makes "the stock went up after
its split" look like a signal when it is really just beta. What remains is graded on
the desk bar:

* **Signal** — is the mean abnormal CAR of the modern (post-2020) split cohort robustly
  > 0? The decisive statistic is a **Newey-West HAC one-sample t** on the date-ordered
  abnormal CARs (events cluster in calendar time — 2022 especially — so a plain t
  overstates independence). With single-digit N and huge idiosyncratic variance, the
  honest prior is that it will NOT clear t = 2.
* **Placebo** — abnormal CARs of the same names entered on random *non-split* dates.
  If the split carries no information, split-date CARs should not exceed the random-date
  cloud.
* **Tradability** — a costed timer buys at the ex-date close (the split is public at
  that instant; the first earned return is t+1 — the single documented execution lag),
  holds the horizon, pays one-way cost × NAV per leg. Long-only for forward splits;
  the primitive also supports shorts paying borrow, for completeness.

Every statistic below is on **abnormal** (market-adjusted, total-return) CARs unless a
name says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (21, 63, 126, 252)          # ~1m / 3m / 6m / 12m in trading days
HCOLS = ("car_1m", "car_3m", "car_6m", "car_12m")
RAWCOLS = ("raw_1m", "raw_3m", "raw_6m", "raw_12m")
ERA_SPLIT = "2020-01-01"


# --------------------------------------------------------------------------- #
# Event-panel assembly (real OR synthetic — identical code path)
# --------------------------------------------------------------------------- #
def build_event_panel(prices: pd.DataFrame, splits: pd.DataFrame,
                      market_col: str = "SPY",
                      horizons: tuple[int, ...] = HORIZONS,
                      era_split: str = ERA_SPLIT,
                      mega: tuple[str, ...] = ("TSLA", "NVDA", "AMZN", "GOOGL", "AAPL"),
                      ) -> pd.DataFrame:
    """One row per split event with **market-adjusted abnormal** forward CARs.

    Entry reference is the **ex-split (effective) date close** — publicly knowable at
    that instant. The forward buy-and-hold return runs from that close to the close
    ``h`` trading days later (first earned return is t+1: the single documented
    execution lag). The market leg is SPY's total return over the *identical* calendar
    window; the abnormal CAR is stock − market.

    Columns: ``ticker, event_date, ratio, era, is_mega``; per horizon ``raw_Nm``
    (stock), ``mkt_Nm`` (SPY), ``car_Nm`` (abnormal); and ``car_ex`` — the tight
    [-1, +1]-session abnormal return *around* the ex-date.
    """
    if market_col not in prices.columns:
        raise ValueError(f"market column {market_col!r} not in price panel")
    mkt = prices[market_col].dropna()
    rows = []
    for _, ev in splits.iterrows():
        tkr = ev["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        if len(s) < max(horizons) + 5:
            continue
        pos = int(np.searchsorted(s.index.values, np.datetime64(pd.Timestamp(ev["date"]))))
        if pos <= 1 or pos >= len(s):
            continue
        p0 = float(s.iloc[pos])
        if not np.isfinite(p0) or p0 <= 0:
            continue
        d0 = s.index[pos]

        def mkt_ret(a, b):
            va = float(mkt.asof(a)); vb = float(mkt.asof(b))
            if not (np.isfinite(va) and np.isfinite(vb)) or va <= 0:
                return np.nan
            return vb / va - 1.0

        rec = {"ticker": tkr, "event_date": pd.Timestamp(ev["date"]),
               "ratio": float(ev["ratio"]),
               "era": "post" if pd.Timestamp(ev["date"]) >= pd.Timestamp(era_split) else "pre",
               "is_mega": tkr in mega}

        # around-ex [-1, +1] abnormal
        if pos - 1 >= 0 and pos + 1 < len(s):
            raw_ex = float(s.iloc[pos + 1]) / float(s.iloc[pos - 1]) - 1.0
            m_ex = mkt_ret(s.index[pos - 1], s.index[pos + 1])
            rec["car_ex"] = raw_ex - m_ex if np.isfinite(m_ex) else np.nan
        else:
            rec["car_ex"] = np.nan

        for h, rawc, mc, cc in zip(horizons, RAWCOLS,
                                   ("mkt_1m", "mkt_3m", "mkt_6m", "mkt_12m"), HCOLS):
            end = pos + h
            if end >= len(s):
                rec[rawc] = rec[mc] = rec[cc] = np.nan
                continue
            raw = float(s.iloc[end]) / p0 - 1.0
            m = mkt_ret(d0, s.index[end])
            rec[rawc] = raw
            rec[mc] = m
            rec[cc] = raw - m if np.isfinite(m) else np.nan
        rows.append(rec)

    cols = ["ticker", "event_date", "ratio", "era", "is_mega", "car_ex"] \
        + list(RAWCOLS) + ["mkt_1m", "mkt_3m", "mkt_6m", "mkt_12m"] + list(HCOLS)
    return pd.DataFrame(rows, columns=cols).sort_values("event_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    """Welch t of mean(a) − mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_t(x, lags: int | None = None) -> float:
    """Newey-West (Bartlett) HAC one-sample t of mean(x) vs 0, on the array **in the
    order given** (pass date-ordered CARs so calendar clustering is respected)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Cohort summaries
# --------------------------------------------------------------------------- #
def horizon_summary(panel: pd.DataFrame, col: str = "car_3m") -> dict:
    """Headline abnormal-CAR stats for one horizon on a (sub)panel: n, mean/median %,
    std %, HAC one-sample t (date-ordered), win rate + Wilson interval."""
    sub = panel.dropna(subset=[col]).sort_values("event_date")
    r = sub[col].to_numpy(dtype=float)
    n = r.size
    if n < 2:
        return {"n": int(n), "mean_pct": float("nan"), "median_pct": float("nan"),
                "std_pct": float("nan"), "t_hac": float("nan"), "t_plain": float("nan"),
                "win": float("nan"), "win_lo": float("nan"), "win_hi": float("nan")}
    k = int((r > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"n": int(n), "mean_pct": float(r.mean() * 100),
            "median_pct": float(np.median(r) * 100), "std_pct": float(r.std(ddof=1) * 100),
            "t_hac": hac_t(r), "t_plain": one_sample_t(r),
            "win": k / n, "win_lo": lo, "win_hi": hi}


def cohort_table(panel: pd.DataFrame, col: str = "car_3m") -> pd.DataFrame:
    """Abnormal-CAR summary for the three cohorts the study cares about: all splits
    since 2010, pre-2020, and post-2020 (the modern era)."""
    rows = []
    for name, sub in [("all 2010+", panel),
                      ("pre-2020", panel[panel["era"] == "pre"]),
                      ("post-2020 (modern)", panel[panel["era"] == "post"]),
                      ("post-2020 mega-cap", panel[(panel["era"] == "post") & panel["is_mega"]])]:
        s = horizon_summary(sub, col)
        s["cohort"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("cohort")


def horizon_sweep(panel: pd.DataFrame, cohort: str = "post") -> pd.DataFrame:
    """For the chosen cohort, the abnormal-CAR summary across all four horizons."""
    if cohort == "post":
        sub = panel[panel["era"] == "post"]
    elif cohort == "pre":
        sub = panel[panel["era"] == "pre"]
    elif cohort == "mega":
        sub = panel[(panel["era"] == "post") & panel["is_mega"]]
    else:
        sub = panel
    rows = []
    for col, lbl in zip(HCOLS, ("1m (21d)", "3m (63d)", "6m (126d)", "12m (252d)")):
        s = horizon_summary(sub, col)
        s["horizon"] = lbl
        rows.append(s)
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Placebo — same names, random NON-split entry dates (matched control)
# --------------------------------------------------------------------------- #
def placebo_pvalue(prices: pd.DataFrame, panel: pd.DataFrame, col: str = "car_3m",
                   market_col: str = "SPY", n_draws: int = 2000, seed: int = 802,
                   cohort: str = "post") -> dict:
    """Right-tail placebo. For the chosen cohort, draw a same-size basket of random
    NON-split entry dates from the same names, compute each basket's mean abnormal CAR,
    and ask how often it beats the observed split-cohort mean.

    Removes the market run (abnormal returns) AND the name mix (same tickers), so what
    is left is: does *entering on the split date specifically* beat entering on a
    random date in the same name? p = share of random baskets whose mean ≥ observed.
    """
    if cohort == "post":
        sub = panel[panel["era"] == "post"]
    elif cohort == "mega":
        sub = panel[(panel["era"] == "post") & panel["is_mega"]]
    else:
        sub = panel
    sub = sub.dropna(subset=[col])
    obs = float(sub[col].mean())
    tickers = list(sub["ticker"])
    h = {"car_1m": 21, "car_3m": 63, "car_6m": 126, "car_12m": 252}[col]
    mkt = prices[market_col].dropna()
    rng = np.random.default_rng(seed)

    means = []
    for _ in range(n_draws):
        vals = []
        for tkr in tickers:
            s = prices[tkr].dropna()
            hi = len(s) - h - 2
            if hi <= 260:
                continue
            pos = int(rng.integers(260, hi))
            p0 = float(s.iloc[pos])
            raw = float(s.iloc[pos + h]) / p0 - 1.0
            m = float(mkt.asof(s.index[pos + h])) / float(mkt.asof(s.index[pos])) - 1.0
            vals.append(raw - m)
        if vals:
            means.append(float(np.mean(vals)))
    means = np.asarray(means)
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": float((means >= obs).mean()),
            "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(panel: pd.DataFrame, col: str = "car_3m", cost_bps: float = 5.0,
                cohort: str = "post", side: str = "long",
                borrow_bps_ann: float = 0.0) -> dict:
    """Buy (long) at the ex-date close, hold the horizon, exit; abnormal (market-hedged)
    return per event, net of 2 × one-way cost × NAV. ``side='short'`` flips the sign and
    charges ``borrow_bps_ann`` pro-rated over the holding horizon (shorts pay borrow).

    Reports gross/net mean %, one-sample t on the net per-event abnormal return,
    an event-frequency-annualised Sharpe, and the worst single event.
    """
    if cohort == "post":
        sub = panel[panel["era"] == "post"]
    elif cohort == "mega":
        sub = panel[(panel["era"] == "post") & panel["is_mega"]]
    else:
        sub = panel
    r = sub.dropna(subset=[col])[col].to_numpy(dtype=float)
    if r.size < 2:
        return {"n": int(r.size), "gross_pct": float("nan"), "net_pct": float("nan"),
                "t_net": float("nan"), "sharpe": float("nan"), "worst_pct": float("nan")}
    h = {"car_1m": 21, "car_3m": 63, "car_6m": 126, "car_12m": 252}[col]
    sign = 1.0 if side == "long" else -1.0
    rt_cost = 2.0 * cost_bps / 1e4
    borrow = (borrow_bps_ann / 1e4) * (h / TRADING_DAYS) if side == "short" else 0.0
    net = sign * r - rt_cost - borrow
    events_per_year = r.size / ((panel["event_date"].max() - panel["event_date"].min()).days / 365.25 or 1)
    sd = net.std(ddof=1)
    sharpe = net.mean() / sd * np.sqrt(events_per_year) if sd > 0 else float("nan")
    return {"n": int(r.size), "gross_pct": float(sign * r.mean() * 100),
            "net_pct": float(net.mean() * 100), "t_net": one_sample_t(net),
            "sharpe": float(sharpe), "worst_pct": float(net.min() * 100),
            "events_per_year": float(events_per_year)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, splits: pd.DataFrame,
                     col: str = "car_3m", market_col: str = "SPY") -> dict:
    """Run the full abnormal-CAR panel on a synthetic world and return the HAC t and
    mean at one horizon (the null must not fire; a planted drift must light up)."""
    panel = build_event_panel(prices, splits, market_col=market_col,
                              era_split="1900-01-01")  # all synthetic events one cohort
    s = horizon_summary(panel, col)
    return {"n": s["n"], "mean_pct": s["mean_pct"], "t_hac": s["t_hac"],
            "t_plain": s["t_plain"]}
