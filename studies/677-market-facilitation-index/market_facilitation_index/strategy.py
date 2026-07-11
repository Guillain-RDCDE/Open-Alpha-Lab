"""Strategy + inference for Study 677 — Market Facilitation Index (BW-MFI).

Bill Williams' **Market Facilitation Index** (*Trading Chaos*, 1995; *New Trading
Dimensions*, 1998) is one bar-to-bar ratio::

    MFI(t) = (High(t) - Low(t)) / Volume(t)

— "how much price moved per unit of volume." Williams crosses its bar-to-bar *direction*
against **volume's** bar-to-bar direction to name four bar "colors":

* **Green**  — MFI up, Volume up.   The market is "in gear": price and volume agree, the
  move should **continue**.
* **Fade**   — MFI down, Volume down. Traders are leaving; the move has stalled.
* **Fake**   — MFI up, Volume down.  Price moved on thin participation — an "unsupported"
  bar, distrust the move.
* **Squat**  — MFI down, Volume up.  Heavy activity, no price progress — the market is
  "squatting" (coiling) before a **violent, direction-unspecified** move; the folklore
  reads it as an imminent **reversal** of the recent trend.

The claim under test: **the color predicts what happens next** — Green bars should see
tomorrow's return continue today's direction; Squat bars should see it reverse. We turn
that into a **continuation score** ``sign(ret_t) * fwd_ret_t`` (positive = continuation,
negative = reversal) and test its mean by state, plus two state-conditioned timers (ride
Green, sidestep Squat) raced net of costs against buy-and-hold.

One execution lag throughout: the color is known at the close of bar *t* (it needs only
that bar's own High/Low/Volume); the position it implies is applied with **one** ``shift``
so it earns the close-to-close return of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
STATES = ("green", "fade", "fake", "squat")


# --------------------------------------------------------------------------- #
# The BW-MFI classifier
# --------------------------------------------------------------------------- #
def raw_mfi(bars: pd.DataFrame) -> pd.Series:
    """BW-MFI = (High - Low) / Volume, using RAW (unadjusted) High/Low/Volume."""
    vol = bars["Volume"].replace(0.0, np.nan)
    return ((bars["High"] - bars["Low"]) / vol).rename("mfi")


def classify_states(bars: pd.DataFrame) -> pd.Series:
    """Bar-by-bar BW color: green / fade / fake / squat (NaN on the first bar or a tie)."""
    mfi = raw_mfi(bars)
    dmfi = mfi.diff()
    dvol = bars["Volume"].diff()
    state = pd.Series(np.nan, index=bars.index, dtype=object)
    state[(dmfi > 0) & (dvol > 0)] = "green"
    state[(dmfi < 0) & (dvol < 0)] = "fade"
    state[(dmfi > 0) & (dvol < 0)] = "fake"
    state[(dmfi < 0) & (dvol > 0)] = "squat"
    return state.rename("state")


# --------------------------------------------------------------------------- #
# Day frame — state, today's return, tomorrow's return, the continuation score
# --------------------------------------------------------------------------- #
def day_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day: state, ``ret`` (today), ``fwd_ret`` (tomorrow), and the
    continuation score ``sign(ret) * fwd_ret``.

    ``fwd_ret(t)`` is the return **earned by a position taken at the close of day t and
    held to the close of day t+1** — the study's one documented execution lag. The last
    row of the tape has no ``fwd_ret`` and is dropped by any stat that needs it.
    """
    close = bars["AdjClose"] if "AdjClose" in bars.columns else bars["Close"]
    df = pd.DataFrame(index=bars.index)
    df["close"] = close
    df["ret"] = close.pct_change()
    df["fwd_ret"] = df["ret"].shift(-1)
    df["state"] = classify_states(bars)
    df["cont_score"] = np.sign(df["ret"]) * df["fwd_ret"]
    return df


# --------------------------------------------------------------------------- #
# Inference primitives (self-contained — mirrors sibling studies 423/424)
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-stat that the mean of ``x`` differs from zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def sharpe(ret: pd.Series) -> float:
    """Annualised Sharpe of a daily return series."""
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


# --------------------------------------------------------------------------- #
# The headline — state-conditioned forward returns and continuation scores
# --------------------------------------------------------------------------- #
def state_stats(df: pd.DataFrame, states: tuple[str, ...] = STATES) -> dict:
    """Per-state forward-return and continuation-score means, Welch t vs ALL other days."""
    d = df.dropna(subset=["fwd_ret", "state"])
    out = {}
    fwd = d["fwd_ret"].to_numpy()
    cont = d["cont_score"].to_numpy()
    for s in states:
        mask = (d["state"] == s).to_numpy()
        a_fwd, b_fwd = fwd[mask], fwd[~mask]
        a_cont, b_cont = cont[mask], cont[~mask]
        k_up = int((a_cont > 0).sum())
        lo, hi = wilson_interval(k_up, mask.sum())
        out[s] = {
            "n": int(mask.sum()),
            "fwd_bps": float(np.nanmean(a_fwd) * 1e4),
            "rest_fwd_bps": float(np.nanmean(b_fwd) * 1e4),
            "welch_t_fwd": welch_t(a_fwd, b_fwd),
            "cont_bps": float(np.nanmean(a_cont) * 1e4),
            "rest_cont_bps": float(np.nanmean(b_cont) * 1e4),
            "welch_t_cont": welch_t(a_cont, b_cont),
            "cont_hit": k_up / mask.sum() if mask.sum() else float("nan"),
            "cont_hit_lo": lo, "cont_hit_hi": hi,
        }
    out["n_total"] = int(len(d))
    return out


def pooled_frame(bar_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack ``day_frame`` across a basket of tickers (adds a ``ticker`` column)."""
    frames = []
    for tkr, bars in bar_dict.items():
        d = day_frame(bars)
        d = d.dropna(subset=["fwd_ret", "state"]).copy()
        d["ticker"] = tkr
        frames.append(d)
    return pd.concat(frames, axis=0)


def permutation_pvalue_state(df: pd.DataFrame, state: str, n_perm: int = 2000,
                             seed: int = 677) -> dict:
    """Label-shuffle placebo for one state's continuation-score gap.

    Keeps the set of continuation scores fixed but reassigns the state label to a
    random subset of the same size as the real state's count, ``n_perm`` times. The
    one-sided p-value is the share of shuffles whose |gap| >= the observed |gap| —
    a permutation test for "is this state's mean any different from a random subset
    of days the same size."
    """
    d = df.dropna(subset=["fwd_ret", "state", "cont_score"])
    cont = d["cont_score"].to_numpy()
    mask = (d["state"] == state).to_numpy()
    n = mask.sum()
    obs_gap = float(np.nanmean(cont[mask]) - np.nanmean(cont[~mask]))
    rng = np.random.default_rng(seed)
    count = 0
    idx = np.arange(len(cont))
    for _ in range(n_perm):
        pick = rng.choice(idx, size=n, replace=False)
        pmask = np.zeros(len(cont), dtype=bool)
        pmask[pick] = True
        gap = np.nanmean(cont[pmask]) - np.nanmean(cont[~pmask])
        if abs(gap) >= abs(obs_gap):
            count += 1
    return {"state": state, "n": int(n), "obs_gap_bps": obs_gap * 1e4,
            "p_value": float((count + 1) / (n_perm + 1)), "n_perm": n_perm}


# --------------------------------------------------------------------------- #
# Timers — state-conditioned position rules, net of costs, one execution lag
# --------------------------------------------------------------------------- #
def green_filter_positions(df: pd.DataFrame) -> pd.Series:
    """Long only while today's bar is Green ("in gear" -> ride the continuation), flat
    otherwise. Position known at close t, applied with one shift by ``book_returns``."""
    return (df["state"] == "green").astype(float).rename("pos")


def squat_avoid_positions(df: pd.DataFrame) -> pd.Series:
    """Always long EXCEPT flat on a Squat bar (sidestep the folklore's "coiled, about to
    reverse" day). Isolates whether avoiding Squat days alone improves risk-adjusted
    return relative to simply holding the tape."""
    return (df["state"] != "squat").astype(float).rename("pos")


def sma_positions(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """50/200 SMA-crossover position — the obvious simpler trend benchmark."""
    c = df["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float).rename("pos")


def book_returns(df: pd.DataFrame, pos: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    """Daily strategy return for a position series, NET of one-way costs x NAV.

    One execution lag: the position known at the close of *t* earns the close-to-close
    return of *t+1* (``pos.shift(1)``). Turnover = ``|pos_t - pos_{t-1}|`` (one-way x
    NAV); long-only positions here, no borrow.
    """
    held = pos.shift(1)
    gross = held * df["ret"]
    turnover = held.diff().abs().fillna(0.0)
    cost = turnover * cost_bps * 1e-4
    return (gross - cost).dropna().rename("ret")


def buy_hold_returns(df: pd.DataFrame) -> pd.Series:
    return df["ret"].dropna().rename("bh")


def excess_vs_excess(strat: pd.Series, bench: pd.Series) -> dict:
    """Race a NET strategy return against buy-and-hold (both already excess-of-cash on a
    long-only, unlevered book), with a HAC t-stat on the daily return difference."""
    a, b = strat.align(bench, join="inner")
    diff = (a - b).to_numpy(dtype=float)
    return {
        "sharpe_strat": sharpe(a), "sharpe_bench": sharpe(b),
        "mean_diff_bps": float(np.nanmean(diff) * 1e4),
        "hac_t_diff": hac_tstat(diff),
        "n_days": int(np.isfinite(diff).sum()),
    }


def run_timer(df: pd.DataFrame, cost_bps: float = 1.0) -> dict:
    """Green filter + Squat-avoidance + the SMA(50/200) benchmark, all vs buy-and-hold."""
    bh = buy_hold_returns(df)
    out = {}
    for name, posfn in (("green", green_filter_positions),
                        ("squat_avoid", squat_avoid_positions),
                        ("sma", sma_positions)):
        pos = posfn(df)
        net = book_returns(df, pos, cost_bps=cost_bps)
        out[name] = excess_vs_excess(net, bh)
        out[name]["flips_per_yr"] = float(
            pos.shift(1).diff().abs().sum() / 2.0
            / max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
        )
    out["bh_sharpe"] = sharpe(bh)
    return out


def permutation_pvalue_timer(df: pd.DataFrame, cost_bps: float = 1.0, n_perm: int = 1000,
                             seed: int = 677) -> dict:
    """Sign-shuffle placebo for the Green-filter timer: how often does a randomly
    reordered Green/flat position series beat buy-and-hold by as much (Sharpe) as the
    real one?"""
    rng = np.random.default_rng(seed)
    pos = green_filter_positions(df)
    bh = buy_hold_returns(df)
    real_net = book_returns(df, pos, cost_bps=cost_bps)
    real = excess_vs_excess(real_net, bh)
    real_gap = real["sharpe_strat"] - real["sharpe_bench"]

    pos_vals = pos.to_numpy(dtype=float)
    idx = pos.index
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pos_vals)
        pp = pd.Series(perm, index=idx, name="pos")
        net = book_returns(df, pp, cost_bps=cost_bps)
        gap = sharpe(net) - real["sharpe_bench"]
        if np.isfinite(gap) and gap >= real_gap:
            count += 1
    return {"real_gap": float(real_gap), "p_value": float((count + 1) / (n_perm + 1)),
            "n_perm": int(n_perm)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bars: pd.DataFrame) -> dict:
    """Run the headline state_stats split on a synthetic world."""
    return state_stats(day_frame(bars))
