"""Measurement + strategy for Study 923 — The Cash Lag.

Two questions, tested separately and stamped separately.

**A. Does the lag exist, and can we measure it?** For each cash vehicle we build
a *labelled proxy* for its realised yield — the trailing ``window``-day total
return, annualised (``realised_yield``) — and relate it to the 13-week bill quote
``^IRX``:

- ``effective_duration`` regresses the vehicle's **daily** total return on the
  **daily** change in ^IRX. The coefficient x (-100) is the vehicle's realised
  duration in years. This is *window-free*: no trailing average, so no averaging
  artefact. It is the cleanest measurement in the study.
- ``lag_profile`` runs a distributed-lag HAC regression of the change in the
  realised-yield proxy on lagged changes in ^IRX, giving a pass-through profile
  (the coefficients should sum to ~1 — full eventual pass-through), a peak lag
  and a positive-weight centroid lag.
- ``lag_window_sweep`` re-runs that centroid across several proxy windows. Part
  of any measured "lag" is mechanical: a trailing ``w``-day return averages the
  rate over ``w`` days, so the centroid drifts with ``w``. The sweep separates the
  artefact from the residual, genuine repricing lag. We report both.

**B. Can you trade it?** ``rate_direction_signal`` takes the sign of the trailing
``lookback``-day change in ^IRX through day ``t`` and lags it one day, so it is
acted on at ``t+1`` (the study's single, documented execution lag). The rule:
rates rising -> hold the *fast* repricer (zero-duration USFR, which takes no
mark-down and picks the new rate up within a week); rates falling -> hold the
*slow* one (SHV, which keeps a stale-high book yield and banks a duration gain).
``switch_backtest`` runs it long-only (no shorting anywhere, therefore no borrow
cost) with a one-way cost charged on NAV at each switch, and measures everything
**excess-of-cash**, where cash is BIL's own total return — so the race is
excess-of-cash against excess-of-cash by construction.

Two placebos guard the trading half: ``reversed_signal`` (the same rule with the
sign flipped) and ``placebo_signal`` (a random switch matched to the real rule's
*turnover*, not merely its frequency — matching frequency alone would churn far
more and lose on cost for uninteresting reasons).

Returns are **simple** (arithmetic) throughout so a long-only one-vehicle
portfolio return is exactly the held vehicle's return minus the switch cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _auto_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = _auto_lags(n)
    u = x - x.mean()
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - l / (lags + 1.0)) * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def hac_ols(y, X, lags: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """OLS with a Newey-West covariance. Returns (coefficients, standard errors).

    An intercept is prepended, so ``coef[0]`` is the constant and ``coef[1:]`` are
    the slopes in the column order of ``X``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if lags is None:
        lags = _auto_lags(n)
    Xd = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    u = y - Xd @ beta
    Z = Xd * u[:, None]
    S = Z.T @ Z / n
    for l in range(1, lags + 1):
        A = Z[l:].T @ Z[:-l] / n
        S += (1.0 - l / (lags + 1.0)) * (A + A.T)
    XX = np.linalg.inv(Xd.T @ Xd / n)
    V = XX @ S @ XX / n
    return beta, np.sqrt(np.clip(np.diag(V), 0.0, None))


def block_bootstrap_mean_ci(
    x,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 923,
    alpha: float = 0.05,
    scale: float = TRADING_DAYS * 1e4,
) -> dict:
    """Circular block-bootstrap CI for the mean of a daily series.

    ``scale`` converts the daily mean into the reported unit (default: annualised
    basis points). Blocks of ``block`` consecutive days preserve the strong daily
    autocorrelation of cash-vehicle returns.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(r.mean() * scale),
        "ci_low": float(lo * scale), "ci_high": float(hi * scale),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n), "n_boot": int(n_boot), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# A. Measuring the lag
# --------------------------------------------------------------------------- #
def realised_yield(close: pd.Series, window: int = 21) -> pd.Series:
    """Labelled proxy for a vehicle's realised yield: trailing total return, annualised (%).

    PROXY: this is *not* an SEC 30-day yield or a book yield — it is the vehicle's
    own realised total return over the last ``window`` sessions, compounded to a
    year and expressed in percent. It therefore contains both the carry (what the
    fund earns) and the mark-to-market (what a rate move does to its NAV), which
    is exactly what a holder lives through.
    """
    s = pd.Series(close).astype(float).dropna()
    return ((s / s.shift(window)) ** (TRADING_DAYS / window) - 1.0).mul(100.0).rename("realised_yield")


def effective_duration(close: pd.Series, rate: pd.Series, hac_lags: int = 5) -> dict:
    """Realised effective duration (years) from daily returns on daily rate changes.

    Regresses the vehicle's daily simple total return on the daily change in the
    rate quote (percentage points). The slope is ``-duration / 100``, so duration
    in years is ``-100 * slope``. Window-free: no trailing average is involved, so
    this measurement carries no averaging artefact.

    ``tstat`` is signed on the *duration*, not on the raw slope: a positive t means
    a reliably positive duration (the vehicle really is marked down when rates rise).
    """
    r = pd.Series(close).astype(float).dropna().pct_change()
    d = pd.Series(rate).astype(float).dropna().diff()
    j = pd.concat([r.rename("r"), d.rename("d")], axis=1, join="inner").dropna()
    if len(j) < 30:
        return {"duration_years": float("nan"), "tstat": float("nan"), "n": len(j)}
    beta, se = hac_ols(j["r"].to_numpy(), j["d"].to_numpy(), lags=hac_lags)
    return {
        "duration_years": float(-100.0 * beta[1]),
        "tstat": float(-beta[1] / se[1]) if se[1] > 0 else float("nan"),
        "n": int(len(j)),
    }


DEFAULT_LAGS = tuple(range(0, 64, 7))


def lag_profile(
    close: pd.Series,
    rate: pd.Series,
    window: int = 21,
    lags=DEFAULT_LAGS,
    hac_lags: int = 42,
) -> dict:
    """Distributed-lag pass-through of ^IRX into a vehicle's realised-yield proxy.

    Regresses the ``window``-day change in the realised-yield proxy on the
    ``window``-day change in the rate quote at each lag in ``lags``. Returns the
    HAC coefficients and t-stats, their sum (eventual pass-through, ~1 if the
    vehicle fully inherits the bill rate), the peak lag, and the positive-weight
    centroid lag (a single "how many days behind?" number).

    Read ``coefs[0]`` — the contemporaneous term — as the duration shock: a
    vehicle with real duration is marked *down* on the day rates rise, so this
    coefficient is negative and scales with WAM.
    """
    lags = list(lags)
    y = realised_yield(close, window=window).diff(window)
    d = pd.Series(rate).astype(float).dropna().diff(window)
    X = pd.concat({f"l{k}": d.shift(k) for k in lags}, axis=1)
    j = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    if len(j) < 5 * len(lags):
        nan = [float("nan")] * len(lags)
        return {"lags": lags, "coefs": nan, "tstats": nan, "sum_beta": float("nan"),
                "peak_lag": float("nan"), "centroid_lag": float("nan"),
                "contemporaneous": float("nan"), "contemporaneous_t": float("nan"),
                "n": len(j), "window": window}
    beta, se = hac_ols(j["y"].to_numpy(), j[[f"l{k}" for k in lags]].to_numpy(), lags=hac_lags)
    coefs = beta[1:]
    tstats = np.where(se[1:] > 0, coefs / np.where(se[1:] > 0, se[1:], 1.0), np.nan)
    pos = np.clip(coefs, 0.0, None)
    centroid = float((pos * np.array(lags, dtype=float)).sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    return {
        "lags": lags,
        "coefs": [float(c) for c in coefs],
        "tstats": [float(t) for t in tstats],
        "sum_beta": float(coefs.sum()),
        "peak_lag": int(lags[int(np.argmax(coefs))]),
        "centroid_lag": centroid,
        "contemporaneous": float(coefs[0]),
        "contemporaneous_t": float(tstats[0]),
        "n": int(len(j)),
        "window": int(window),
    }


def lag_window_sweep(close: pd.Series, rate: pd.Series, windows=(5, 10, 21, 42)) -> list[dict]:
    """Centroid lag as a function of the proxy window — the averaging-artefact check.

    A trailing ``w``-day return mechanically averages the rate over ``w`` days, so
    some of the measured lag is the *ruler*, not the vehicle. If the centroid rose
    one-for-one with ``w`` the whole "lag" would be an artefact; a centroid that
    stays well above ``w`` at small ``w`` is genuine repricing delay.
    """
    return [dict(window=int(w), **{k: lag_profile(close, rate, window=w)[k]
                                   for k in ("centroid_lag", "peak_lag", "sum_beta", "n")})
            for w in windows]


# --------------------------------------------------------------------------- #
# B. Trading the lag
# --------------------------------------------------------------------------- #
def rate_direction_signal(rate: pd.Series, lookback: int = 21) -> pd.Series:
    """+1 when the rate quote has risen over the trailing ``lookback`` days, else -1.

    The change is measured on data through day ``t`` and the series is shifted one
    day, so the position is taken at ``t+1``. That single shift is the study's
    ONLY execution lag.
    """
    d = pd.Series(rate).astype(float).dropna().diff(lookback)
    sig = np.sign(d).replace(0.0, -1.0)
    return sig.shift(1).rename("signal")


def reversed_signal(signal: pd.Series) -> pd.Series:
    """Placebo 1: the same rule with the sign flipped (hold the slow one when rates rise)."""
    return (-pd.Series(signal)).rename("reversed_signal")


def placebo_signal(signal: pd.Series, seed: int = 0) -> pd.Series:
    """Placebo 2: a random switch matched to ``signal``'s *turnover* and long-share.

    Matching the switch *count* (not merely the fraction of +1 days) is the fair
    null: an i.i.d. coin flip would trade every other day and lose on friction for
    reasons that have nothing to do with the idea. We instead flip a coin only on
    a random subset of days sized to the real rule's switch count, and carry the
    state forward in between.
    """
    s = pd.Series(signal).dropna()
    n = len(s)
    if n < 3:
        return pd.Series(index=pd.Series(signal).index, dtype=float, name="placebo_signal")
    n_switch = int(np.abs(s.diff().fillna(0.0) / 2.0).sum())
    rng = np.random.default_rng(seed)
    flip = np.zeros(n, dtype=bool)
    if n_switch > 0:
        flip[rng.choice(n, size=min(n_switch, n), replace=False)] = True
    out = np.empty(n)
    state = float(s.iloc[0])
    for i in range(n):
        if flip[i]:
            state = -state
        out[i] = state
    res = pd.Series(out, index=s.index, name="placebo_signal")
    return res.reindex(pd.Series(signal).index)


def switch_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    fast: str = "USFR",
    slow: str = "SHV",
    bench: str = "BIL",
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Long-only switch between the fast and slow repricer, excess of the bench.

    Parameters
    ----------
    prices:
        Daily total-return closes; must contain ``fast``, ``slow`` and ``bench``.
    signal:
        A {+1, -1} series, already lagged one day. +1 -> hold ``fast``, -1 -> ``slow``.
    cost_bps:
        One-way cost in bps charged on NAV per leg traded. A switch trades two
        legs (sell one vehicle, buy the other) so it costs ``2 x cost_bps``.
        PROXY/ASSUMPTION — swept in :func:`cost_sweep`. No shorting anywhere, so
        there is no borrow cost to charge.

    Returns a frame with ``r_fast``, ``r_slow``, ``r_bench``, ``signal``,
    ``r_switch`` (net of cost) and ``excess`` (= ``r_switch - r_bench``).
    """
    px = prices[[fast, slow, bench]].dropna().sort_index()
    rets = px.pct_change()
    sig = pd.Series(signal).reindex(px.index).astype(float)
    w_fast = (sig > 0).astype(float)
    turnover = w_fast.diff().abs().fillna(0.0) * 2.0  # two legs traded per switch
    cost = turnover * cost_bps * 1e-4
    r_switch = (w_fast * rets[fast] + (1.0 - w_fast) * rets[slow] - cost).rename("r_switch")
    out = pd.concat(
        [rets[fast].rename("r_fast"), rets[slow].rename("r_slow"),
         rets[bench].rename("r_bench"), sig.rename("signal"), r_switch],
        axis=1,
    )
    out["excess"] = out["r_switch"] - out["r_bench"]
    return out.loc[sig.dropna().index].dropna()


def summary(excess: pd.Series) -> dict:
    """Headline stats for a daily excess-of-cash series.

    ``ann_bp`` is the annualised mean excess in basis points — the natural unit for
    a cash study, where the whole dispersion is tens of bp. ``sharpe`` is the
    annualised Sharpe *of the excess series*; on cash-vs-cash differences its
    denominator is minuscule, so it is reported for completeness and never leaned
    on. ``tstat`` is the HAC t on the daily excess.
    """
    e = pd.Series(excess).astype(float).dropna()
    n = len(e)
    sd = e.std(ddof=1)
    return {
        "n_days": int(n),
        "ann_bp": float(e.mean() * TRADING_DAYS * 1e4),
        "tstat": newey_west_t(e.to_numpy()),
        "sharpe": float(e.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "vol_ann_bp": float(sd * np.sqrt(TRADING_DAYS) * 1e4),
        "hit_rate": float((e > 0).mean()),
    }


def evaluate(
    prices: pd.DataFrame,
    lookback: int = 21,
    fast: str = "USFR",
    slow: str = "SHV",
    bench: str = "BIL",
    rate_col: str = "^IRX",
    cost_bps: float = 2.0,
    placebo_seed: int = 923,
) -> dict:
    """Run the switch rule, its two placebos, and the static arms on one frame.

    Everything is excess-of-``bench``. Returns a dict of summaries plus the switch
    count and the fraction of days the rule is in the fast vehicle.
    """
    sig = rate_direction_signal(prices[rate_col], lookback=lookback)
    bt = switch_backtest(prices, sig, fast=fast, slow=slow, bench=bench, cost_bps=cost_bps)
    bt_rev = switch_backtest(prices, reversed_signal(sig), fast=fast, slow=slow,
                             bench=bench, cost_bps=cost_bps)
    bt_pl = switch_backtest(prices, placebo_signal(sig, seed=placebo_seed), fast=fast,
                            slow=slow, bench=bench, cost_bps=cost_bps)
    n_switch = int((bt["signal"].diff().abs().fillna(0.0) / 2.0).sum())
    return {
        "switch": summary(bt["excess"]),
        "reversed": summary(bt_rev["excess"]),
        "placebo": summary(bt_pl["excess"]),
        "static_fast": summary(bt["r_fast"] - bt["r_bench"]),
        "static_slow": summary(bt["r_slow"] - bt["r_bench"]),
        "n_switches": n_switch,
        "frac_fast": float((bt["signal"] > 0).mean()),
        "cost_bps": float(cost_bps),
        "lookback": int(lookback),
        "excess": bt["excess"],
        "bt": bt,
    }


def cost_sweep(prices: pd.DataFrame, cost_grid=(0.0, 1.0, 2.0, 5.0, 10.0), **kw) -> list[dict]:
    """Gross-to-net sweep of the one-way cost assumption (0 bps = the gross edge)."""
    rows = []
    for c in cost_grid:
        ev = evaluate(prices, cost_bps=c, **kw)
        rows.append({"cost_bps": float(c), "ann_bp": ev["switch"]["ann_bp"],
                     "tstat": ev["switch"]["tstat"], "n_switches": ev["n_switches"]})
    return rows


def lookback_sweep(prices: pd.DataFrame, lookbacks=(5, 21, 63, 126), cost_bps: float = 2.0, **kw) -> list[dict]:
    """Pre-registered signal-window grid, reported in full (gross and net).

    Reporting every window we looked at — rather than the best one — is the point:
    with four windows a nominal |t| ~ 2 on the winner would be unremarkable.
    """
    rows = []
    for L in lookbacks:
        gross = evaluate(prices, lookback=L, cost_bps=0.0, **kw)
        net = evaluate(prices, lookback=L, cost_bps=cost_bps, **kw)
        rows.append({
            "lookback": int(L),
            "gross_bp": gross["switch"]["ann_bp"], "gross_t": gross["switch"]["tstat"],
            "net_bp": net["switch"]["ann_bp"], "net_t": net["switch"]["tstat"],
            "reversed_net_bp": net["reversed"]["ann_bp"],
            "n_switches": net["n_switches"],
        })
    return rows


def era_cut(prices: pd.DataFrame, split: str = "2020-01-01", cost_bps: float = 2.0, **kw) -> dict:
    """Split the sample and re-run both halves of the study on each era.

    A real effect should show in both halves; the 2014-2019 era spans one gentle
    hiking cycle, the 2020-2026 era the zero floor, the 2022-23 spike and the
    2024-26 cuts.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = prices.loc[sl]
        if len(sub) < 300:
            out[tag] = None
            continue
        ev_g = evaluate(sub, cost_bps=0.0, **kw)
        ev_n = evaluate(sub, cost_bps=cost_bps, **kw)
        out[tag] = {
            "n_days": ev_n["switch"]["n_days"],
            "gross_bp": ev_g["switch"]["ann_bp"], "gross_t": ev_g["switch"]["tstat"],
            "net_bp": ev_n["switch"]["ann_bp"], "net_t": ev_n["switch"]["tstat"],
            "static_fast_bp": ev_n["static_fast"]["ann_bp"], "static_fast_t": ev_n["static_fast"]["tstat"],
            "static_slow_bp": ev_n["static_slow"]["ann_bp"], "static_slow_t": ev_n["static_slow"]["tstat"],
        }
    return out


def duration_table(prices: pd.DataFrame, vehicles, rate_col: str = "^IRX") -> pd.DataFrame:
    """Realised effective duration (years) + HAC t for each vehicle in one table."""
    rows = []
    for v in vehicles:
        d = effective_duration(prices[v], prices[rate_col])
        rows.append({"vehicle": v, **d})
    return pd.DataFrame(rows).set_index("vehicle")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, cost_bps: float = 2.0,
                     lookback: int = 21) -> dict:
    """Run both halves of the study on a synthetic panel and report what it found.

    On the planted world (``signal_strength=1``) the durations must order with the
    planted WAM, the centroid lag must be longest for the longest-WAM vehicle, and
    the switch rule must earn a positive gross excess (the rate path trends). On
    the null (``signal_strength=0``) durations collapse to ~0 and the switch rule
    must find nothing.
    """
    names = truth["names"]
    fastest, slowest = truth["fastest"], truth["slowest"]
    bench = names[len(names) // 2]
    durs = {v: effective_duration(prices[v], prices["^IRX"]) for v in names}
    cents = {v: lag_profile(prices[v], prices["^IRX"])["centroid_lag"] for v in names}
    ev_g = evaluate(prices, lookback=lookback, fast=fastest, slow=slowest,
                    bench=bench, cost_bps=0.0)
    ev_n = evaluate(prices, lookback=lookback, fast=fastest, slow=slowest,
                    bench=bench, cost_bps=cost_bps)
    return {
        "durations": {v: durs[v]["duration_years"] for v in names},
        "duration_t": {v: durs[v]["tstat"] for v in names},
        "centroids": cents,
        "duration_spread": float(durs[slowest]["duration_years"] - durs[fastest]["duration_years"]),
        "gross_bp": ev_g["switch"]["ann_bp"], "gross_t": ev_g["switch"]["tstat"],
        "net_bp": ev_n["switch"]["ann_bp"], "net_t": ev_n["switch"]["tstat"],
        "fast": fastest, "slow": slowest, "bench": bench,
        "n_switches": ev_n["n_switches"],
    }
