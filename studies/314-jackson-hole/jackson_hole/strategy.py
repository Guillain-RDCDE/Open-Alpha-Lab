"""The Jackson-Hole event study and its honest inference.

The folklore: the S&P "drifts" around the Fed's late-August Jackson Hole symposium —
either anticipating a dovish keynote in the days before, or reacting to it on/after the
speech. We make that falsifiable with a classic **event study**: tag the trading
session of (and around) each Friday keynote, measure the abnormal return versus all
other days, and ask whether one observation per year can clear the inference bar.

Honest machinery:
  * **event_window** tags sessions at a fixed offset from each speech day (offset 0 =
    the speech day; −1 = the day before; +1 = the day after). The rule is calendar-known
    — the symposium date is announced months ahead — so a long-before / day-of position
    needs **no execution lag**. A *react-to-the-speech* trade (enter at the speech-day
    close, hold the next session) is the one place a lag matters, and ``react_trade``
    applies exactly one ``shift``, documented.
  * **event_table** compares event days to all other days with a Welch t on the
    difference in means, and reports a **HAC (Newey-West) t** on the event-day mean —
    the inference-bar statistic.
  * **block_bootstrap_ci** gives a circular-block-bootstrap CI on the event-day mean, so
    the thin, autocorrelation-respecting sample is not over-trusted.
  * **react_trade** is the only *traded* arm: long SPY from the speech-day close to the
    next close, costed one-way × NAV (charged twice for the round trip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


# ---------------------------------------------------------------------------
# Event tagging
# ---------------------------------------------------------------------------
def event_window(
    index: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
    offset: int = 0,
) -> pd.Series:
    """Boolean over ``index``, True on the trading session ``offset`` sessions from each
    event date (0 = the session at/just before the event date; +1 = the next session;
    −1 = the prior session). Robust to event dates on a non-trading day (snaps to the
    first session on/after the date, then applies ``offset``)."""
    idx = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=idx)
    for d in pd.DatetimeIndex(event_dates):
        pos = idx.searchsorted(d)            # first session >= d
        # if d is itself a session, searchsorted lands on it; clamp into range
        if pos >= len(idx):
            continue
        j = pos + offset
        if 0 <= j < len(idx):
            mask.iloc[j] = True
    return mask


def _hac_t(x: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``x`` (lag rule ⌊4(n/100)^(2/9)⌋)."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def event_table(
    ret: pd.Series,
    event_dates: pd.DatetimeIndex,
    offset: int = 0,
) -> dict:
    """Compare event days (at ``offset``) to all other days.

    Returns the event-day mean and the rest-of-sample mean, the Welch t on the
    *difference*, the **HAC t on the event-day mean** (the inference-bar number), the
    event-day win-rate, and the count of events. With one event per year, ``n_event`` is
    tiny — read every number against that.
    """
    ret = pd.Series(ret).astype(float).dropna()
    mask = event_window(ret.index, event_dates, offset).reindex(ret.index).fillna(False)
    ev, rest = ret[mask], ret[~mask]
    n_ev = int(mask.sum())
    if n_ev < 2 or len(rest) < 2:
        return {
            "offset": offset, "event_mean": float("nan"), "rest_mean": float("nan"),
            "diff_bps": float("nan"), "welch_t": float("nan"), "hac_t": float("nan"),
            "win_rate": float("nan"), "n_event": n_ev, "n_total": int(len(ret)),
        }
    se, sr = ev.std(ddof=1), rest.std(ddof=1)
    denom = np.sqrt(se**2 / len(ev) + sr**2 / len(rest))
    welch = (ev.mean() - rest.mean()) / denom if denom > 0 else float("nan")
    return {
        "offset": offset,
        "event_mean": float(ev.mean()),
        "rest_mean": float(rest.mean()),
        "diff_bps": float((ev.mean() - rest.mean()) * 1e4),
        "welch_t": float(welch),
        "hac_t": _hac_t(ev.to_numpy()),
        "win_rate": float((ev > 0).mean()),
        "n_event": n_ev,
        "n_total": int(len(ret)),
    }


def offset_sweep(
    ret: pd.Series,
    event_dates: pd.DatetimeIndex,
    offsets=range(-5, 6),
) -> pd.DataFrame:
    """An event-table row for each offset in ``offsets`` — the whole drift profile
    around the keynote (−5 … +5 sessions)."""
    return pd.DataFrame([event_table(ret, event_dates, o) for o in offsets])


# ---------------------------------------------------------------------------
# Robust CI on the event-day mean
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    ret: pd.Series,
    event_dates: pd.DatetimeIndex,
    offset: int = 0,
    n_boot: int = 5000,
    block: int = 3,
    seed: int = 314,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular-block-bootstrap CI for the event-day mean return (at ``offset``).

    We resample the *event-day returns* in blocks of ``block`` (so any clustering across
    consecutive years is respected) and take percentile bounds. With ~31 events the CI
    is wide by construction — that width is the point.
    """
    ret = pd.Series(ret).astype(float).dropna()
    mask = event_window(ret.index, event_dates, offset).reindex(ret.index).fillna(False)
    ev = ret[mask].to_numpy()
    n = ev.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = ev[idx[:n]].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# The only traded arm — react to the speech
# ---------------------------------------------------------------------------
def react_trade(
    ret: pd.Series,
    event_dates: pd.DatetimeIndex,
    hold: int = 1,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Long SPY from the **speech-day close** for ``hold`` sessions, one trade per year.

    Execution lag: the speech is a calendar-known Friday-morning event, but a
    *reaction* trade can only act on what it sees — so we enter at the **speech-day
    close** (one ``shift`` past the morning speech) and earn the following ``hold``
    sessions' returns. ``cost_bps`` is a one-way fee charged on entry *and* exit
    (round-trip = 2 × cost_bps), deducted from the compounded gross.

    Columns: ``year, entry_date, ret_gross, ret_net``.
    """
    ret = pd.Series(ret).astype(float).dropna()
    idx = ret.index
    rows = []
    for d in pd.DatetimeIndex(event_dates):
        pos = idx.searchsorted(d)
        if pos >= len(idx):
            continue
        # the speech-day session is `pos` (or the prior one if d is non-trading)
        speech_pos = pos if idx[pos] == d else pos - 1
        first = speech_pos + 1               # one shift: earn from the NEXT session
        last = first + hold
        if first < 0 or last > len(idx):
            continue
        window = ret.iloc[first:last]
        if window.empty:
            continue
        g = float((1.0 + window).prod() - 1.0)
        net = (1.0 + g) * (1.0 - cost_bps * 1e-4) ** 2 - 1.0
        rows.append({
            "year": int(d.year),
            "entry_date": idx[speech_pos],
            "ret_gross": g,
            "ret_net": net,
        })
    return pd.DataFrame(rows)


def summarize_trades(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for the reaction ledger: n, win-rate, mean (bps), per-trade Sharpe,
    and a (small-sample) HAC t on the mean."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"),
                "sharpe": float("nan"), "hac_t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "hac_t": _hac_t(r),
    }


def summary(daily: pd.Series) -> dict:
    """Annualised mean/vol/Sharpe of a daily-return series."""
    r = pd.Series(daily).astype(float).dropna()
    if len(r) < 2:
        return {k: float("nan") for k in ("mean", "vol", "sharpe", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    return {"mean": float(mean), "vol": float(std),
            "sharpe": float(mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan"),
            "n": int(len(r))}
