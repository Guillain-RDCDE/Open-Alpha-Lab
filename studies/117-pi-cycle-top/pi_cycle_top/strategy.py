"""The strategy and its honest controls — Study 117 (Pi-Cycle-Top).

The folk recipe: when the 111-day MA of BTC-USD daily close crosses *above* the 2× 350-day
MA, it allegedly signals the Bitcoin cycle top within a few days. Sell / short. The
study pins this against two fair controls:

- **Random-MA-cross control**: pairs of random MA windows of the same widths but with
  different multipliers, so the 'π ≈ 111/35.35...' numerology is eliminated. Do other
  MA cross dates do equally well, worse, or better?
- **Random-date control**: identical-length forward windows starting on random days
  (same calendar, same number of windows). Tests whether the Pi-Cycle dates specifically
  predict outsized drawdowns — or whether any window of BTC has a fat left tail.

The honest test: after each Pi-Cycle signal, how far does BTC fall in the next 30/60/90
days? Compare to random-start windows of the same length. With n≤3 events and BTC's
massive daily vol (~4%), the study's power floor is brutally low — that is the point.

No look-ahead: the crossover is confirmed on closes up to *t*; the trade enters at
*t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator
# ---------------------------------------------------------------------------
def ma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average, NaN until ``n`` closes are available."""
    return close.rolling(n, min_periods=n).mean()


def pi_cycle_signals(
    bars: pd.DataFrame,
    fast_n: int = 111,
    slow_n: int = 350,
    slow_mult: float = 2.0,
    col: str = "close",
) -> pd.DataFrame:
    """Dates where the fast MA crosses *above* the (slow_mult × slow MA).

    The Pi-Cycle-Top indicator fires when MA(111) > 2 × MA(350): the idea being that
    the ratio 111 / 350 ≈ π/10, so the crossover is "encoded in pi". The signal is
    formed on the *close* of the crossover bar — the trade enters the next bar's open.

    Returns a DataFrame with columns ``signal_date, fast_ma, slow_ma, slow_ma_2x,
    btc_price`` indexed by position.
    """
    close = bars[col]
    f = ma(close, fast_n)
    s = ma(close, slow_n)
    spread = f - slow_mult * s

    # Detect up-crossings: spread goes from ≤ 0 to > 0
    sign = np.sign(spread)
    crossed_above = (sign > 0) & (sign.shift(1) <= 0) & sign.shift(1).notna()

    rows = []
    for dt in bars.index[crossed_above]:
        rows.append(
            {
                "signal_date": dt,
                "fast_ma": float(f.loc[dt]),
                "slow_ma": float(s.loc[dt]),
                "slow_ma_2x": float(slow_mult * s.loc[dt]),
                "btc_price": float(close.loc[dt]),
            }
        )
    return pd.DataFrame(rows)


def days_to_actual_top(
    bars: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    look_ahead_days: int = 90,
    col: str = "close",
) -> pd.DataFrame:
    """For each signal date, find the local price maximum in the next ``look_ahead_days``
    calendar days and compute the distance (in calendar days) to that maximum.

    Also computes the forward returns at 30, 60, and 90 days from the signal.
    Columns: ``signal_date, btc_at_signal, peak_date, peak_price, days_to_peak,
              ret_30d, ret_60d, ret_90d``.
    """
    close = bars[col]
    rows = []
    for sig_dt in signal_dates:
        # Signals fire on *t*; the trade enters at *t+1*'s open — but for the
        # 'distance to actual top' narrative we use *t* itself (close of signal bar).
        mask = (bars.index > sig_dt) & (
            bars.index <= sig_dt + pd.Timedelta(days=look_ahead_days)
        )
        forward = close[mask]
        if forward.empty:
            continue
        peak_dt = forward.idxmax()
        btc_sig = float(close.asof(sig_dt)) if hasattr(close, "asof") else float(close.get(sig_dt, np.nan))

        def _fret(days: int) -> float:
            target = sig_dt + pd.Timedelta(days=days)
            # First bar on or after target date
            avail = bars.index[bars.index >= target]
            if avail.empty:
                return float("nan")
            return float(np.log(close.loc[avail[0]] / btc_sig)) if btc_sig > 0 else float("nan")

        rows.append(
            {
                "signal_date": sig_dt,
                "btc_at_signal": btc_sig,
                "peak_date": peak_dt,
                "peak_price": float(forward.max()),
                "days_to_peak": int((peak_dt - sig_dt).days),
                "ret_30d": _fret(30),
                "ret_60d": _fret(60),
                "ret_90d": _fret(90),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forward-return engine — signal vs controls
# ---------------------------------------------------------------------------
def forward_returns(
    bars: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    horizon_days: int = 90,
    col: str = "close",
) -> pd.DataFrame:
    """Log forward-return from the close of each signal date over ``horizon_days``
    calendar days.

    Entry is at the *next* available trading-day open after each signal date (no
    look-ahead). Exit is the close ``horizon_days`` calendar days later (first bar
    on or after the target date).

    Columns: ``signal_date, entry_date, entry_px, exit_date, exit_px, log_ret, pct_ret``.
    """
    close = bars[col]
    rows = []
    for sig_dt in signal_dates:
        after = bars.index[bars.index > sig_dt]
        if len(after) == 0:
            continue
        entry_date = after[0]
        entry_px = float(bars["open"].loc[entry_date])
        # Exit: first bar on or after signal_date + horizon_days
        target_exit = sig_dt + pd.Timedelta(days=horizon_days)
        exit_candidates = bars.index[bars.index >= target_exit]
        if exit_candidates.empty:
            continue
        exit_date = exit_candidates[0]
        exit_px = float(close.loc[exit_date])
        log_r = np.log(exit_px / entry_px)
        rows.append(
            {
                "signal_date": sig_dt,
                "entry_date": entry_date,
                "entry_px": entry_px,
                "exit_date": exit_date,
                "exit_px": exit_px,
                "log_ret": log_r,
                "pct_ret": float(np.expm1(log_r)),
            }
        )
    return pd.DataFrame(rows)


def random_control_returns(
    bars: pd.DataFrame,
    n_events: int,
    horizon_days: int = 90,
    seed: int = 117,
    n_draws: int = 500,
    col: str = "close",
) -> pd.Series:
    """Monte-Carlo control: draw ``n_draws`` sets of ``n_events`` random starting
    dates from the tape (each at least ``horizon_days`` before the tape end) and
    compute the mean forward log-return for each set.

    This answers: *"how often does a random equally-sized bundle of BTC windows
    produce returns as bad as the Pi-Cycle signal windows?"*  If the indicator has
    no power, its windows should look like random windows of BTC.

    Returns a Series of length ``n_draws`` (mean log-ret per draw).
    """
    rng = np.random.default_rng(seed)
    close = bars[col]
    # Eligible positions: must have a full horizon ahead
    eligible = np.array(
        [i for i in range(len(bars)) if
         (bars.index[-1] - bars.index[i]).days >= horizon_days + 5]
    )
    if len(eligible) < n_events:
        return pd.Series(dtype=float)

    means = []
    for _ in range(n_draws):
        idxs = rng.choice(eligible, size=n_events, replace=False)
        rets = []
        for i in idxs:
            entry_px = float(bars["open"].iloc[i + 1]) if i + 1 < len(bars) else float("nan")
            target = bars.index[i] + pd.Timedelta(days=horizon_days)
            exits = bars.index[bars.index >= target]
            if exits.empty or not np.isfinite(entry_px) or entry_px <= 0:
                continue
            exit_px = float(close.loc[exits[0]])
            rets.append(np.log(exit_px / entry_px))
        if rets:
            means.append(float(np.mean(rets)))
    return pd.Series(means, name="random_mean_log_ret")


def alternative_ma_signals(
    bars: pd.DataFrame,
    fast_n: int = 111,
    slow_n: int = 350,
    mult_variants: list[float] | None = None,
    col: str = "close",
) -> dict[float, pd.DataFrame]:
    """Sweep the slow_mult parameter around 2.0 to show the indicator is sensitive
    to the chosen multiplier — a hallmark of curve-fitting.

    Returns a dict mapping each multiplier to its signal DataFrame.
    """
    if mult_variants is None:
        mult_variants = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
    return {m: pi_cycle_signals(bars, fast_n=fast_n, slow_n=slow_n, slow_mult=m, col=col)
            for m in mult_variants}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(returns: np.ndarray | pd.Series, label: str = "") -> dict:
    """Headline statistics for an array of event forward log-returns.

    Returns n, mean, std, min, max, and a HAC t-stat (Newey-West). With n≤3 the
    t-stat is explicitly unreliable (and flagged via ``low_power_warning``); we compute
    it anyway so the inference bar is honestly stated.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "label": label,
        "n": int(n),
        "mean": float(r.mean()) if n else float("nan"),
        "std": float(r.std(ddof=1)) if n > 1 else float("nan"),
        "min": float(r.min()) if n else float("nan"),
        "max": float(r.max()) if n else float("nan"),
        "tstat": float("nan"),
        "low_power_warning": n < 10,
    }
    if n > 1:
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, min(lags + 1, n)):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def power_analysis(
    bars: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    horizon_days: int = 90,
    seed: int = 117,
    n_draws: int = 2000,
) -> dict:
    """Bootstrap power: empirical p-value of the Pi-Cycle forward mean vs the
    random-day null (one-sided: fraction of random bundles *worse* than the signal).

    Returns p_value, observed mean, random control mean/p90/p95.
    """
    events = forward_returns(bars, signal_dates, horizon_days=horizon_days)
    if events.empty:
        return {"p_value": float("nan"), "n_events": 0}
    obs_mean = float(events["log_ret"].mean())
    n_ev = len(events)
    ctrl = random_control_returns(
        bars, n_events=n_ev, horizon_days=horizon_days, seed=seed, n_draws=n_draws
    )
    # One-sided: we claim the signal predicts *negative* returns (sell signal)
    p_value = float((ctrl <= obs_mean).mean())
    return {
        "obs_mean_log_ret": obs_mean,
        "n_events": int(n_ev),
        "p_value": float(p_value),
        "ctrl_p10": float(ctrl.quantile(0.10)),
        "ctrl_p5": float(ctrl.quantile(0.05)),
        "ctrl_mean": float(ctrl.mean()),
        "ctrl_std": float(ctrl.std()),
    }
