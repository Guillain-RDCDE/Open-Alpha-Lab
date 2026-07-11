"""Strategy + inference for Study 669 — RSI-Divergence.

The claim: **bullish RSI divergence marks reversals** — price makes a *lower* swing low while
RSI(14) makes a *higher* swing low at the same point. The textbook story: falling momentum
readings on a falling price ("bearish momentum fading") means the sellers are running out of
gas, and price should turn up shortly after the pattern confirms.

Pipeline, deterministic and honestly lagged:

1. **Swing lows.** A bar is a candidate swing low if its Close is the minimum of a centred
   ``2*order+1`` window. This candidate can only be *confirmed* ``order`` bars later — you
   need to see the following ``order`` bars to know nothing undercut it, exactly the way a
   chartist actually draws the pattern. Confirmation date = swing bar + ``order`` trading days.
2. **Divergence.** Compare each confirmed swing low to the *previous* confirmed swing low: if
   price is lower (a lower low) **and** RSI(14) is higher (a higher low) at the same two points,
   that's a bullish divergence, flagged on the confirmation date.
3. **One documented execution lag.** Enter at the **next trading day's open** after the
   confirmation date (zero look-ahead: everything used to confirm the pattern is public by that
   day's close). Hold for a fixed horizon (5 / 10 / 20 trading days), exit at that day's close.
4. **Three comparisons.** Divergence-conditional forward returns vs (a) the **unconditional**
   forward-return distribution of the same instrument (every day is a candidate entry, same
   formula) and (b) a **random-signal control** — the same *number* of signals per ticker, drawn
   from random eligible dates, repeated over many seeds. A pooled Newey-West (HAC) dummy
   regression cross-checks the Welch split for the serial correlation baked into overlapping
   forward-return windows.
5. **A timer with costs.** Turn the signal into a trade: one round trip = 2 x one-way cost x
   NAV per signal, long-only, no borrow (this is a *long* pattern).

The decisive number is the divergence-day Welch/HAC *t* on the REAL tape, against BOTH the
unconditional baseline and the random-signal placebo — plus whether costs leave anything on the
table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (5, 10, 20)


# --------------------------------------------------------------------------- #
# Indicator + swing-low + divergence detection
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (Wilder-smoothed average gain/loss, standard 14-period definition)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0.0, 100.0)


def swing_lows(close: pd.Series, order: int = 5) -> pd.Series:
    """Boolean flag: bar is the strict minimum of its centred ``2*order+1``-bar window.

    Candidates only — NOT yet confirmable in real time (needs ``order`` future bars). Ties
    inside a flat window are not flagged twice thanks to the centred rolling-min comparison.
    """
    roll_min = close.rolling(window=2 * order + 1, center=True).min()
    return (close == roll_min) & roll_min.notna()


def detect_bullish_divergence(df: pd.DataFrame, order: int = 5,
                               rsi_period: int = 14) -> pd.DataFrame:
    """Confirmed bullish price/RSI divergences: lower price low + higher RSI low.

    One row per event: ``swing_date`` (the low itself), ``confirm_date`` (swing_date +
    ``order`` trading days — the earliest date the pattern is actually knowable), the two
    swing dates' price and RSI levels.
    """
    close = df["Close"]
    r = rsi(close, rsi_period)
    sl = swing_lows(close, order)
    swing_idx = close.index[sl]
    pos_of = {d: i for i, d in enumerate(close.index)}
    n = len(close.index)

    rows = []
    for prev, curr in zip(swing_idx[:-1], swing_idx[1:]):
        p_prev, p_curr = float(close[prev]), float(close[curr])
        r_prev, r_curr = float(r[prev]), float(r[curr])
        if np.isnan(r_prev) or np.isnan(r_curr):
            continue
        if p_curr < p_prev and r_curr > r_prev:
            conf_pos = pos_of[curr] + order
            if conf_pos < n:
                rows.append({
                    "prev_swing_date": prev, "swing_date": curr,
                    "confirm_date": close.index[conf_pos],
                    "price_low_prev": p_prev, "price_low": p_curr,
                    "rsi_low_prev": r_prev, "rsi_low": r_curr,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Forward returns — one documented execution lag: next-day open -> h-day close
# --------------------------------------------------------------------------- #
def forward_return_series(df: pd.DataFrame, h: int, entry_lag: int = 1) -> pd.Series:
    """Log return from (t + entry_lag) OPEN to (t + entry_lag + h - 1) CLOSE, indexed by t.

    ``entry_lag=1`` is the study's single documented execution convention: a signal known at
    the close of day t is actionable at the next session's open — zero look-ahead. Every date
    in the index (not just signal dates) gets a value, so this doubles as the unconditional
    baseline generator: reindex it to any subset of dates to get that subset's forward-return
    distribution under the identical formula.
    """
    close, open_ = df["Close"], df["Open"]
    entry = open_.shift(-entry_lag)
    exit_ = close.shift(-(entry_lag + h - 1))
    return np.log(exit_ / entry)


# --------------------------------------------------------------------------- #
# Inference primitives
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


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d.

    b is exactly the treated-minus-rest mean difference; the NW t is the serial-correlation
    -robust cross-check for a forward-return series whose windows overlap.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~np.isnan(y) & ~np.isnan(d)
    y, d = y[keep], d[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


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
# Basket-wide event + forward-return assembly
# --------------------------------------------------------------------------- #
def basket_events(data: dict[str, pd.DataFrame], order: int = 5,
                   rsi_period: int = 14) -> pd.DataFrame:
    """Confirmed bullish-divergence events across every ticker in ``data``, tagged."""
    frames = []
    for tkr, df in data.items():
        ev = detect_bullish_divergence(df, order=order, rsi_period=rsi_period)
        if len(ev):
            ev = ev.copy()
            ev["ticker"] = tkr
        frames.append(ev)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def basket_forward_series(data: dict[str, pd.DataFrame], h: int) -> dict[str, pd.Series]:
    """{ticker: forward-return series at horizon h}, one entry per instrument."""
    return {tkr: forward_return_series(df, h) for tkr, df in data.items()}


def headline_stats(data: dict[str, pd.DataFrame], events: pd.DataFrame, h: int,
                    nw_lags: int | None = None) -> dict:
    """Divergence-conditional forward return vs the unconditional (same-instrument) baseline.

    Pools every ticker's signal-day and non-signal-day observations, Welch t on the split, and
    a Newey-West (HAC) cross-check on the pooled dummy regression (lag = h, the overlap window).
    """
    fwd = basket_forward_series(data, h)
    lags = h if nw_lags is None else nw_lags

    sig_vals, base_vals = [], []
    y_all, d_all = [], []
    for tkr, s in fwd.items():
        s = s.dropna()
        ev_dates = set(events.loc[events["ticker"] == tkr, "confirm_date"]) if len(events) else set()
        d = s.index.isin(ev_dates)
        sig_vals.append(s.values[d])
        base_vals.append(s.values[~d])
        y_all.append(s.values)
        d_all.append(d.astype(float))
    sig = np.concatenate(sig_vals) if sig_vals else np.array([])
    base = np.concatenate(base_vals) if base_vals else np.array([])
    y = np.concatenate(y_all) if y_all else np.array([])
    d = np.concatenate(d_all) if d_all else np.array([])

    k_up = int((sig > 0).sum())
    lo, hi = wilson_interval(k_up, len(sig))
    return {
        "h": h, "n_sig": len(sig), "n_base": len(base),
        "sig_mean_bps": float(np.nanmean(sig)) * 1e4 if len(sig) else float("nan"),
        "base_mean_bps": float(np.nanmean(base)) * 1e4 if len(base) else float("nan"),
        "gap_bps": (float(np.nanmean(sig)) - float(np.nanmean(base))) * 1e4
                    if len(sig) and len(base) else float("nan"),
        "welch_t": welch_t(sig, base),
        "nw_t": newey_west_t(y, d, lags=lags),
        "hit_up": k_up, "hit_rate": k_up / len(sig) if len(sig) else float("nan"),
        "hit_lo": lo, "hit_hi": hi,
    }


def random_signal_placebo(data: dict[str, pd.DataFrame], events: pd.DataFrame, h: int,
                           n_draws_per_seed: int = 200, n_seeds: int = 20,
                           base_seed: int = 669) -> dict:
    """Random-signal control: per ticker, draw as many random eligible dates as real signals.

    Repeated over ``n_seeds`` independent seeds x ``n_draws_per_seed`` draws each, so no single
    lucky stream decides it. p = share of draws whose pooled mean forward return is >= the
    observed divergence-signal mean (a RIGHT-tail test — the claim predicts a positive bounce).
    """
    fwd = basket_forward_series(data, h)
    per_ticker = {}
    for tkr, s in fwd.items():
        s = s.dropna()
        n_sig = int((events["ticker"] == tkr).sum()) if len(events) else 0
        per_ticker[tkr] = (s.values, n_sig)

    obs = headline_stats(data, events, h)["sig_mean_bps"] / 1e4

    means = []
    for seed_i in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed_i)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for tkr, (vals, n_sig) in per_ticker.items():
                if n_sig == 0 or len(vals) == 0:
                    continue
                k = min(n_sig, len(vals))
                idx = rng.choice(len(vals), size=k, replace=False)
                draw_vals.append(vals[idx])
            if draw_vals:
                means.append(np.concatenate(draw_vals).mean())
    means = np.asarray(means)
    return {
        "obs": obs, "placebo_mean": float(means.mean()), "placebo_sd": float(means.std(ddof=1)),
        "p_value": float((means >= obs).mean()), "n_draws": len(means), "draws": means,
    }


# --------------------------------------------------------------------------- #
# Sub-period contrast (justified split: pre- vs post- 2018 zero-commission/retail-flow era)
# --------------------------------------------------------------------------- #
def era_contrast(data: dict[str, pd.DataFrame], events: pd.DataFrame, h: int,
                  split: str) -> dict:
    """Divergence-signal forward return before vs since ``split``, and the Welch t OF THE
    DIFFERENCE between the two eras."""
    fwd = basket_forward_series(data, h)
    early_vals, late_vals = [], []
    for tkr, s in fwd.items():
        s = s.dropna()
        ev_dates = set(events.loc[events["ticker"] == tkr, "confirm_date"]) if len(events) else set()
        sig = s[s.index.isin(ev_dates)]
        early_vals.append(sig[sig.index < split].values)
        late_vals.append(sig[sig.index >= split].values)
    early = np.concatenate(early_vals) if early_vals else np.array([])
    late = np.concatenate(late_vals) if late_vals else np.array([])
    return {
        "n_early": len(early), "n_late": len(late),
        "early_bps": float(np.nanmean(early)) * 1e4 if len(early) else float("nan"),
        "late_bps": float(np.nanmean(late)) * 1e4 if len(late) else float("nan"),
        "welch_t_diff": welch_t(late, early),
    }


# --------------------------------------------------------------------------- #
# Third axis — the timer with costs
# --------------------------------------------------------------------------- #
def timer_with_costs(data: dict[str, pd.DataFrame], events: pd.DataFrame, h: int,
                      cost_bps: float = 5.0) -> dict:
    """Trade every confirmed divergence: enter next open, hold h days, exit at close.

    One round trip = 2 x ``cost_bps`` one-way x NAV per signal, long-only (this is a bullish
    pattern), no borrow. Gross/net per trade, hit rate with Wilson CI, worst single trade.
    """
    fwd = basket_forward_series(data, h)
    vals = []
    for tkr, s in fwd.items():
        s = s.dropna()
        ev_dates = set(events.loc[events["ticker"] == tkr, "confirm_date"]) if len(events) else set()
        vals.append(s[s.index.isin(ev_dates)].values)
    a = np.concatenate(vals) if vals else np.array([])
    gross = float(np.nanmean(a)) if len(a) else float("nan")
    net = gross - 2.0 * cost_bps / 1e4
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n": len(a), "gross_bps": gross * 1e4, "net_bps": net * 1e4,
        "hit_rate": k_up / len(a) if len(a) else float("nan"),
        "hit_lo": lo, "hit_hi": hi,
        "worst_bps": float(np.nanmin(a)) * 1e4 if len(a) else float("nan"),
        "best_bps": float(np.nanmax(a)) * 1e4 if len(a) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, h: int = 10, order: int = 5,
                      rsi_period: int = 14) -> dict:
    """Run the headline Welch/NW split on one synthetic instrument."""
    ev = detect_bullish_divergence(df, order=order, rsi_period=rsi_period)
    s = forward_return_series(df, h).dropna()
    ev_dates = set(ev["confirm_date"]) if len(ev) else set()
    d = s.index.isin(ev_dates)
    sig, base = s.values[d], s.values[~d]
    return {
        "n_sig": len(sig), "n_base": len(base),
        "sig_mean_bps": float(np.nanmean(sig)) * 1e4 if len(sig) else float("nan"),
        "welch_t": welch_t(sig, base),
        "nw_t": newey_west_t(s.values, d.astype(float), lags=h),
    }
