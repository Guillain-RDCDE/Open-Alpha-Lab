"""Strategy + inference for Study 652 — Index-Deletion-Bounce.

The claim (Chen, Noronha & Singal 2004, *The price response to S&P 500 index additions and
deletions*, JF): stocks **deleted** from the S&P 500 get dumped by index funds into the
effective date — forced, price-insensitive selling — and then **rebound**, because unlike an
*addition* (where informed active managers happily supply the shares an index fund needs to
buy, arbitraging the pop away) a *deletion* asks funds to sell into a market with few natural
buyers at that exact moment; once the forced flow ends, the price should mean-revert back
toward fundamental value. CNS's original finding was that, unlike the inclusion effect, the
**deletion effect had NOT decayed** as of their sample.

Measurements, all against a **market-adjusted abnormal return** (stock log return minus SPY
log return on the same day — the return CNS themselves report):

* **Event-window CAR by offset**, offsets [-5..+40] around the effective date (offset 0). Mean
  cumulative abnormal return (CAR) at each offset across events, one-sample *t* (mean / SE)
  and a percentile bootstrap CI (event-level resampling — the natural resampling unit here,
  since events are cross-sectional, not one time series).
* **The dump** — CAR over [-5..0]: is the pre-effective drop real?
* **The rebound** — CAR over [0..+40] (and its own sub-legs): is there a reversal after the
  forced selling ends?
* **Random-day placebo** — for each ticker with real tape, draw a random non-event trading day
  from ITS OWN history and run the identical CAR machinery; repeat over many seeds. Tests
  whether the rebound is special to the deletion event or just generic mean-reversion in
  distressed small-caps.
* **Era split** — first half vs second half of the sample (justified: literally cuts the
  17-year span in two), the CNS "no decay" claim's within-sample analogue.
* **Third axis (tradability)** — the long-the-deleted timer: enter at the close of the
  effective date (the date is public days ahead — zero look-ahead), hold ``N`` trading days,
  exit; gross/net of one-way costs x 2 per leg, excess vs SPY over the same holding window.

The decisive number is the post-effective CAR Welch/one-sample *t* on the REAL tape; the
honest question is whether a distressed, thin, soon-forgotten name can actually be bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared shape with the rest of the desk)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0. NaN if < 2 obs."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 652,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling EVENTS with replacement."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Event window on the real tape
# --------------------------------------------------------------------------- #
def _log_ret(px: pd.Series) -> pd.Series:
    return np.log(px).diff()


def event_ar(stock: pd.DataFrame, spy: pd.DataFrame, effective: pd.Timestamp,
             lo: int = -5, hi: int = 40) -> pd.Series | None:
    """Per-offset abnormal (market-adjusted) daily log return around ``effective``.

    Offset 0 is the trading day on/after ``effective`` present on the STOCK's own tape (the
    effective date itself, or the next session the stock actually traded). Returns a Series
    indexed by integer offset [lo..hi], or None if the stock's tape doesn't cover the full
    window (handled honestly — the event is dropped, not silently zero-filled).
    """
    idx = stock.index
    pos = idx.searchsorted(effective)
    if pos >= len(idx) or idx[pos] > effective + pd.Timedelta(days=7):
        return None
    lo_pos, hi_pos = pos + lo, pos + hi
    if lo_pos < 1 or hi_pos >= len(idx):
        return None
    window_idx = idx[lo_pos - 1: hi_pos + 1]  # one extra day at the front for the diff
    s_ret = _log_ret(stock.loc[window_idx, "Close"])
    m = spy.reindex(window_idx)["Close"]
    if m.isna().sum() > 2:
        return None
    m_ret = _log_ret(m)
    ar = (s_ret - m_ret).iloc[1:]
    ar.index = range(lo, hi + 1)
    if ar.isna().sum() > 3:
        return None
    return ar.fillna(0.0)


def build_event_panel(tapes: dict[str, pd.DataFrame], spy: pd.DataFrame,
                       events: pd.DataFrame, lo: int = -5, hi: int = 40
                       ) -> tuple[pd.DataFrame, list[str]]:
    """AR panel: rows = events with usable tape, columns = offsets [lo..hi]. Also the drop list."""
    rows, tickers, dropped = [], [], []
    for _, row in events.iterrows():
        t = row["ticker"]
        if t not in tapes:
            dropped.append(t)
            continue
        ar = event_ar(tapes[t], spy, row["effective"], lo, hi)
        if ar is None:
            dropped.append(t)
            continue
        rows.append(ar)
        tickers.append(t)
    panel = pd.DataFrame(rows, index=tickers)
    return panel, dropped


def car_by_offset(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean CAR (cumulative AR from the panel's first column) by offset, with one-sample t."""
    car = panel.cumsum(axis=1)
    out = pd.DataFrame({
        "mean_car": car.mean(axis=0),
        "t": car.apply(lambda col: one_sample_t(col.values), axis=0),
        "n": car.notna().sum(axis=0),
    })
    return out


def window_car(panel: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Per-event CAR summed over offsets [lo..hi] (inclusive), as a plain array."""
    cols = [c for c in panel.columns if lo <= c <= hi]
    return panel[cols].sum(axis=1).values


def announce_to_effective_car(tapes: dict[str, pd.DataFrame], spy: pd.DataFrame,
                               events: pd.DataFrame) -> dict:
    """Market-adjusted CAR from the session before ANNOUNCE to the effective date, per event.

    This is the informed-selling leg: the S&P press release is public days before the forced
    index-fund flow at the effective-date close, so a decline over this window would be the
    market front-running the coming removal (the same identity 249-index-inclusion reports for
    the ADD side, mirrored here for the DELETE side).
    """
    vals = []
    for _, row in events.iterrows():
        t = row["ticker"]
        if t not in tapes:
            continue
        idx = tapes[t].index
        a_pos, e_pos = idx.searchsorted(row["announce"]), idx.searchsorted(row["effective"])
        if a_pos < 1 or e_pos >= len(idx) or e_pos <= a_pos:
            continue
        window = idx[a_pos - 1: e_pos + 1]
        s = _log_ret(tapes[t].loc[window, "Close"]).iloc[1:]
        m = _log_ret(spy.reindex(window)["Close"]).iloc[1:]
        vals.append(float((s - m).sum()))
    vals = np.asarray(vals)
    return {"n": len(vals), "mean_car": float(np.nanmean(vals)), "t": one_sample_t(vals)}


# --------------------------------------------------------------------------- #
# Random-day placebo — same tickers, random non-event anchor days
# --------------------------------------------------------------------------- #
def placebo_car(tapes: dict[str, pd.DataFrame], spy: pd.DataFrame,
                 events: pd.DataFrame, lo: int = -5, hi: int = 40,
                 n_draws: int = 500, base_seed: int = 652) -> dict:
    """For each real event's ticker, draw a random anchor day from ITS own tape (excluding a
    +-60 session buffer around the true effective date) and compute the same post-window CAR.
    Repeat ``n_draws`` times; report the observed mean post-CAR vs the placebo distribution.
    """
    real_tickers = [t for t in events["ticker"] if t in tapes]
    obs_panel, _ = build_event_panel(tapes, spy, events, lo, hi)
    obs_post = window_car(obs_panel, 1, hi)
    obs_mean = float(np.nanmean(obs_post))

    rng = np.random.default_rng(base_seed)
    placebo_means = []
    for _ in range(n_draws):
        vals = []
        for t in real_tickers:
            idx = tapes[t].index
            eff_row = events.loc[events["ticker"] == t, "effective"].iloc[0]
            eff_pos = idx.searchsorted(eff_row)
            candidates = [p for p in range(-lo + 1, len(idx) - hi - 1)
                          if abs(p - eff_pos) > (hi - lo)]
            if not candidates:
                continue
            p = candidates[rng.integers(0, len(candidates))]
            anchor = idx[p]
            ar = event_ar(tapes[t], spy, anchor, lo, hi)
            if ar is not None:
                vals.append(ar.loc[1:hi].sum())
        if vals:
            placebo_means.append(float(np.mean(vals)))
    placebo_means = np.asarray(placebo_means)
    p_value = float((placebo_means >= obs_mean).mean()) if len(placebo_means) else float("nan")
    return {"obs_mean": obs_mean, "placebo_mean": float(placebo_means.mean()),
            "placebo_sd": float(placebo_means.std(ddof=1)), "n_draws": len(placebo_means),
            "p_value": p_value}


# --------------------------------------------------------------------------- #
# Era contrast (within-sample "has it decayed?" check)
# --------------------------------------------------------------------------- #
def era_contrast(panel: pd.DataFrame, events: pd.DataFrame, split: str,
                  lo: int = 1, hi: int = 40) -> dict:
    """Post-effective CAR [lo..hi], first half of the sample vs second half."""
    tickers_in_panel = panel.index
    eff_by_ticker = events.set_index("ticker")["effective"]
    early_mask = eff_by_ticker.loc[tickers_in_panel] < pd.Timestamp(split)
    early = window_car(panel.loc[early_mask.values], lo, hi)
    late = window_car(panel.loc[~early_mask.values], lo, hi)
    return {"n_early": len(early), "n_late": len(late),
            "early_car": float(np.nanmean(early)), "late_car": float(np.nanmean(late)),
            "t_early": one_sample_t(early), "t_late": one_sample_t(late),
            "t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Third axis — the long-the-deleted timer, with costs
# --------------------------------------------------------------------------- #
def long_timer(panel: pd.DataFrame, hold_days: int = 40, cost_bps: float = 5.0) -> dict:
    """Long the deleted stock from the effective-date close, held ``hold_days``, net of costs.

    Enter at the close of offset 0 (the effective date — public days ahead, zero look-ahead),
    exit at the close of offset ``hold_days``. Return is already market-adjusted (excess vs
    SPY, i.e. "excess-vs-excess" is automatic since AR already nets out the benchmark). One
    round trip = 2 x one-way cost x NAV.
    """
    ret = window_car(panel, 1, hold_days)
    gross = float(np.nanmean(ret))
    net = gross - 2.0 * cost_bps / 1e4
    t_gross = one_sample_t(ret)
    net_arr = ret - 2.0 * cost_bps / 1e4
    return {"n": len(ret), "gross": gross, "net": net,
            "t_gross": t_gross, "t_net": one_sample_t(net_arr),
            "hit_rate": float((ret > 0).mean()), "worst": float(np.nanmin(ret)),
            "best": float(np.nanmax(ret))}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(events_syn: list[pd.DataFrame], lo: int = -5, hi: int = 40) -> dict:
    """Run the headline CAR machinery on a list of synthetic event paths."""
    panel = pd.DataFrame({i: ev.loc[lo:hi, "ar"] for i, ev in enumerate(events_syn)}).T
    panel.columns = list(range(lo, hi + 1))
    dump = window_car(panel, -5, 0)
    reb = window_car(panel, 1, hi)
    return {"t_dump": one_sample_t(dump), "t_rebound": one_sample_t(reb),
            "mean_dump": float(np.nanmean(dump)), "mean_rebound": float(np.nanmean(reb))}
