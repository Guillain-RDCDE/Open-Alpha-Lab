"""Strategy + inference for Study 442 — Anchored VWAP (the "price magnet").

The claim: anchor a volume-weighted average price to a significant event (here the **session
open**, the strongest clean intraday anchor) and the resulting **AVWAP** line becomes a price
magnet — price *respects* it, bouncing off it or reverting toward it. We test it as an intraday
event study on 5-minute bars:

    * **AVWAP** = the running, anchored volume-weighted average price within each session,
      computed from the session open (typical price = (H+L+C)/3, weighted by bar volume).
    * **Touch/cross event** = a bar whose close crosses the AVWAP line (the price moves from one
      side of the line to the other), the bars a believer would trade around.
    * **The magnet test** = after a touch, does price **revert** back through the level (bounce)
      or **continue**? We measure the signed reaction over the next ``H`` bars, oriented so that
      a genuine support/resistance "respect" of the level shows up as a positive **reversion**.
    * **The control** = the *same* test run at **randomly placed levels** drawn from the session
      price range. If AVWAP is special, the AVWAP reaction must beat the random-level reaction.

Inference, the desk's shared spirit:

  * a **one-sample / HAC (Newey-West) t** of the post-touch reaction against zero, and of the
    AVWAP reaction minus the random-control reaction (the level-specialness test);
  * a **label-shuffle / random-level placebo** — re-run at thousands of random levels and ask
    how often a random level reacts as strongly as AVWAP (the honest "is the line special?" null);
  * **one-bar execution lag** (act on the *next* bar's open after the touch is confirmed) and a
    **half-spread** cost charged on entry+exit (intraday levels usually die to the spread).

The decisive number is the AVWAP post-touch reaction net of the spread, and whether it clears
``t = 2`` *and* beats the random-level control. Most intraday level folklore lands NONE×MIRAGE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # post-touch reaction horizons, in 5-minute bars


# --------------------------------------------------------------------------- #
# Anchored VWAP
# --------------------------------------------------------------------------- #
def session_avwap(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-session anchored VWAP (anchored to the 09:30 open), bar-by-bar.

    Returns the bars frame with three columns added: ``avwap`` (the running anchored VWAP up to
    and including each bar), ``sess`` (the session date), and ``stretch`` (close/avwap − 1, the
    signed distance of the close from the line). VWAP uses the typical price (H+L+C)/3 weighted
    by bar volume, the standard intraday VWAP convention.
    """
    df = bars.copy()
    df["sess"] = df.index.tz_convert("America/New_York").date \
        if df.index.tz is not None else df.index.date
    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typ * df["volume"]
    g = df.groupby("sess")
    df["avwap"] = g.apply(
        lambda x: (typ.loc[x.index] * x["volume"]).cumsum() / x["volume"].cumsum()
    ).reset_index(level=0, drop=True)
    df["stretch"] = df["close"] / df["avwap"] - 1.0
    return df


# --------------------------------------------------------------------------- #
# Touch / cross detection + the post-touch reaction
# --------------------------------------------------------------------------- #
def _crossings(close: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Boolean array: True where close crosses ``level`` between t-1 and t (sign change)."""
    side = np.sign(close - level)
    prev = np.concatenate([[0.0], side[:-1]])
    cross = (side != 0) & (prev != 0) & (side != prev)
    return cross


def _session_reactions(close: np.ndarray, level: np.ndarray, side_before: np.ndarray,
                       horizon: int, lag: int) -> list[float]:
    """Signed reversion reactions for crossings of ``level`` within one session.

    For a cross at bar ``i`` (close moves from one side of the level to the other), we observe
    the cross at ``i`` close, **enter on the next bar's** (``i + lag``) close, and measure the
    return to ``i + lag + horizon``. The reaction is oriented as a **reversion** trade: if the
    price was *below* the level before the cross (it rose up through it), a "respect" of the
    level as resistance means it falls back — so a reversion bet is short, and we sign the raw
    forward return by ``-sign(move through the level)``. A positive mean reaction == the level is
    respected as support/resistance (bounce); negative == price tends to continue through it.
    """
    out = []
    n = len(close)
    cross = _crossings(close, level)
    for i in np.flatnonzero(cross):
        entry = i + lag
        exit_ = entry + horizon
        if entry >= n or exit_ >= n:
            continue
        # direction of the move through the level at the cross (up = +1, down = -1)
        move_dir = np.sign(close[i] - level[i])
        fwd = close[exit_] / close[entry] - 1.0
        # reversion: bet against the move through the level
        out.append(float(-move_dir * fwd))
    return out


def avwap_reactions(df_avwap: pd.DataFrame, horizon: int, lag: int = 1) -> np.ndarray:
    """Pool the signed post-touch reversion reactions for AVWAP crossings across all sessions.

    ``df_avwap`` is the output of :func:`session_avwap`. One reaction per AVWAP cross; reactions
    are oriented so a *positive* mean == the AVWAP level is respected (support/resistance magnet),
    a *negative* mean == price continues through it. The 1-bar lag is the one documented
    execution lag (act on the next bar after the touch is confirmed).
    """
    out: list[float] = []
    for _, x in df_avwap.groupby("sess"):
        close = x["close"].to_numpy(dtype=float)
        level = x["avwap"].to_numpy(dtype=float)
        out.extend(_session_reactions(close, level, None, horizon, lag))
    return np.asarray(out, dtype=float)


def random_level_reactions(df_avwap: pd.DataFrame, horizon: int, lag: int = 1,
                           n_levels: int = 20, seed: int = 442) -> np.ndarray:
    """Control: the same reversion test at **randomly placed horizontal levels**.

    For each session we draw ``n_levels`` flat levels uniformly inside that session's
    [low, high] range and pool the post-cross reactions, oriented identically to AVWAP. If AVWAP
    is a *special* magnet, its reaction must beat this random-level baseline. Deterministic given
    ``seed``.
    """
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _, x in df_avwap.groupby("sess"):
        close = x["close"].to_numpy(dtype=float)
        lo = float(x["low"].min())
        hi = float(x["high"].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        for lvl in rng.uniform(lo, hi, size=n_levels):
            level = np.full_like(close, lvl)
            out.extend(_session_reactions(close, level, None, horizon, lag))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def hac_t(sample: np.ndarray, lags: int = 5) -> float:
    """Newey-West (HAC) t of mean(``sample``) against 0.

    Events are ordered in tape time; nearby touches share microstructure, so a HAC correction
    on the mean is the autocorrelation-robust statistic the desk's inference bar requires.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = float(e[k:] @ e[:-k]) / n
        var += 2.0 * w * gk
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance) — AVWAP reaction vs random control."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


def placebo_pvalue(df_avwap: pd.DataFrame, horizon: int, lag: int = 1,
                   n_draws: int = 2000, seed: int = 442) -> dict:
    """Random-level placebo null for the AVWAP reaction.

    Draw ``n_draws`` random horizontal levels (one per session, repeated) and ask how often a
    random level's mean reaction is **at least as large** as AVWAP's. ``p`` = that share — the
    honest answer to "is the AVWAP line any more of a magnet than a line drawn at random?".
    """
    obs = float(np.nanmean(avwap_reactions(df_avwap, horizon, lag)))
    rng = np.random.default_rng(seed)
    sessions = [x for _, x in df_avwap.groupby("sess")]
    means = np.empty(n_draws)
    for d in range(n_draws):
        react: list[float] = []
        for x in sessions:
            close = x["close"].to_numpy(dtype=float)
            lo = float(x["low"].min()); hi = float(x["high"].max())
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            lvl = rng.uniform(lo, hi)
            react.extend(_session_reactions(close, np.full_like(close, lvl), None,
                                            horizon, lag))
        means[d] = np.nanmean(react) if react else 0.0
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(reactions: np.ndarray, cost_bps: float = 1.0) -> dict:
    """Mean post-touch reaction gross vs net of a round-trip half-spread.

    Every touch is a fresh round trip (enter on the next bar, exit ``horizon`` bars later):
    ``2 * cost_bps`` charged (in + out). ``cost_bps`` is the one-way effective half-spread on the
    most liquid US tape (~1 bp for SPY/QQQ). Intraday reactions are tiny, so the spread is the
    binding constraint — this is where the magnet usually dies.
    """
    r = np.asarray(reactions, dtype=float)
    r = r[np.isfinite(r)]
    gross = float(r.mean()) if len(r) else float("nan")
    net = gross - 2.0 * cost_bps / 1e4
    return {"n": int(len(r)), "gross": gross, "net": net,
            "win": float((r > 0).mean()) if len(r) else float("nan"),
            "cost_bps": cost_bps}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def summarize(bars: pd.DataFrame, horizon: int, lag: int = 1, n_levels: int = 20,
              placebo: bool = False, n_draws: int = 2000, cost_bps: float = 1.0,
              seed: int = 442) -> dict:
    """Headline stats for one horizon on one ticker's bars.

    Computes the AVWAP, pools the post-touch reactions, the random-level control reactions, the
    one-sample + HAC t of AVWAP, the Welch t of AVWAP minus control, the net-of-cost reaction,
    and (optionally) the random-level placebo p.
    """
    df = session_avwap(bars)
    av = avwap_reactions(df, horizon, lag)
    rc = random_level_reactions(df, horizon, lag, n_levels=n_levels, seed=seed)
    nc = net_of_costs(av, cost_bps=cost_bps)
    p = placebo_pvalue(df, horizon, lag, n_draws=n_draws, seed=seed)["p_value"] \
        if placebo else float("nan")
    return {
        "horizon": horizon, "n_avwap": int(len(av)), "n_ctrl": int(len(rc)),
        "avwap_mean": float(np.nanmean(av)) if len(av) else float("nan"),
        "ctrl_mean": float(np.nanmean(rc)) if len(rc) else float("nan"),
        "t_one": ttest_vs_zero(av), "t_hac": hac_t(av),
        "t_vs_ctrl": welch_t(av, rc),
        "win": nc["win"], "gross": nc["gross"], "net": nc["net"],
        "p_placebo": p, "cost_bps": cost_bps,
    }


def run_experiment(bars_by_ticker: dict[str, pd.DataFrame], horizon: int = 6,
                   lag: int = 1, n_levels: int = 20, cost_bps: float = 1.0,
                   placebo: bool = False, n_draws: int = 2000, seed: int = 442) -> dict:
    """Pool all tickers into one basket-wide AVWAP magnet test at one horizon.

    Concatenates each ticker's per-session AVWAP reactions and random-level controls, then runs
    the same inference on the pooled samples. Returns the pooled headline dict plus per-ticker
    AVWAP means for the dispersion picture.
    """
    av_all: list[float] = []
    rc_all: list[float] = []
    per_ticker = {}
    for tk, bars in bars_by_ticker.items():
        df = session_avwap(bars)
        av = avwap_reactions(df, horizon, lag)
        rc = random_level_reactions(df, horizon, lag, n_levels=n_levels, seed=seed)
        av_all.extend(av.tolist())
        rc_all.extend(rc.tolist())
        per_ticker[tk] = float(np.nanmean(av)) if len(av) else float("nan")
    av_all = np.asarray(av_all, dtype=float)
    rc_all = np.asarray(rc_all, dtype=float)
    nc = net_of_costs(av_all, cost_bps=cost_bps)
    out = {
        "horizon": horizon, "n_avwap": int(len(av_all)), "n_ctrl": int(len(rc_all)),
        "avwap_mean": float(np.nanmean(av_all)) if len(av_all) else float("nan"),
        "ctrl_mean": float(np.nanmean(rc_all)) if len(rc_all) else float("nan"),
        "t_one": ttest_vs_zero(av_all), "t_hac": hac_t(av_all),
        "t_vs_ctrl": welch_t(av_all, rc_all),
        "win": nc["win"], "gross": nc["gross"], "net": nc["net"],
        "cost_bps": cost_bps, "per_ticker": per_ticker,
    }
    if placebo:
        # pooled placebo: draw random levels across all sessions of all tickers
        rng = np.random.default_rng(seed)
        all_sessions = []
        for bars in bars_by_ticker.values():
            df = session_avwap(bars)
            all_sessions.extend([x for _, x in df.groupby("sess")])
        means = np.empty(n_draws)
        for d in range(n_draws):
            react: list[float] = []
            for x in all_sessions:
                close = x["close"].to_numpy(dtype=float)
                lo = float(x["low"].min()); hi = float(x["high"].max())
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    continue
                lvl = rng.uniform(lo, hi)
                react.extend(_session_reactions(close, np.full_like(close, lvl), None,
                                                horizon, lag))
            means[d] = np.nanmean(react) if react else 0.0
        out["p_placebo"] = float((means >= out["avwap_mean"]).mean())
    return out
