"""Strategy + inference for Study 760 — Michigan-Sentiment-Day.

Two believer claims about the University of Michigan Consumer Sentiment release, tested
on their own tapes:

**A. The release-day drift.** The sentiment print (preliminary, mid-month Friday, 10:00
ET) is a market-moving number, so SPY should *react* on release day and, if the print
carries a surprise, *drift* in the surprise's direction the next day (a tradable
post-announcement drift). We measure:

  * the **release-day** SPY return vs an average day (an event-study mean);
  * the **next-day** SPY return conditional on the surprise sign
    (surprise = sentiment_t − sentiment_{t−1}), entered at the release-day close (so the
    print is already public — no look-ahead) — the *tradable* drift.

**B. The level / regime test.** The contrarian folklore (Fisher–Statman): buy when
sentiment is **low and turning up** — "low-then-rising marks bottoms." We split forward
1/3/6/12-month SPY returns by a sentiment **regime** — LOW (below the trailing 30th
percentile, expanding, no look-ahead) and RISING (level above its value three months
prior) — and compare the LOW-and-RISING set to the unconditional base rate, with:

  * a **Welch two-sample t** (LOW-and-RISING mean vs the unconditional mean);
  * a **circular block bootstrap** p-value with a 12-month block, because overlapping
    H-month forward returns are heavily autocorrelated and the naive Welch t on them is
    *inflated* — the block bootstrap respects that dependence (White/Politis-Romano);
  * an **episode count** (regime months separated by > 2-month gaps) — the honest measure
    of how many *independent* bottoms the signal actually rests on;
  * a cost-net **timing overlay** raced against buy-and-hold.

The decisive tension: the level test *looks* strong on the naive t precisely where the
statistic is least trustworthy (long overlapping horizons, a few clustered post-crash
recoveries), and the release-day drift is a non-event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)
ANN = 12


# ===========================================================================
# A. RELEASE-DAY DRIFT (daily tape)
# ===========================================================================
def _next_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp):
    """First trading day on or after ``d`` (None if past the tape)."""
    pos = idx.searchsorted(d)
    return idx[pos] if pos < len(idx) else None


def release_day_returns(spy: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Close-to-close SPY return on each release day (next trading day on/after the date)."""
    ret = spy.pct_change()
    idx = ret.index
    rel = pd.DatetimeIndex(
        [t for t in (_next_trading_day(idx, d) for d in dates) if t is not None and t <= idx[-1]]
    )
    return ret.reindex(rel).dropna()


def surprise_sign(sent: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Sign of the month-over-month change in sentiment, aligned to each release date.

    surprise_t = sent_t − sent_{t−1} (a labelled proxy for beat-vs-consensus, since no free
    history of the Street's forecast exists). Indexed by release-day trading date.
    """
    lvl = sent.copy()
    lvl.index = [pd.Timestamp(d.year, d.month, 1) for d in lvl.index]
    chg = lvl - lvl.shift(1)
    return chg


def drift_by_surprise(spy: pd.Series, sent: pd.Series, dates: pd.DatetimeIndex,
                      lag: int = 1) -> dict:
    """Next-``lag``-day SPY return after each release, split by the surprise sign.

    Enter at the release-day close (the print is public); capture the return over the
    following ``lag`` trading day(s). A tradable post-announcement drift => BEAT-day drift
    significantly above MISS-day drift. Returns means (bp) and a Welch t of (beat − miss).
    """
    ret = spy.pct_change()
    idx = ret.index
    pos = {d: i for i, d in enumerate(idx)}
    chg = surprise_sign(sent, dates)
    beat_r, miss_r = [], []
    for d in dates:
        t = _next_trading_day(idx, d)
        if t is None or t >= idx[-1]:
            continue
        i = pos[t]
        if i + lag >= len(idx):
            continue
        fwd = float(spy.iloc[i + lag] / spy.iloc[i] - 1.0)
        s = chg.get(pd.Timestamp(d.year, d.month, 1), np.nan)
        if np.isnan(s):
            continue
        (beat_r if s > 0 else miss_r).append(fwd)
    beat_r, miss_r = np.array(beat_r), np.array(miss_r)
    return {
        "n_beat": int(len(beat_r)), "n_miss": int(len(miss_r)),
        "beat_mean": float(beat_r.mean()) if len(beat_r) else float("nan"),
        "miss_mean": float(miss_r.mean()) if len(miss_r) else float("nan"),
        "t_beat_minus_miss": welch_t(beat_r, miss_r),
    }


def release_day_summary(spy: pd.Series, dates: pd.DatetimeIndex) -> dict:
    """Release-day mean return vs the all-day mean, with a Welch t."""
    rel = release_day_returns(spy, dates)
    alld = spy.pct_change().dropna()
    return {
        "n_release": int(len(rel)), "n_all": int(len(alld)),
        "release_mean": float(rel.mean()), "all_mean": float(alld.mean()),
        "t_vs_all": welch_t(rel.values, alld.values),
    }


# ===========================================================================
# B. LEVEL / REGIME TEST (monthly tape)
# ===========================================================================
def sentiment_pct(frame: pd.DataFrame, min_periods: int = 36) -> pd.Series:
    """Expanding percentile rank of the current sentiment level (no look-ahead).

    pct_t = share of months up to and including t whose level is <= level_t. Uses only
    past+present data (expanding), so it is knowable in real time.
    """
    return frame["sent"].expanding(min_periods=min_periods).apply(
        lambda x: float((x.iloc[-1] >= x).mean()), raw=False)


def regime_mask(frame: pd.DataFrame, low_q: float = 0.30, k: int = 3,
                min_periods: int = 36) -> dict:
    """Boolean masks: LOW (pct <= low_q), RISING (level > level_{t-k}), and LOW&RISING."""
    pct = sentiment_pct(frame, min_periods=min_periods)
    low = pct <= low_q
    rising = frame["sent"] > frame["sent"].shift(k)
    return {"low": low, "rising": rising, "low_rising": (low & rising)}


def forward_returns(frame: pd.DataFrame, months: int, lag: int = 0) -> pd.Series:
    """Forward ``months``-month SPY return, entered ``lag`` months after the signal month.

    Default ``lag = 0``: the sentiment print is already public mid-month, so entering at
    month-``t`` end (after the print) and holding to t+months carries no look-ahead.
    """
    spy = frame["spy"]
    entry = spy.shift(-lag)
    return spy.shift(-lag - months) / entry - 1.0


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) − mean(base) (unequal variance). NaN if either < 2."""
    sample, base = np.asarray(sample, float), np.asarray(base, float)
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((sample.mean() - base.mean()) / se)


def block_bootstrap_p(frame: pd.DataFrame, months: int, low_q: float = 0.30, k: int = 3,
                      block: int = 12, n_draws: int = 5000, seed: int = 760,
                      min_periods: int = 36) -> dict:
    """Circular-block-bootstrap p for the LOW&RISING forward-return excess.

    Overlapping ``months``-horizon returns are autocorrelated; the naive Welch t on them
    over-rejects. We resample the full forward-return series in circular blocks of length
    ``block`` (default 12, matched to the worst overlap), read off the first ``k`` values
    as a pseudo-signal set (k = number of real signal months), and ask how often the
    pseudo-excess is >= the observed excess. p = P[block-boot excess >= observed].
    """
    fwd = forward_returns(frame, months)
    mask = regime_mask(frame, low_q=low_q, k=k, min_periods=min_periods)["low_rising"]
    ok = fwd.notna() & mask.notna()
    fwd_ok, mask_ok = fwd[ok], mask[ok].astype(bool)
    sig = fwd_ok[mask_ok].values
    allv = fwd_ok.values
    kk, n = len(sig), len(allv)
    if kk == 0 or n == 0:
        return {"k": 0, "obs_excess": float("nan"), "p_value": float("nan")}
    obs = float(sig.mean() - allv.mean())
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_draws):
        pos = []
        while len(pos) < n:
            s = int(rng.integers(0, n))
            pos.extend(range(s, s + block))
        pos = [p % n for p in pos[:n]]
        boot = allv[pos]
        if (boot[:kk].mean() - boot.mean()) >= obs:
            cnt += 1
    return {"k": kk, "obs_excess": obs, "p_value": cnt / n_draws}


def n_episodes(frame: pd.DataFrame, low_q: float = 0.30, k: int = 3, gap: int = 2,
               min_periods: int = 36) -> int:
    """Independent LOW&RISING episodes (signal months separated by > ``gap`` months)."""
    mask = regime_mask(frame, low_q=low_q, k=k, min_periods=min_periods)["low_rising"]
    fire = frame.index[mask.fillna(False)]
    if len(fire) == 0:
        return 0
    ords = np.array([d.year * 12 + d.month for d in fire])
    return int(1 + (np.diff(ords) > gap).sum())


def summarize_regime(frame: pd.DataFrame, months: int, low_q: float = 0.30, k: int = 3,
                     min_periods: int = 36, boot: bool = True) -> dict:
    """Headline stats for one horizon: LOW / LOW&RISING / base forward means, Welch t, and
    the block-bootstrap p (the honest, autocorrelation-aware significance)."""
    fwd = forward_returns(frame, months)
    m = regime_mask(frame, low_q=low_q, k=k, min_periods=min_periods)
    base = fwd.dropna().values
    lr = fwd[m["low_rising"] & fwd.notna()].dropna().values
    lo = fwd[m["low"] & fwd.notna()].dropna().values
    out = {
        "months": months,
        "n_low_rising": int(len(lr)), "n_low": int(len(lo)), "n_base": int(len(base)),
        "low_rising_mean": float(lr.mean()) if len(lr) else float("nan"),
        "low_mean": float(lo.mean()) if len(lo) else float("nan"),
        "base_mean": float(base.mean()) if len(base) else float("nan"),
        "t_low_rising": welch_t(lr, base),
        "t_low": welch_t(lo, base),
    }
    if boot:
        out["p_block"] = block_bootstrap_p(frame, months, low_q=low_q, k=k)["p_value"]
    return out


# --------------------------------------------------------------------------- #
# Tradability — the contrarian sentiment-timing overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, low_q: float = 0.30, k: int = 3, lag: int = 1,
                   cost_bps: float = 10.0, min_periods: int = 36) -> dict:
    """Long SPY only in LOW&RISING months, else in cash (the believers' bottom-buyer rule).

    Position for month t+lag = 1 if LOW&RISING at t else 0 (one-month execution lag). One-way
    cost ``cost_bps`` charged per switch (turnover one-way x NAV). Sharpe is excess-of-zero
    (cash leg earns 0 — a conservative, labelled simplification). Gross and net reported and
    raced against buy-and-hold on the same month-end tape (total-return adjusted, labelled).
    """
    spy = frame["spy"]
    r = spy.pct_change()
    mask = regime_mask(frame, low_q=low_q, k=k, min_periods=min_periods)["low_rising"]
    pos = mask.astype(float).shift(lag)
    df = pd.DataFrame({"r": r, "pos": pos}).dropna()
    switches = df["pos"].diff().abs().fillna(df["pos"].abs())
    c = cost_bps / 1e4
    gross = df["pos"] * df["r"]
    net = gross - switches * c

    def _ann(x):
        mu = x.mean() * ANN
        vol = x.std(ddof=1) * np.sqrt(ANN)
        return float(mu), float(vol), (float(mu / vol) if vol > 0 else float("nan"))

    bh_mu, bh_vol, bh_sh = _ann(df["r"])
    g_mu, _gv, g_sh = _ann(gross)
    n_mu, n_vol, n_sh = _ann(net)
    return {
        "n_months": int(len(df)), "n_switches": int((switches > 0).sum()),
        "exposure": float((df["pos"] > 0).mean()),
        "bh_mean": bh_mu, "bh_sharpe": bh_sh,
        "overlay_gross_mean": g_mu, "overlay_gross_sharpe": g_sh,
        "overlay_net_mean": n_mu, "overlay_net_vol": n_vol, "overlay_net_sharpe": n_sh,
        "cost_bps": cost_bps,
    }
