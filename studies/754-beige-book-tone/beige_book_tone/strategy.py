"""Strategy + inference for Study 754 — Beige-Book-Tone.

The believers' rule (Fed-watching folklore): a **positive-tone Beige Book** — the Fed's
anecdotal survey read through the Loughran-McDonald finance dictionary — precedes an equity
**drift up** in the days after release. Operationalised as an event study on the real
release calendar aligned to daily SPY:

    Each release has a tone ``s`` (the LM net-tone proxy). A release is "POSITIVE" when
    ``s > thresh`` (default 0). For horizon ``h`` trading days we measure the forward return
    from the **release-day close** to ``h`` days later (the Beige Book prints ~2pm ET, so by
    the 4pm close it is public — the close-to-close drift carries NO look-ahead).

We test it by splitting per-event forward returns into a POSITIVE-tone set and a
NEGATIVE-tone set and comparing each to the unconditional per-event base rate, with:

  * a **Welch two-sample t** of the POSITIVE-set forward mean against the base mean;
  * a **placebo / randomization null** — draw the same number of random release-like
    events many times and ask how often chance is at least as bullish as the POSITIVE set;
  * a **continuous tone->drift regression** with a **Newey-West (HAC) t** on the slope —
    the honest test that the *amount* of tone maps to the *amount* of drift;
  * **one execution convention** — the release-day close is the entry (the book is public
    by then); the forward window runs from there (no look-ahead);
  * **one-way costs** × turnover for a tradable "long the post-release window on a positive
    book" overlay, raced against always being long that window.

The decisive question is not whether the Beige Book *describes* the economy (it does, by
construction — it is a survey of current conditions) but whether its *tone* leads the
**price**, which has had every prior data point and the FOMC's own signalling to digest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)            # forward horizons in trading days


# --------------------------------------------------------------------------- #
# Event alignment + forward returns (release-day close = entry, no look-ahead)
# --------------------------------------------------------------------------- #
def _entry_positions(spy: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions in ``spy`` of the first trading day on/after each release date.

    A release Wednesday that is a market holiday resolves forward to the next trading day.
    Releases past the last trading day are marked -1 and dropped by callers.
    """
    pos = spy.index.searchsorted(dates, side="left")
    pos = np.where(pos >= len(spy), -1, pos)
    return pos


def event_forward_returns(spy: pd.Series, dates: pd.DatetimeIndex, h: int) -> pd.Series:
    """Per-event ``h``-day forward SPY return from the release-day close.

    Entry = close of the first trading day on/after the release (the book is public by the
    close). Return = spy[i+h]/spy[i] - 1. NaN (dropped) where the window overruns the tape.
    Indexed by the release date.
    """
    vals = spy.values
    n = len(vals)
    pos = _entry_positions(spy, dates)
    out = np.full(len(dates), np.nan)
    for j, i in enumerate(pos):
        if i < 0 or i + h >= n:
            continue
        out[j] = vals[i + h] / vals[i] - 1.0
    return pd.Series(out, index=dates, name=f"fwd_{h}d")


def split_returns(rel: pd.DataFrame, spy: pd.Series, h: int, thresh: float = 0.0):
    """(positive_fwd, negative_fwd, all_fwd) arrays of per-event forward returns, NaNs dropped."""
    fwd = event_forward_returns(spy, rel.index, h)
    tone = rel["tone"].reindex(fwd.index)
    ok = fwd.notna() & tone.notna()
    fwd, tone = fwd[ok], tone[ok]
    pos = fwd[tone > thresh].values.astype(float)
    neg = fwd[tone <= thresh].values.astype(float)
    allv = fwd.values.astype(float)
    return pos, neg, allv


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if sample < 2."""
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(rel: pd.DataFrame, spy: pd.Series, h: int, thresh: float = 0.0,
                   n_draws: int = 20_000, seed: int = 754) -> dict:
    """Small-sample placebo null for the drift claim.

    Draw ``k`` random per-event forward returns (k = number of POSITIVE releases) many
    times and ask how often a random draw's mean is **at least** the POSITIVE set's mean
    (i.e. as bullish or more). p = P[random-draw mean >= positive mean]. A real drift => small p.
    """
    pos, _neg, allv = split_returns(rel, spy, h, thresh=thresh)
    k = len(pos)
    if k == 0 or len(allv) == 0:
        return {"k": 0, "pos_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    n = len(allv)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = allv[rng.integers(0, n, size=k)].mean()
    obs = float(pos.mean())
    return {"k": k, "pos_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs).mean())}


def summarize(rel: pd.DataFrame, spy: pd.Series, h: int, thresh: float = 0.0) -> dict:
    """Headline stats for one horizon: n, POSITIVE vs NEGATIVE vs base forward means and
    up-rates, the Welch t (POSITIVE vs base), and the placebo p (drift)."""
    pos, neg, allv = split_returns(rel, spy, h, thresh=thresh)
    pl = placebo_pvalue(rel, spy, h, thresh=thresh)
    return {
        "h": h,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "pos_mean": float(pos.mean()) if len(pos) else float("nan"),
        "neg_mean": float(neg.mean()) if len(neg) else float("nan"),
        "base_mean": float(allv.mean()) if len(allv) else float("nan"),
        "pos_uprate": float((pos > 0).mean()) if len(pos) else float("nan"),
        "base_uprate": float((allv > 0).mean()) if len(allv) else float("nan"),
        "t": welch_t(pos, allv),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# Continuous tone -> drift regression with a Newey-West (HAC) t on the slope
# --------------------------------------------------------------------------- #
def _newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int) -> float:
    """HAC (Newey-West) standard error of the slope in a simple OLS y ~ a + b x."""
    n = len(x)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx == 0:
        return float("nan")
    u = xc * resid                                   # score for the slope
    s = float((u ** 2).sum())
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)                    # Bartlett kernel
        s += 2.0 * w * float((u[L:] * u[:-L]).sum())
    return float(np.sqrt(s) / sxx)


def tone_drift_regression(rel: pd.DataFrame, spy: pd.Series, h: int,
                          lags: int = 4) -> dict:
    """Regress per-event h-day forward return on tone: r = a + b*tone + e.

    Reports the slope ``b`` (extra drift per unit of tone), the ordinary and Newey-West
    (HAC, Bartlett, ``lags``) t-statistics on ``b``, and the Pearson correlation. A real
    "tone leads the tape" effect needs ``b > 0`` with HAC ``|t| >= 2``.
    """
    fwd = event_forward_returns(spy, rel.index, h)
    tone = rel["tone"].reindex(fwd.index)
    ok = fwd.notna() & tone.notna()
    y = fwd[ok].values.astype(float)
    x = tone[ok].values.astype(float)
    n = len(y)
    if n < 3 or x.std() == 0:
        return {"n": n, "beta": float("nan"), "t_ols": float("nan"),
                "t_hac": float("nan"), "corr": float("nan")}
    xc, yc = x - x.mean(), y - y.mean()
    beta = float((xc * yc).sum() / (xc ** 2).sum())
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    # ordinary SE
    s2 = float((resid ** 2).sum() / (n - 2))
    se_ols = float(np.sqrt(s2 / (xc ** 2).sum()))
    se_hac = _newey_west_se(x, resid, lags)
    corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": n, "beta": beta,
        "t_ols": beta / se_ols if se_ols else float("nan"),
        "t_hac": beta / se_hac if se_hac and not np.isnan(se_hac) else float("nan"),
        "corr": corr,
    }


# --------------------------------------------------------------------------- #
# Tradability — "long the post-release window on a positive book" overlay
# --------------------------------------------------------------------------- #
def event_overlay(rel: pd.DataFrame, spy: pd.Series, h: int = 5, thresh: float = 0.0,
                  cost_bps: float = 1.0) -> dict:
    """Go long SPY for ``h`` trading days after each POSITIVE-tone Beige Book, else flat.

    Entry at the release-day close, exit ``h`` days later; a round-trip costs ``2 * cost_bps``
    (one-way × NAV, charged on entry and exit). Compares the per-event net return of the
    POSITIVE-tone events to (a) the per-event base rate over ALL events and (b) simple
    buy-and-hold over the whole tape, annualised by the release frequency. Gross and net
    both reported; price-only total-return SPY (no separate cash leg — the strategy is flat
    between windows, a conservative, labelled simplification).
    """
    pos, _neg, allv = split_returns(rel, spy, h, thresh=thresh)
    c = cost_bps / 1e4
    per_event_gross = float(pos.mean()) if len(pos) else float("nan")
    per_event_net = per_event_gross - 2.0 * c
    base_event = float(allv.mean()) if len(allv) else float("nan")

    # annualise: ~8 releases/yr, but only positive ones are traded
    yrs = (rel.index.max() - rel.index.min()).days / 365.25
    trades_per_yr = len(pos) / yrs if yrs > 0 else float("nan")
    ann_net = per_event_net * trades_per_yr
    # per-event Sharpe (event returns, not annualised) for a scale-free read
    ev_sharpe = (float(pos.mean()) / float(pos.std(ddof=1))
                 if len(pos) > 1 and pos.std(ddof=1) > 0 else float("nan"))

    # buy-and-hold annualised over the same span (total-return SPY)
    bh_days = len(spy)
    bh_ann = float((spy.iloc[-1] / spy.iloc[0]) ** (252.0 / bh_days) - 1.0) if bh_days > 1 else float("nan")

    return {
        "h": h, "n_trades": int(len(pos)), "cost_bps": cost_bps,
        "per_event_gross": per_event_gross, "per_event_net": per_event_net,
        "base_event": base_event, "event_sharpe": ev_sharpe,
        "trades_per_yr": trades_per_yr, "ann_net": ann_net, "bh_ann": bh_ann,
    }
