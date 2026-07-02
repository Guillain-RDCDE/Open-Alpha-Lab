"""Data layer for Study 588 (LLM-Headline-Sentiment) — the deterministic synthetic tape.

The 2020s pitch: point a large language model at the day's news headlines, have it emit a
mood score in [-1, +1], and use that score to forecast *tomorrow's* market. It is the modern
upgrade of dictionary/lexicon sentiment (Loughran-McDonald word counts) — an LLM reads
context, sarcasm and framing a bag-of-words can't. The seductive claim: richer sentiment →
real day-ahead predictability.

**Why synthetic-only.** A no-key retail stack cannot reach a long, *dated, point-in-time*
history of LLM-scored headlines: the headlines themselves are behind paywalls/licences, and
scoring them with an LLM as-of each historical day (without leaking the future the model was
trained on) is exactly the hard part. Vendor "AI sentiment" feeds are short, proprietary and
un-reproducible. So there is no honest free real tape here — the study is a **method demo**,
built entirely on a deterministic synthetic world, and is therefore capped at **WEAK** on the
SIGNAL axis (a REAL stamp needs a robust *t* >= 2 on a REAL tape; this study has none). The
data-availability limitation is stated openly on the SIGNAL axis, exactly as the repo's
lego-returns / whisky-cask / sneaker-resale synthetic-only studies do.

Two generators, one schema (a daily frame indexed by date):

- ``synthetic_series`` — one LLM-sentiment score per day plus the *forward* (next-day) market
  return. A single knob, ``beta``, plants the predictive link: ``forward_ret = beta * sentiment
  + noise``. ``beta > 0`` = a genuine day-ahead edge (the positive control); ``beta = 0`` = the
  null (sentiment is pure noise vs the future). Deterministic given ``seed``. Tests never touch
  the network — there is no network to touch.

- ``synthetic_feature_bank`` — the *overfitting trap*. A researcher rarely tests one sentiment
  recipe; they try many (different prompts, models, aggregations, lookbacks) and report the
  best. This returns ``n_features`` candidate sentiment columns, of which ``n_true`` carry a
  real ``beta`` and the rest are pure noise. The multiple-testing machinery in ``strategy.py``
  shows how the *best-of-K* looks significant even when ``n_true = 0``.

No look-ahead by construction: ``sentiment`` on day *t* is the score computed from day-*t*
headlines (known at that close); ``forward_ret`` is the *next* day's return (t -> t+1). The
one execution lag — trade tomorrow on today's score — is baked into the schema, not bolted on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 588
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic tape."""

    beta: float          # forward-return sensitivity to sentiment (0 = null)
    daily_vol: float     # idiosyncratic daily return vol

    @property
    def has_edge(self) -> bool:
        return self.beta != 0.0


# ---------------------------------------------------------------------------
# Single-signal synthetic series — the headline generator
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 1500,
    beta: float = 0.0022,
    daily_vol: float = 0.011,
    persist: float = 0.35,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily tape: one LLM-sentiment score vs the *next* day's return.

    - ``sentiment_t`` — a persistent (AR(1), coefficient ``persist``) score in ~[-1, 1],
      standing in for "the LLM read today's headlines and scored the mood". Persistence mimics
      real news mood (a bad-news regime lingers for days).
    - ``forward_ret_t`` — the market return from close *t* to close *t+1*:

          forward_ret_t = beta * sentiment_t + daily_vol * noise_t

      With ``beta > 0`` today's mood genuinely tilts tomorrow's return (the edge); ``beta = 0``
      makes sentiment pure noise against the future (the null). ``daily_vol`` sets the swamp of
      idiosyncratic return the tiny sentiment tilt must beat — realistic equity daily vol.

    The signal-to-noise here is deliberately *modest* (a small ``beta`` against ~1.1%/day vol),
    because real headline-sentiment edges, where they exist, are small. Returns ``(df, truth)``
    with columns ``sentiment`` and ``forward_ret`` on a business-day index.
    """
    rng = np.random.default_rng(seed)

    # Persistent standardized sentiment (AR(1)), then squashed into [-1, 1].
    z = np.zeros(n_days)
    innov = rng.standard_normal(n_days)
    for t in range(1, n_days):
        z[t] = persist * z[t - 1] + np.sqrt(1.0 - persist**2) * innov[t]
    sentiment = np.tanh(0.9 * z)

    noise = rng.standard_normal(n_days)
    forward_ret = beta * sentiment + daily_vol * noise

    idx = pd.bdate_range("2019-01-02", periods=n_days, freq="C")
    df = pd.DataFrame({"sentiment": sentiment, "forward_ret": forward_ret}, index=idx)
    df.index.name = "date"
    return df, WorldTruth(beta=beta, daily_vol=daily_vol)


# ---------------------------------------------------------------------------
# Feature bank — the multiple-testing / data-mining trap
# ---------------------------------------------------------------------------
def synthetic_feature_bank(
    n_days: int = 1500,
    n_features: int = 40,
    n_true: int = 0,
    true_beta: float = 0.0022,
    daily_vol: float = 0.011,
    persist: float = 0.35,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """A bank of ``n_features`` candidate sentiment signals vs one forward-return series.

    Mimics a researcher trying many sentiment recipes (prompts x models x aggregations) and
    reporting the winner. Exactly ``n_true`` of the features carry a real link to the *same*
    forward-return series (each with ``true_beta``); the rest are independent noise.

    Returns ``(features, forward_ret, is_true)``:
      - ``features`` — a ``(n_days, n_features)`` frame of persistent sentiment scores.
      - ``forward_ret`` — the single next-day return series the features are tested against.
      - ``is_true`` — a boolean mask marking which columns are genuinely predictive.

    With ``n_true = 0`` the whole bank is noise, yet the *best-of-K* feature will look
    significant — that is the trap ``strategy.py`` corrects for.
    """
    rng = np.random.default_rng(seed + 7)
    feats = np.zeros((n_days, n_features))
    for j in range(n_features):
        innov = rng.standard_normal(n_days)
        zc = np.zeros(n_days)
        for t in range(1, n_days):
            zc[t] = persist * zc[t - 1] + np.sqrt(1.0 - persist**2) * innov[t]
        feats[:, j] = np.tanh(0.9 * zc)

    is_true = np.zeros(n_features, dtype=bool)
    if n_true > 0:
        is_true[:n_true] = True

    # Forward returns driven only by the "true" features (if any), plus common noise.
    signal = feats[:, is_true].sum(axis=1) * (true_beta if n_true > 0 else 0.0)
    noise = rng.standard_normal(n_days)
    forward_ret = signal + daily_vol * noise

    idx = pd.bdate_range("2019-01-02", periods=n_days, freq="C")
    cols = [f"sent_{j:02d}" for j in range(n_features)]
    features = pd.DataFrame(feats, index=idx, columns=cols)
    features.index.name = "date"
    fr = pd.Series(forward_ret, index=idx, name="forward_ret")
    return features, fr, is_true


# ---------------------------------------------------------------------------
# Real tape — intentionally absent (synthetic-only study)
# ---------------------------------------------------------------------------
def fetch_real(fetch: bool = False) -> pd.DataFrame:
    """No real tape exists for dated, point-in-time LLM-scored headline sentiment.

    A no-key retail stack cannot reach a long, reproducible history of LLM-scored headlines
    (paywalled text + the as-of scoring problem). This function always returns an empty frame,
    by design, so the reproducible core stays fully offline and the SIGNAL axis is capped at
    WEAK. ``fetch`` is accepted for interface parity and ignored.
    """
    return pd.DataFrame()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0.0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
