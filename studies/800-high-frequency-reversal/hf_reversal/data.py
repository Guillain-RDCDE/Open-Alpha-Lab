"""Data layer for Study 800 — High-Frequency (Weekly) Cross-Sectional Reversal.

The Jegadeesh (1990) / Lehmann (1990) short-horizon reversal, taken to the **weekly**
horizon: rank the cross-section each Friday by the *last five trading days'* return; the
biggest losers are claimed to beat the biggest winners the following week. It is the
amplified, faster-turning cousin of study 329 (the *monthly* reversal), and its central
weakness is microstructure: at a one-week horizon a large slice of the raw spread can be
pure **bid-ask bounce** rather than tradable mean-reversion.

Two tapes, one shape (a week x ticker panel of Friday closes, plus a matching panel of
Corwin-Schultz effective-spread estimates for the honest bid-ask-bounce haircut):

* ``synthetic_panel`` — a *deterministic, offline* generator with TWO tunable knobs:

    - ``reversal``  plants genuine one-week cross-sectional anti-persistence (last week's
      losers earn the most next week) that *survives* a skip-a-week gap — a real signal.
    - ``bounce``    plants pure **bid-ask bounce**: each observed weekly close lands on a
      random side (bid/ask), which manufactures a *spurious* skip=0 reversal that
      **vanishes** the moment you skip a week. This is the null in a bottle for the
      microstructure test — ``reversal = bounce = 0`` is the pure null (no spread at all).

* ``load_real`` — the real weekly close + effective-spread panel for a liquid US
  cross-section, built cache-first from the **shared daily S&P 500 OHLCV panel** already
  in the repo (``studies/01-overnight-anomaly/.../panel_503_*.parquet``, yfinance
  ``auto_adjust=True`` total-return prices). Cache-only by default so the test-suite and
  the reproducible core never touch the network.

SURVIVORSHIP GUARD — read before any cross-sectional number
-----------------------------------------------------------
The ticker list is the *current* S&P 500 membership projected backwards (the shared
panel is keyed to ``quantlab.universe.sp500_symbols``). Delisted names — the very losers
a reversal book would have been long — are absent, so every positive real-tape spread is
an **upper bound**. This caveat is NAMED on the Signal axis wherever the numbers are
published, per the desk's house rule.

No look-ahead is baked in here — that discipline lives in ``strategy.trailing_return``
(one shift, applied once). The signal for holding week t is the return known at the close
of week ``t-1-skip``; the return earned is week t's.
"""

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# This study's own weekly caches (built once from the shared daily panel).
WEEKLY_CLOSE_CACHE = os.path.join(DEFAULT_CACHE, "hfr_weekly_close.parquet")
WEEKLY_SPREAD_CACHE = os.path.join(DEFAULT_CACHE, "hfr_weekly_spread.parquet")

# Where the shared daily S&P 500 OHLCV panel lives (built by the overnight-anomaly study).
_SHARED_DAILY_GLOBS = [
    os.path.join(REPO_ROOT, "studies", "01-overnight-anomaly", "notebooks", "_cache",
                 "panel_*.parquet"),
    os.path.join(REPO_ROOT, "studies", "01-overnight-anomaly", "_cache", "panel_*.parquet"),
    os.path.join(REPO_ROOT, "studies", "25-clean-slate", "_cache", "panel_*.parquet"),
]

# Last fully-complete week pinned for the headline run (partial weeks are dropped). The
# shared daily panel ends 2026-06-11; this is the last Friday of the last complete month.
AS_OF_LAST_WEEK = "2026-05-29"

TRADING_WEEKS = 52


# ---------------------------------------------------------------------------
# Corwin-Schultz (2012) high-low effective-spread estimator
# ---------------------------------------------------------------------------
def corwin_schultz_daily(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz (2012) two-day high-low proportional effective-spread estimate.

    For each pair of consecutive days the estimator backs out the (roundtrip)
    proportional bid-ask spread from the ratio of the two-day high-low range to the
    single-day ranges. Negative point estimates (noise) are floored at zero. Returns a
    daily series aligned to the *second* day of each pair.

    This is a pure price-based liquidity proxy — no quote data needed — and it is exactly
    the quantity that drives the bid-ask-bounce haircut: illiquid, wide-spread losers pay
    more to trade than a flat basis-point assumption admits.
    """
    h = np.log(high.astype(float))
    l = np.log(low.astype(float))
    hl = (h - l) ** 2                              # single-day log-range squared
    # two-day beta and gamma
    beta = hl + hl.shift(1)
    h2 = np.maximum(h, h.shift(1))
    l2 = np.minimum(l, l.shift(1))
    gamma = (h2 - l2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s.where(s > 0, 0.0)                          # floor negative estimates at 0
    return s


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (two knobs)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 150,
    n_weeks: int = 400,
    reversal: float = 0.0,
    bounce: float = 0.0,
    base_ret: float = 0.0015,
    ret_vol: float = 0.03,
    correction_weeks: int = 3,
    seed: int = 800,
) -> tuple[pd.DataFrame, dict]:
    """A firm x week panel of *observed* weekly closes with tunable reversal + bounce.

    Two independent planted effects, so the study can tell a *real* reversal apart from a
    *microstructure mirage* — the whole point of the skip-a-week test:

    1. **Genuine, persistent reversion (``reversal``, gradual-overreaction model).** Each
       week a firm draws a transient shock ``s[t]``; the price over-reacts to it and then
       gives it back gradually over the next ``correction_weeks`` weeks::

           r[t] = base_ret + s[t] - (reversal / K) * (s[t-1] + ... + s[t-K]) + noise

       A firm that spiked last week (a winner, big ``+s``) keeps correcting *down* for the
       next few weeks, so last week's losers keep reverting *up* — and because the
       correction is spread over K weeks, the reversal **outlives a one-week skip** (still
       detectable at ``skip=1``). This is the real, tradable-in-principle signal.

    2. **Pure bid-ask bounce (``bounce``).** The observed close = ``true_price * (1 +
       bounce * e)`` with ``e = ±1`` i.i.d. per (week, firm) — the print lands on the bid
       or the ask. This injects negative first-order autocorrelation into *observed*
       returns, manufacturing a **spurious** ``skip=0`` reversal that **evaporates** the
       instant a one-week gap is inserted.

    Knob semantics:
      - ``reversal = bounce = 0``   -> pure null (no spread at any skip).
      - ``reversal > 0, bounce = 0`` -> a real reversal that *survives* ``skip=1``.
      - ``reversal = 0, bounce > 0`` -> a pure microstructure mirage: ``skip=0`` only.

    Returns ``(obs_price_df, truth)`` — a (week x firm) frame of observed cumulative
    prices (base 100) and the planted parameters. ``pd.period_range`` builds the weekly
    label so there is no ``date_range(periods=BIG)`` timestamp-overflow risk.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.period_range("2010-01-01", periods=n_weeks, freq="W-FRI").to_timestamp(how="end").normalize()
    firms = [f"F{j:03d}" for j in range(n_firms)]

    s = rng.normal(0.0, ret_vol, (n_weeks, n_firms))        # transient over-reaction shocks
    noise = rng.normal(0.0, ret_vol * 0.5, (n_weeks, n_firms))
    K = max(1, int(correction_weeks))
    log_ret = base_ret + s + noise
    for k in range(1, K + 1):
        log_ret[k:] -= (reversal / K) * s[:-k]              # give the shock back gradually

    true_price = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    e = rng.choice(np.array([-1.0, 1.0]), size=(n_weeks, n_firms))
    obs_price = true_price * (1.0 + bounce * e)

    obs_df = pd.DataFrame(obs_price, index=weeks, columns=firms)
    truth = {
        "reversal": reversal, "bounce": bounce, "base_ret": base_ret,
        "ret_vol": ret_vol, "correction_weeks": K, "n_firms": n_firms,
        "n_weeks": n_weeks, "seed": seed,
    }
    return obs_df, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first weekly close + effective-spread panel
# ---------------------------------------------------------------------------
def _shared_daily_path() -> str | None:
    """Locate the shared daily S&P 500 OHLCV panel (biggest match wins), else None."""
    hits: list[str] = []
    for pat in _SHARED_DAILY_GLOBS:
        hits.extend(glob.glob(pat))
    hits = [h for h in hits if os.path.getsize(h) > 1_000_000]  # skip tiny hourly panels
    if not hits:
        return None
    return max(hits, key=os.path.getsize)


def have_real() -> bool:
    """True if this study's weekly caches already exist (offline-ready)."""
    return os.path.exists(WEEKLY_CLOSE_CACHE) and os.path.exists(WEEKLY_SPREAD_CACHE)


def build_weekly_from_daily(daily_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (and cache) the weekly close + weekly effective-spread panels from the
    shared daily OHLCV panel. Offline: reads a parquet already on disk.

    * Weekly close = the last daily close of each ``W-FRI`` week (total-return prices).
    * Weekly spread = the mean daily Corwin-Schultz estimate within each week (the
      average roundtrip proportional spread a trader would cross that week).

    Only names with a reasonably complete history are kept. Caches two small parquets
    under this study's own ``_cache/`` and returns them, trimmed to ``AS_OF_LAST_WEEK``.
    """
    daily_path = daily_path or _shared_daily_path()
    if daily_path is None or not os.path.exists(daily_path):
        raise FileNotFoundError(
            "No shared daily S&P 500 panel found. Expected a panel_*.parquet under "
            "studies/01-overnight-anomaly/. Run that study's fetch once, or rely on the "
            "synthetic core (which covers offline CI)."
        )
    raw = pd.read_parquet(daily_path)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    tickers = list(dict.fromkeys(raw.columns.get_level_values(0)))

    closes, spreads = {}, {}
    for tk in tickers:
        try:
            sub = raw[tk]
        except KeyError:
            continue
        if not {"High", "Low", "Close"}.issubset(sub.columns):
            continue
        c = sub["Close"].dropna()
        if len(c) < 500:                              # ~2 years of daily history minimum
            continue
        wk_close = c.resample("W-FRI").last()
        cs = corwin_schultz_daily(sub["High"], sub["Low"])
        wk_spread = cs.resample("W-FRI").mean()
        closes[tk] = wk_close
        spreads[tk] = wk_spread.reindex(wk_close.index)

    close_df = pd.DataFrame(closes).sort_index()
    spread_df = pd.DataFrame(spreads).reindex(close_df.index)[close_df.columns]

    os.makedirs(DEFAULT_CACHE, exist_ok=True)
    close_df.to_parquet(WEEKLY_CLOSE_CACHE)
    spread_df.to_parquet(WEEKLY_SPREAD_CACHE)
    close_df = close_df.loc[:AS_OF_LAST_WEEK]
    spread_df = spread_df.loc[:AS_OF_LAST_WEEK]
    return close_df, spread_df


def load_real(asof: str = AS_OF_LAST_WEEK) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache-first weekly close + effective-spread panels; never touches the network.

    If this study's weekly caches exist, read them; otherwise build them once from the
    shared daily panel already on disk (still offline). Trimmed to ``asof`` (partial
    trailing week dropped). Raises ``FileNotFoundError`` if neither cache nor shared
    daily panel is available — the synthetic core covers that case for CI.

    **Survivorship bias**: current S&P 500 membership projected backwards; positive
    results are upper bounds. Named on the Signal axis.
    """
    if have_real():
        close_df = pd.read_parquet(WEEKLY_CLOSE_CACHE).sort_index()
        spread_df = pd.read_parquet(WEEKLY_SPREAD_CACHE).reindex(close_df.index)
        if close_df.index.tz is not None:
            close_df.index = close_df.index.tz_localize(None)
            spread_df.index = spread_df.index.tz_localize(None)
        return close_df.loc[:asof], spread_df.loc[:asof]
    return build_weekly_from_daily()


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a prices frame (last row) for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
