"""Data layer for Study 327 (Disposition-Effect / capital-gains-overhang).

The disposition effect (Shefrin & Statman 1985) is the documented tendency of investors
to **sell winners too early and ride losers too long**. Grinblatt & Han (2005) turned it
into a *cross-sectional* asset-pricing claim: if a stock sits on a large **unrealised
capital gain**, disposition-prone holders create excess *selling* pressure that holds the
price below fundamentals — an under-reaction the patient buyer can harvest. The proxy is
**capital-gains overhang** ``g = (P - R) / P`` where ``R`` is a turnover-weighted
*reference price* (the volume-weighted average cost basis of current holders). Stocks with
a high overhang (deep in the money) are predicted to **out**perform; deep-underwater stocks
(negative overhang) under-perform.

Two tapes, one shape (a month × ticker panel of overhang and forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A tunable ``overhang_premium``
  knob controls how strongly the overhang signal predicts next-month returns;
  ``overhang_premium=0`` is the null — the sort is a coin, so tests assert the edge appears
  only when the premium is planted. This is the positive control AND the null in one bottle.
- ``load_real`` — the real tape: monthly overhang built from daily price + volume for a
  small, *named* cross-section (Dow-30-ish liquid large caps), cache-first via a parquet
  under the study ``_cache/`` (or the shared desk ``_cache/``), with a graceful
  ``yfinance`` fallback that only fires on an explicit ``fetch=True``.

**Survivorship bias** is explicit and *named on the Signal axis*: the real cross-section is
a hand-picked list of *current* large caps projected backwards (``allow_survivorship_bias``
must be passed True to ``load_real``), so any cross-sectional magnitude is an upper bound.
The bias direction here is subtle — survivors are precisely the names whose past gains were
*not* mean-reverted, which can bias the overhang premium in *either* direction — so we treat
the real tape as illustrative, never certifying.

No look-ahead: the overhang on the last day of month M (a quantity known at that close)
predicts the return *of month M+1*. One execution lag, applied once, in ``strategy``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
STUDY_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

# A small, named, liquid large-cap cross-section (Dow-30-style). Hand-picked => survivorship.
REAL_UNIVERSE = [
    "AAPL", "MSFT", "JPM", "JNJ", "PG", "KO", "XOM", "WMT", "HD", "MCD",
    "CVX", "MRK", "CAT", "IBM", "DIS", "CSCO", "VZ", "INTC", "AXP", "BA",
    "GS", "MMM", "NKE", "TRV", "UNH", "WBA", "AMGN", "HON", "CRM", "DOW",
]

# Grinblatt-Han reference-price decay horizon (their paper uses ~260 trading weeks; on a
# daily tape we use a 5-year (1260-day) turnover-weighted window).
REF_WINDOW_DAYS = 1260
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted premium parameters for a synthetic panel."""

    overhang_premium: float  # bps of extra monthly return per unit of overhang z-score
    momentum_premium: float  # nuisance: a separate 12-1 momentum loading (the known confound)

    @property
    def has_overhang_edge(self) -> bool:
        return self.overhang_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 60,
    n_months: int = 240,
    overhang_premium: float = 0.004,
    momentum_premium: float = 0.006,
    base_ret: float = 0.006,
    ret_vol: float = 0.075,
    seed: int = 327,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible firm × month panel with a known capital-gains-overhang effect.

    Each firm follows a monthly log-price random walk. The **overhang** is built the same
    way the real loader does — ``g = (P - R)/P`` with ``R`` an exponentially turnover-weighted
    trailing reference price — so the synthetic signal has the same construction (and the same
    mechanical correlation with momentum) as the real one. Next month's return is::

        ret_{t+1} = base_ret + overhang_premium * z(overhang_t)
                            + momentum_premium * z(mom12_1_t) + noise

    where the premiums are the planted cross-sectional loadings. ``overhang_premium = 0`` is
    the null — the overhang sort becomes a coin flip; ``momentum_premium`` plants the known
    confound (overhang and 12-1 momentum are mechanically correlated) so a test can check the
    engine *isolates* overhang rather than re-discovering momentum.

    Returns ``(overhang, mom, fwd_ret, truth)`` all indexed by a decorative monthly period
    (as a Timestamp) with columns = firm identifiers.
    """
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # --- raw monthly log returns (the price process) ---
    mret = base_ret + ret_vol * rng.standard_normal((n_months, n_firms))
    log_px = np.cumsum(mret, axis=0)
    px = 100.0 * np.exp(log_px)  # shape (n_months, n_firms), strictly positive

    # --- overhang via an EW turnover-weighted reference price (monthly approx) ---
    # R_t = lambda * P_{t-1} + (1-lambda) * R_{t-1}; g_t = (P_t - R_t)/P_t.
    lam = 0.10  # ~10-month memory; the monthly analogue of the daily 5y window
    ref = np.empty_like(px)
    ref[0] = px[0]
    for t in range(1, n_months):
        ref[t] = lam * px[t - 1] + (1.0 - lam) * ref[t - 1]
    overhang = (px - ref) / px

    # --- 12-1 momentum (the confound): cumulative return t-12..t-1 ---
    mom = np.full((n_months, n_firms), np.nan)
    for t in range(13, n_months):
        mom[t] = px[t - 1] / px[t - 13] - 1.0

    # --- cross-sectional z-scores within each month ---
    def _z(a: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a)
        for t in range(a.shape[0]):
            row = a[t]
            valid = np.isfinite(row)
            if valid.sum() < 2:
                continue
            mu = row[valid].mean()
            sd = row[valid].std() + 1e-9
            out[t, valid] = (row[valid] - mu) / sd
        return out

    z_over = _z(overhang)
    z_mom = _z(mom)

    noise = ret_vol * rng.standard_normal((n_months, n_firms))
    fwd = base_ret + overhang_premium * z_over + momentum_premium * z_mom + noise

    # Decorative monthly index (period_range -> Timestamp; never date_range(periods=BIG)).
    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp()
    idx.name = "month"
    over_df = pd.DataFrame(overhang, index=idx, columns=firms)
    mom_df = pd.DataFrame(mom, index=idx, columns=firms)
    fwd_df = pd.DataFrame(fwd, index=idx, columns=firms)
    return over_df, mom_df, fwd_df, WorldTruth(overhang_premium, momentum_premium)


# ---------------------------------------------------------------------------
# Grinblatt-Han overhang from a daily price/volume tape
# ---------------------------------------------------------------------------
def reference_price(close: pd.Series, volume: pd.Series,
                    window: int = REF_WINDOW_DAYS) -> pd.Series:
    """Turnover-weighted reference price (the Grinblatt-Han cost-basis proxy).

    ``R_t = sum_{k=1..window} w_{t,k} * P_{t-k}`` where the weight on day ``t-k`` is the
    turnover on ``t-k`` times the probability that the shares bought then have *not* yet
    been traded since (a survival product of (1 - turnover)). The weights are normalised to
    sum to 1 over the window. Turnover is proxied by ``volume`` rescaled to a daily fraction
    (shares outstanding is unobserved here, so we use a robust rank of volume — the exact
    scale washes out in the normalisation).

    Returns ``R`` aligned to ``close.index`` (NaN until ``window`` observations exist).
    """
    p = close.to_numpy(dtype=float)
    v = volume.to_numpy(dtype=float)
    n = len(p)
    # Daily turnover proxy in (0,1): squash volume to a stable fraction. The normalisation
    # below makes the absolute scale irrelevant; only the *relative* daily turnover matters.
    med = np.nanmedian(v[v > 0]) if np.any(v > 0) else 1.0
    turn = np.clip(v / (5.0 * med + 1e-12), 1e-4, 0.5)

    one_minus_turn = 1.0 - turn
    out = np.full(n, np.nan)
    for t in range(window, n):
        ks = np.arange(1, window + 1)            # lags 1..window (most recent first)
        tk = turn[t - ks]                        # turnover on day t-k
        # Survival weight: shares bought on day t-k survive untraded through t-1 with
        # probability prod_{j=1..k-1} (1 - turnover_{t-j}). For k=1 the product is empty (=1).
        surv = np.concatenate(([1.0], np.cumprod(one_minus_turn[t - 1:t - window:-1])))
        w = tk * surv
        s = w.sum()
        if s > 0:
            out[t] = float((w * p[t - ks]).sum() / s)
    return pd.Series(out, index=close.index, name="ref")


def overhang_from_bars(close: pd.Series, volume: pd.Series,
                       window: int = REF_WINDOW_DAYS) -> pd.Series:
    """Daily capital-gains overhang ``g_t = (P_t - R_t)/P_t`` for one name."""
    ref = reference_price(close, volume, window)
    return (close - ref) / close


# ---------------------------------------------------------------------------
# Real tape — small named cross-section, cache-first
# ---------------------------------------------------------------------------
def _panel_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "disposition_panel.parquet")


def load_real(
    *,
    allow_survivorship_bias: bool = False,
    fetch: bool = False,
    cache_dir: str | None = None,
    universe: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly overhang, 12-1 momentum, and next-month returns for the real cross-section.

    Returns ``(overhang, mom, fwd_ret)`` — all month × ticker DataFrames where the signal at
    month-end M aligns to the return of month M+1 (one calendar-known lag, applied once).

    **Cache-first.** Reads ``disposition_panel.parquet`` from the study ``_cache/`` (or the
    shared desk ``_cache/``). On a cache miss it returns three empty DataFrames unless
    ``fetch=True``, in which case it pulls daily bars from ``yfinance``, builds the panel and
    writes the cache. The reproducible core and the test-suite never reach this branch.

    **Survivorship.** The universe is a hand-picked list of *current* large caps; callers must
    pass ``allow_survivorship_bias=True`` to proceed (the opt-in guard, mirroring
    ``quantlab.universe``). Any cross-sectional magnitude from this tape is therefore an upper
    bound, and the caveat travels with every published number.
    """
    if not allow_survivorship_bias:
        raise SurvivorshipBiasError(
            "load_real() builds a cross-section on a hand-picked list of *current* large caps "
            "projected backwards, so it is survivorship-biased. Pass "
            "allow_survivorship_bias=True to opt in and treat magnitudes as upper bounds."
        )

    uni = list(universe) if universe is not None else list(REAL_UNIVERSE)

    # Cache-first: study cache, then shared cache.
    for cdir in [cache_dir] if cache_dir else [STUDY_CACHE, SHARED_CACHE]:
        path = _panel_cache_path(cdir)
        if os.path.exists(path):
            panel = pd.read_parquet(path)
            return _split_panel(panel)

    if not fetch:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    panel = _build_real_panel(uni)
    out_dir = cache_dir or STUDY_CACHE
    os.makedirs(out_dir, exist_ok=True)
    panel.to_parquet(_panel_cache_path(out_dir))
    return _split_panel(panel)


def _split_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A stored long panel (columns: month, ticker, overhang, mom, fwd_ret) -> three wides."""
    over = panel.pivot(index="month", columns="ticker", values="overhang").sort_index()
    mom = panel.pivot(index="month", columns="ticker", values="mom").sort_index()
    fwd = panel.pivot(index="month", columns="ticker", values="fwd_ret").sort_index()
    for d in (over, mom, fwd):
        d.index = pd.DatetimeIndex(d.index)
        d.index.name = "month"
    return over, mom, fwd


def _build_real_panel(universe: list[str]) -> pd.DataFrame:
    """Download daily bars, build the monthly overhang/momentum/fwd-return long panel."""
    import yfinance as yf  # lazy: only on fetch

    recs = []
    for tk in universe:
        raw = yf.download(tk, start="2005-01-01", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].astype(float)
        vol = raw["Volume"].astype(float)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
            vol.index = vol.index.tz_localize(None)

        g = overhang_from_bars(close, vol)
        # Month-end snapshots.
        m_close = close.resample("ME").last()
        m_g = g.resample("ME").last()
        m_ret = m_close.pct_change()
        mom12_1 = m_close.shift(1) / m_close.shift(13) - 1.0
        fwd = m_ret.shift(-1)  # next-month return aligned to month-end signal

        df = pd.DataFrame({
            "month": m_close.index,
            "ticker": tk,
            "overhang": m_g.to_numpy(),
            "mom": mom12_1.to_numpy(),
            "fwd_ret": fwd.to_numpy(),
        })
        recs.append(df)
    if not recs:
        return pd.DataFrame(columns=["month", "ticker", "overhang", "mom", "fwd_ret"])
    panel = pd.concat(recs, ignore_index=True)
    # Drop the last (partial / NaN-forward) month per the as-of discipline.
    panel = panel.dropna(subset=["overhang", "fwd_ret"])
    return panel


class SurvivorshipBiasError(RuntimeError):
    """Raised when a survivorship-biased cross-section is used without the opt-in."""


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint (SHA-1 of flattened values), for the as-of stamp."""
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
