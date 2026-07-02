"""Data layer for Study 579 (Equity-Bond-Corr-Flip) — the stock-bond correlation regime.

The folklore: the *sign* of the trailing equity-bond return correlation tells you when the 60/40
hedge works. For two decades (roughly 2000-2020) stocks and bonds were **negatively** correlated —
bonds rallied when stocks fell, so the bond sleeve of a 60/40 book cushioned equity drawdowns. In
2022 the correlation **flipped positive** — bonds and stocks fell together, and the 60/40 had its
worst year in a generation. The claim under test: *the sign of the rolling correlation is a timing
signal* — when it goes positive, cut the 60/40 (or the bond hedge) because diversification has
stopped working.

This study works with what a no-key retail stack can reach:

- **Daily adjusted-close prices** for **SPY** (S&P 500 ETF, the equity leg) and **TLT** (20+yr
  Treasury ETF, the bond leg), from yfinance. TLT launched 2002-07, so the honest tape starts
  there — it spans the negative-correlation era, the 2022 flip, and the aftermath.

Two tapes, one schema:

- ``synthetic_series`` — a deterministic, offline generator. A single knob, ``regime_edge``,
  plants the effect: forward 60/40 (or equity) returns are *worse* in positive-correlation months
  than in negative-correlation months. ``regime_edge = 0`` is the null (forward returns are
  independent of the correlation sign). The study's positive control and null in one bottle; tests
  never touch the network.
- ``fetch_prices`` — the real yfinance tape (daily SPY + TLT), cache-first into this study's own
  ``_cache/`` so the reproducible core never needs the network.

No look-ahead: the correlation regime at month *t* is computed from a *trailing* window ending at
month *t*'s close, and it is compared to the *forward* (t → t+1, or a multi-month horizon) return.
The signal is fully public at the close it is measured; the position is entered at that close. One
documented execution lag; the robustness sweep and synthetic control use the identical convention.

Data-availability note: this is a real-tape study, but the 2022 correlation flip is a **single**
macro regime change — there is one negative→positive transition of interest in 24 years of data,
so any "the flip predicts the 60/40 breaks" claim rests on very few effective observations. That
low effective-N is named on the SIGNAL axis, not buried.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

EQUITY = "SPY"
BOND = "TLT"
TICKERS = [EQUITY, BOND]

TRADING_DAYS = 252
MONTH_DAYS = 21


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic series."""

    regime_edge: float  # forward-return penalty (per month) when corr is positive vs negative

    @property
    def has_effect(self) -> bool:
        return self.regime_edge != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 288,
    regime_edge: float = 0.010,
    seed: int = 579,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible monthly equity/bond world with a tunable correlation-regime effect.

    We simulate a latent, slowly-switching correlation regime ``rho_t`` that spends long spells
    negative and occasionally flips positive (a crude cartoon of the real 2000-2022 history). Each
    month, equity and bond returns are drawn with that month's correlation. The *planted* effect is
    on the **forward** 60/40 return:

        fwd_6040_t = base_6040_t - regime_edge * 1{rho_t > 0}

    With ``regime_edge > 0`` the 60/40 does *worse* in the month *after* a positive-correlation
    reading (the claim: positive corr flags a broken hedge). ``regime_edge = 0`` is the null (the
    forward return is independent of the correlation sign). The returned monthly frame carries the
    columns the strategy consumes: equity return, bond return, the trailing-corr sign, and the
    realised forward 60/40 return — the same schema ``fetch_prices`` is reduced to.

    Returns ``(monthly, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Latent regime: a two-state Markov chain, sticky, mostly in the negative-corr state.
    rho_neg, rho_pos = -0.4, 0.5
    p_stay = 0.94
    state = 0  # 0 = negative-corr regime, 1 = positive-corr regime
    rhos = np.empty(n_months)
    for i in range(n_months):
        if rng.random() > p_stay:
            state = 1 - state
        rhos[i] = rho_pos if state == 1 else rho_neg

    eq_mu, eq_sd = 0.008, 0.043       # monthly equity drift/vol (~10%/yr, ~15% vol)
    bd_mu, bd_sd = 0.003, 0.030       # monthly bond drift/vol
    eq_ret = np.empty(n_months)
    bd_ret = np.empty(n_months)
    for i in range(n_months):
        rho = rhos[i]
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]])
        eq_ret[i] = eq_mu + eq_sd * z[0]
        bd_ret[i] = bd_mu + bd_sd * z[1]

    base_6040 = 0.6 * eq_ret + 0.4 * bd_ret

    # Trailing-corr SIGN measured from a rolling window of the *realised* returns (no look-ahead):
    corr_sign = _rolling_corr_sign(eq_ret, bd_ret, window=6)

    # Forward 60/40 return: shift base back by one month so fwd_t is the (t -> t+1) return, then
    # plant the penalty as a function of THIS month's corr sign.
    fwd_6040 = np.empty(n_months)
    fwd_6040[:-1] = base_6040[1:]
    fwd_6040[-1] = np.nan
    penalty = regime_edge * (corr_sign > 0).astype(float)
    fwd_6040 = fwd_6040 - penalty
    fwd_eq = np.empty(n_months)
    fwd_eq[:-1] = eq_ret[1:]
    fwd_eq[-1] = np.nan
    fwd_eq = fwd_eq - penalty  # same knob drives the equity leg too (so the control fires either way)

    idx = pd.period_range("2002-08", periods=n_months, freq="M").to_timestamp("M")
    monthly = pd.DataFrame(
        {
            "eq_ret": eq_ret,
            "bd_ret": bd_ret,
            "corr_sign": corr_sign,
            "fwd_6040": fwd_6040,
            "fwd_eq": fwd_eq,
            "rho_true": rhos,
        },
        index=idx,
    )
    monthly.index.name = "date"
    return monthly.iloc[:-1].copy(), WorldTruth(regime_edge=regime_edge)


def _rolling_corr_sign(a: np.ndarray, b: np.ndarray, window: int = 6) -> np.ndarray:
    """Sign (+1/-1/0) of the trailing ``window``-month Pearson correlation, no look-ahead.

    The correlation at index ``i`` uses returns ``[i-window+1 .. i]`` inclusive — fully public at
    the close of month ``i``. The first ``window-1`` entries are NaN-safe (set to 0 / no signal).
    """
    n = len(a)
    out = np.zeros(n)
    for i in range(n):
        lo = i - window + 1
        if lo < 0:
            out[i] = 0.0
            continue
        aa, bb = a[lo : i + 1], b[lo : i + 1]
        if aa.std() > 0 and bb.std() > 0:
            c = float(np.corrcoef(aa, bb)[0, 1])
            out[i] = np.sign(c)
        else:
            out[i] = 0.0
    return out


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "ebcf_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2002-07-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Daily adjusted-close prices for SPY + TLT, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = _retry(
        lambda: yf.download(
            TICKERS, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="any")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def monthly_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Reduce daily SPY/TLT prices to the monthly schema the strategy consumes.

    Returns a monthly frame with equity/bond returns, the trailing 6-month correlation sign, and
    forward (t -> t+1) 60/40 and equity returns — the identical columns ``synthetic_series``
    produces. Drops the final partial month (no complete forward return).
    """
    if prices.empty or EQUITY not in prices.columns or BOND not in prices.columns:
        return pd.DataFrame()
    px = prices[[EQUITY, BOND]].dropna()
    # month-end closes -> monthly simple returns
    m = px.resample("ME").last()
    ret = m.pct_change().dropna()
    eq = ret[EQUITY].to_numpy(dtype=float)
    bd = ret[BOND].to_numpy(dtype=float)
    base_6040 = 0.6 * eq + 0.4 * bd
    corr_sign = _rolling_corr_sign(eq, bd, window=6)

    n = len(ret)
    fwd_6040 = np.full(n, np.nan)
    fwd_6040[:-1] = base_6040[1:]
    fwd_eq = np.full(n, np.nan)
    fwd_eq[:-1] = eq[1:]

    out = pd.DataFrame(
        {
            "eq_ret": eq,
            "bd_ret": bd,
            "corr_sign": corr_sign,
            "fwd_6040": fwd_6040,
            "fwd_eq": fwd_eq,
            "rho_true": np.nan,
        },
        index=ret.index,
    )
    out.index.name = "date"
    # drop the last row (no forward return) — no partial-period leakage
    return out.iloc[:-1].copy()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
