"""Data layer for Study 80 (Cold-Open).

Two tapes, one shape (an annual return series for the S&P 500 index):

- ``synthetic_annual`` — a *deterministic, offline* generator.  A ``signal_strength``
  knob controls whether January's sign genuinely predicts the rest-of-year direction.
  ``signal_strength = 0`` is the null — January sign and Feb-Dec return are independent
  — so tests can assert the hit-rate is near chance (50 %) only when we plant an effect,
  and not otherwise.  This is the study's null in a bottle.
- ``fetch_gspc`` — real ^GSPC daily prices (``yfinance``), **cache-only** by default so
  the test-suite and the reproducible core never touch the network.  The study requires
  annual January and Feb-Dec returns, which are derived from daily closes and then cached
  as a parquet under ``_cache/``.

No look-ahead is baked in here — signals are formed on the January close (end-of-month),
positions entered on 1 February.  The rest-of-year (Feb-Dec) return is the outcome.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ^GSPC daily history back to 1950 is what we want; the claim is "as goes January,
# so goes the year", coined by Stock Traders Almanac (Hirsch 1972) citing the index
# from 1950 onward.
TICKER = "^GSPC"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 75,
    annual_vol: float = 0.16,
    annual_drift: float = 0.08,
    signal_strength: float = 0.0,
    jan_vol_frac: float = 0.35,
    start_year: int = 1950,
    seed: int = 80,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual GSPC-like tape with a known January-Barometer signal.

    January return is drawn from N(drift * jan_vol_frac, (annual_vol * jan_vol_frac) ** 2).
    The Feb-Dec return is drawn as::

        r_fbd = drift * (1 - jan_vol_frac) + eps
        eps ~ N(0, ((1 - jan_vol_frac) * annual_vol)^2)

    and then a ``signal_strength`` knob *tilts* the Feb-Dec mean by
    ``+signal_strength * annual_vol`` when January was positive, and
    ``-signal_strength * annual_vol`` when it was negative.

    ``signal_strength = 0`` is the null (independence); a positive value plants
    the January Barometer effect.  Returns ``(df, truth)`` where ``df`` has
    columns ``year, jan_ret, fbd_ret`` (Feb-Dec total return).
    """
    rng = np.random.default_rng(seed)
    jan_sd = annual_vol * jan_vol_frac
    fbd_sd = annual_vol * (1.0 - jan_vol_frac)
    jan_mu = annual_drift * jan_vol_frac
    fbd_mu = annual_drift * (1.0 - jan_vol_frac)

    years = np.arange(start_year, start_year + n_years)
    jan_ret = rng.normal(jan_mu, jan_sd, n_years)
    jan_sign = np.sign(jan_ret)  # +1 or -1
    # Base Feb-Dec return, then tilt by signal_strength times the annual vol.
    fbd_base = rng.normal(fbd_mu, fbd_sd, n_years)
    fbd_ret = fbd_base + signal_strength * annual_vol * jan_sign

    df = pd.DataFrame(
        {"year": years, "jan_ret": jan_ret, "fbd_ret": fbd_ret}
    ).set_index("year")

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "signal_strength": signal_strength,
        "jan_vol_frac": jan_vol_frac,
        "start_year": start_year,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "gspc_annual.parquet")


def _cache_path_daily(cache_dir: str) -> str:
    return os.path.join(cache_dir, "gspc_daily.parquet")


def _build_annual(daily: pd.DataFrame) -> pd.DataFrame:
    """Convert a daily price frame to annual January + Feb-Dec returns.

    January return: close on the last trading day of January vs close on the last
    trading day of December prior (i.e. the year-end before).  Feb-Dec return:
    year-end close vs the last-January close.  Both are log-returns — consistent
    with how portfolio maths compounds and how the original studies are framed.

    No look-ahead: the Jan signal is formed entirely within January; the Feb-Dec
    outcome begins on 1 February.
    """
    close = daily["close"].sort_index()
    # Resample to month-end closes.
    monthly = close.resample("ME").last()
    monthly.index = pd.PeriodIndex(monthly.index.to_period("M"))

    rows = []
    for y in range(monthly.index.year.min() + 1, monthly.index.year.max() + 1):
        dec_prev = pd.Period(f"{y - 1}-12", freq="M")
        jan_cur = pd.Period(f"{y}-01", freq="M")
        dec_cur = pd.Period(f"{y}-12", freq="M")
        if dec_prev not in monthly.index or jan_cur not in monthly.index:
            continue
        if dec_cur not in monthly.index:
            continue
        p_dec_prev = monthly[dec_prev]
        p_jan_cur = monthly[jan_cur]
        p_dec_cur = monthly[dec_cur]
        jan_ret = np.log(p_jan_cur / p_dec_prev)
        fbd_ret = np.log(p_dec_cur / p_jan_cur)
        rows.append({"year": y, "jan_ret": jan_ret, "fbd_ret": fbd_ret})

    if not rows:
        return pd.DataFrame(columns=["jan_ret", "fbd_ret"])
    out = pd.DataFrame(rows).set_index("year")
    return out


def fetch_gspc(
    start: str = "1950-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Annual January + Feb-Dec returns for ^GSPC; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  Returns a DataFrame with columns ``jan_ret``
    (January log-return) and ``fbd_ret`` (Feb-Dec log-return), indexed by year.

    The effective sample is 1950-onward (as in the original Hirsch 1972 / Stock Traders
    Almanac claim) — roughly 75 Januaries, giving a study with **tiny effective n**.  We
    say so loudly.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached annual GSPC series at {path}.  "
                f"Call fetch_gspc(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(path)

    # Lazy import: only when we actually go to the network.
    import yfinance as yf

    raw = yf.download(
        TICKER,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKER}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    daily = raw.rename(columns=str.lower)[["close"]]
    daily.index.name = "date"
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    os.makedirs(cache_dir, exist_ok=True)
    # Cache the daily frame too (for verification).
    daily_path = _cache_path_daily(cache_dir)
    daily.to_parquet(daily_path)

    annual = _build_annual(daily)
    annual.to_parquet(path)
    return annual


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (jan_ret column), for the as-of stamp."""
    arr = df["jan_ret"].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
