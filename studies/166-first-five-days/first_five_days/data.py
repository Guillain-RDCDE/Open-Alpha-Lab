"""Data layer for Study 166 (First-Five-Days).

Two tapes, one shape (an annual return series for the S&P 500 index):

- ``synthetic_annual`` — a *deterministic, offline* generator.  A ``signal_strength``
  knob controls whether the first-five-days return genuinely predicts the full-year
  direction.  ``signal_strength = 0`` is the null — the first-five-days and the
  full-year return are independent — so tests can assert the hit-rate is near chance
  only when we plant an effect, and not otherwise.  This is the study's null in a bottle.

- ``fetch_gspc`` — real ^GSPC daily prices (``yfinance``), **cache-only** by default so
  the test-suite and reproducible core never touch the network.  Returns a frame with
  the first-five-days log-return and the full-year log-return for each year since 1950.

No look-ahead is baked in here — the first-five-days signal is formed on the close of
the fifth trading day of January; the full-year return uses year-end closes.  The
"first-five-days" and the full-year return share those five trading days' worth of
overlap (a small confound we note explicitly).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The first-five-days Early Warning System was popularised by Yale Hirsch in the
# Stock Trader's Almanac (original edition 1967/1968, codified as the 'first-five-days'
# rule in subsequent editions).  The rule applies to ^GSPC from 1950 onward — the same
# starting point as the broader Almanac seasonal system.
TICKER = "^GSPC"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 76,
    annual_vol: float = 0.16,
    annual_drift: float = 0.08,
    signal_strength: float = 0.0,
    f5d_vol_frac: float = 0.12,
    start_year: int = 1950,
    seed: int = 166,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual GSPC-like tape with a known first-five-days signal.

    The first-five-days (F5D) return is drawn from
    ``N(drift * f5d_vol_frac, (annual_vol * f5d_vol_frac)^2)``.  The full-year return is
    drawn as::

        r_year = drift + eps
        eps ~ N(0, annual_vol^2)

    and then a ``signal_strength`` knob *tilts* the full-year mean by
    ``+signal_strength * annual_vol`` when F5D was positive, and
    ``-signal_strength * annual_vol`` when it was negative.

    ``signal_strength = 0`` is the null (independence); a positive value plants the
    First-Five-Days effect.  Returns ``(df, truth)`` where ``df`` has columns
    ``year, ret_5d, ret_year``.
    """
    rng = np.random.default_rng(seed)
    f5d_sd = annual_vol * f5d_vol_frac
    f5d_mu = annual_drift * f5d_vol_frac

    years = np.arange(start_year, start_year + n_years)
    ret_5d = rng.normal(f5d_mu, f5d_sd, n_years)
    f5d_sign = np.sign(ret_5d)

    # Base full-year return, then tilt by signal_strength times the annual vol.
    ret_year_base = rng.normal(annual_drift, annual_vol, n_years)
    ret_year = ret_year_base + signal_strength * annual_vol * f5d_sign

    df = pd.DataFrame(
        {"year": years, "ret_5d": ret_5d, "ret_year": ret_year}
    ).set_index("year")

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "signal_strength": signal_strength,
        "f5d_vol_frac": f5d_vol_frac,
        "start_year": start_year,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "first_five_days_annual.parquet")


def _cache_path_daily(cache_dir: str) -> str:
    return os.path.join(cache_dir, "^GSPC_split_only.parquet")


def _build_annual(daily: pd.DataFrame) -> pd.DataFrame:
    """Convert a daily price frame to annual first-five-days + full-year returns.

    First-five-days return: log(close on 5th trading day of Jan / close on last trading
    day of December prior).

    Full-year return: log(close on last trading day of December / close on last trading
    day of December prior).

    Both are log-returns.  No look-ahead: the F5D signal is formed on the 5th trading
    day of January; the full-year outcome is known only at December year-end.

    Note: the F5D return is a component of the full-year return (the first five trading
    days overlap), so a raw correlation is expected even under the null.  We measure
    whether the *sign* of F5D predicts the *sign* of the full year, which is a directional
    prediction test that goes beyond the mechanical overlap.
    """
    close = daily["close"].sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    rows = []
    for y in range(1950, close.index.year.max() + 1):
        try:
            # Last trading day of prior December
            dec_prev = close.loc[f"{y - 1}-12"]
            if len(dec_prev) == 0:
                continue
            p_dec_prev = dec_prev.iloc[-1]

            # Fifth trading day of January (0-indexed: iloc[4])
            jan = close.loc[f"{y}-01"]
            if len(jan) < 5:
                continue
            p_jan5 = jan.iloc[4]
            ret_5d = float(np.log(p_jan5 / p_dec_prev))

            # Last trading day of December (year-end close)
            dec_cur = close.loc[f"{y}-12"]
            if len(dec_cur) == 0:
                continue
            p_dec_cur = dec_cur.iloc[-1]
            ret_year = float(np.log(p_dec_cur / p_dec_prev))

            rows.append({"year": y, "ret_5d": ret_5d, "ret_year": ret_year})
        except KeyError:
            # Year is incomplete (e.g. current year without December yet) — skip.
            continue

    if not rows:
        return pd.DataFrame(columns=["ret_5d", "ret_year"])
    out = pd.DataFrame(rows).set_index("year")
    return out


def fetch_gspc(
    start: str = "1949-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Annual first-five-days + full-year returns for ^GSPC; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  Returns a DataFrame with columns ``ret_5d``
    (log-return for first five trading days of January) and ``ret_year`` (full-year
    log-return, Jan-thru-Dec from the prior year-end), indexed by year.

    The effective sample is 1950-onward (consistent with the Stock Trader's Almanac
    claim) — approximately 76 years.  We are explicit that n ≈ 76 is tiny: a large
    t-stat on such a small sample does not reach the desk's inference bar for REAL.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            # Fall back to pre-existing shared GSPC cache (avoids a redundant download).
            shared_path = _cache_path_daily(cache_dir)
            if os.path.exists(shared_path):
                raw = pd.read_parquet(shared_path)
                # Normalise to a flat column set with a lowercase 'close' column.
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(-1)
                raw.columns = [str(c).lower() for c in raw.columns]
                if "close" not in raw.columns:
                    raise RuntimeError(
                        f"Cannot find 'close' column in {shared_path}; "
                        f"columns: {list(raw.columns)}"
                    )
                return _build_annual(raw)
            raise FileNotFoundError(
                f"No cached annual F5D series at {path} and no shared GSPC daily cache at "
                f"{shared_path}.  Call fetch_gspc(fetch=True) once to populate the cache."
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
    annual = _build_annual(daily)
    annual.to_parquet(path)
    return annual


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (ret_5d column), for the as-of stamp."""
    arr = df["ret_5d"].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
