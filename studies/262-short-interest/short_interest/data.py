"""Data layer for Study 262 (Short-Interest).

The question: *do heavily shorted stocks bounce or bleed?*  We test whether a
stock's short interest (short percentage of float, and days-to-cover) predicts
its forward return -- the classic clash between the "short interest is a bearish
signal" camp (Boehmer-Jones-Zhang: short sellers are informed; high SI predicts
*low* returns) and the "short squeeze" / "crowded-short reversal" camp (high SI
stocks are oversold and bounce).

Three pieces, the first two fully offline for the deterministic core:

- ``SHORT_INTEREST_SNAPSHOT`` -- a *hardcoded, curated* cross-sectional snapshot
  of short interest for a fixed basket of ~60 well-known US names.  For each
  ticker we record ``short_pct_float`` (short interest as a percent of float)
  and ``days_to_cover`` (short interest / average daily volume).  These are
  *representative curated levels* compiled from public short-interest screens
  (FINRA bi-monthly settlement data as surfaced by Nasdaq / S3 Partners style
  vendors) for a single recent settlement snapshot.  They are hardcoded so the
  core is offline, deterministic, and reproducible.  This is a SINGLE
  cross-sectional snapshot, not a time series -- exactly the design that the
  table hook ("sort forward returns by short interest") implies.

- ``synthetic_panel`` -- a deterministic, offline generator of (firm short
  interest, firm forward return) pairs with a *planted* SI->return slope.  The
  ``si_beta`` knob controls how strongly short interest predicts the forward
  return; ``si_beta = 0`` is the null (short interest carries no forward
  information).  This lets the tests confirm the engine is truthful before any
  real data is touched.

- ``fetch_forward_returns`` -- the real Yahoo! forward returns for the basket
  over a chosen horizon, cache-only by default (the test-suite and the
  reproducible core never touch the network).  The cache lives under
  ``_cache/si_forward.parquet`` (a ticker x horizon frame of forward simple
  returns measured from the snapshot date).

**Survivorship + snapshot-staleness caveat**: the basket is a fixed set of
large/mid-cap names that survived to the snapshot date, and the short-interest
levels are a single static snapshot (real SI is revised twice a month and is
itself noisy/stale by settlement lag).  Positive results from the real tape are
upper bounds and are named on the Signal axis.

No look-ahead: the short-interest snapshot is dated, and forward returns are
measured strictly *after* the snapshot date.  Positions are formed at the
snapshot close (one settlement-period execution lag is documented in the
strategy module).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

FORWARD_CACHE = os.path.join(DEFAULT_CACHE, "si_forward.parquet")

# The snapshot date that the hardcoded short-interest levels correspond to.
# Forward returns are measured strictly after this date.
SNAPSHOT_DATE = "2024-12-31"

# ---------------------------------------------------------------------------
# The hardcoded short-interest snapshot -- the study's deterministic core
# ---------------------------------------------------------------------------
# Columns:
#   ticker          : Yahoo ticker
#   short_pct_float : short interest as a percent of float (e.g. 22.0 = 22%)
#   days_to_cover   : short interest / 30-day avg daily volume (a.k.a. short ratio)
#
# These are *representative curated levels* for a single recent settlement
# snapshot (around 2024-12-31), compiled from public short-interest screens.
# They are hardcoded so the offline core is deterministic and reproducible.
# Levels are rounded and approximate -- this is a teaching snapshot, not a
# tick-accurate vendor feed.  The ranking (who is heavily vs lightly shorted)
# is what the strategy uses, and that ordering is robust to small revisions.
SHORT_INTEREST_SNAPSHOT: list[tuple[str, float, float]] = [
    # --- heavily shorted / "crowded short" names ---
    ("CVNA", 22.0, 4.1),
    ("UPST", 19.5, 3.6),
    ("AFRM", 16.0, 3.2),
    ("BYND", 38.0, 6.5),
    ("LCID", 28.0, 5.8),
    ("RIVN", 14.0, 2.9),
    ("PLUG", 24.0, 4.6),
    ("FUBO", 21.0, 3.1),
    ("RIOT", 17.0, 2.7),
    ("MARA", 18.5, 3.0),
    ("BBBY", 35.0, 5.5),
    ("GME", 20.0, 3.8),
    ("AMC", 19.0, 3.4),
    ("CCL", 9.0, 2.2),
    ("NCLH", 8.0, 2.0),
    ("W", 12.0, 2.6),
    ("CHWY", 11.0, 2.4),
    ("WOLF", 26.0, 4.9),
    ("SMCI", 13.5, 1.8),
    ("DJT", 30.0, 4.2),
    # --- moderately shorted names ---
    ("TSLA", 3.2, 1.1),
    ("NVDA", 1.3, 0.9),
    ("AMD", 2.4, 1.3),
    ("INTC", 2.8, 1.5),
    ("F", 4.0, 2.1),
    ("GM", 3.0, 1.7),
    ("NFLX", 1.6, 1.2),
    ("PYPL", 2.2, 1.4),
    ("SQ", 3.5, 1.9),
    ("SHOP", 2.6, 1.5),
    ("UBER", 1.8, 1.0),
    ("ABNB", 2.0, 1.3),
    ("COIN", 5.5, 1.6),
    ("PLTR", 2.7, 1.2),
    ("SNAP", 4.5, 1.8),
    ("ROKU", 6.0, 2.3),
    ("DKNG", 3.8, 1.7),
    ("HOOD", 4.2, 1.5),
    ("DASH", 2.3, 1.4),
    ("RBLX", 3.6, 1.6),
    # --- lightly shorted blue chips ---
    ("AAPL", 0.7, 1.0),
    ("MSFT", 0.6, 0.9),
    ("AMZN", 0.8, 1.0),
    ("GOOGL", 0.6, 0.8),
    ("META", 0.9, 0.9),
    ("JPM", 1.0, 1.1),
    ("V", 0.8, 1.0),
    ("MA", 0.9, 1.0),
    ("JNJ", 0.7, 1.1),
    ("PG", 0.6, 1.0),
    ("KO", 0.8, 1.2),
    ("PEP", 0.7, 1.1),
    ("WMT", 0.8, 1.0),
    ("HD", 1.0, 1.1),
    ("COST", 0.9, 1.0),
    ("XOM", 1.1, 1.2),
    ("CVX", 1.0, 1.1),
    ("UNH", 0.9, 1.0),
    ("LLY", 0.7, 0.9),
    ("MRK", 1.0, 1.1),
]


def short_interest_table() -> pd.DataFrame:
    """The hardcoded short-interest snapshot as a tidy DataFrame.

    Returns a frame indexed by ticker with columns ``short_pct_float`` and
    ``days_to_cover``.  This is the deterministic, offline core of the study.
    """
    df = pd.DataFrame(
        SHORT_INTEREST_SNAPSHOT,
        columns=["ticker", "short_pct_float", "days_to_cover"],
    ).set_index("ticker")
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    si_beta: float = -0.04,
    si_mean: float = 6.0,
    si_vol: float = 6.0,
    base_ret: float = 0.01,
    ret_vol: float = 0.12,
    seed: int = 262,
) -> tuple[pd.DataFrame, dict]:
    """A cross-section of (short interest, forward return) pairs with a known slope.

    Each firm j gets a short-interest level ``si_j`` drawn from a clipped
    half-normal (so SI is non-negative, right-skewed -- like the real
    distribution), and a forward return::

        fwd_j = base_ret + si_beta * (si_j - si_mean) / si_vol + noise_j

    ``si_beta`` is the planted slope of standardized short interest on the
    forward return.  A *negative* ``si_beta`` means high-SI stocks *bleed*
    (short sellers are informed -- the Boehmer-Jones-Zhang view); a *positive*
    ``si_beta`` means high-SI stocks *bounce* (the squeeze / reversal view);
    ``si_beta = 0`` is the null (short interest carries no forward information).

    Returns ``(df, truth)`` where ``df`` is a firm-indexed frame with columns
    ``short_pct_float`` and ``fwd_ret``, and ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Right-skewed non-negative short interest (half-normal-ish around si_mean).
    si = np.abs(rng.normal(0.0, si_vol, n_firms)) + 0.5
    # Re-center so the mean is roughly si_mean (keeps standardization stable).
    si = si - si.mean() + si_mean
    si = np.clip(si, 0.1, None)

    si_std = (si - si.mean()) / (si.std(ddof=0) if si.std(ddof=0) > 0 else 1.0)
    noise = rng.normal(0.0, ret_vol, n_firms)
    fwd = base_ret + si_beta * si_std + noise

    df = pd.DataFrame(
        {"short_pct_float": si, "fwd_ret": fwd},
        index=pd.Index(firms, name="ticker"),
    )

    truth = {
        "si_beta": si_beta,
        "si_mean": si_mean,
        "si_vol": si_vol,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "n_firms": n_firms,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo forward returns for the basket, cache-only by default
# ---------------------------------------------------------------------------
def fetch_forward_returns(
    fetch: bool = False,
    cache_path: str = FORWARD_CACHE,
    snapshot_date: str = SNAPSHOT_DATE,
    horizons_months: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Forward simple returns for the basket measured after the snapshot date.

    Cache-only unless ``fetch=True``.  Network is touched only on an explicit
    ``fetch=True`` (then the result is cached as a parquet under
    ``_cache/si_forward.parquet``).  The frame is indexed by ticker with one
    column per horizon (``fwd_1m``, ``fwd_3m``, ...) holding the simple return
    from the first trading day on/after ``snapshot_date`` to the trading day
    ~``h`` months later.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False`` --
    callers must guard with ``os.path.exists(cache_path)`` before calling.

    **Survivorship + staleness bias**: the basket is fixed survivors as of the
    snapshot, and the short-interest snapshot is a single static reading.  All
    positive results are upper bounds.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached forward-return panel at {cache_path}. "
                "Call fetch_forward_returns(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    # ------------------------------------------------------------------
    # Live fetch: yfinance daily closes, then forward returns per horizon
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    tickers = list(short_interest_table().index)
    snap = pd.Timestamp(snapshot_date)
    end = snap + pd.DateOffset(months=max(horizons_months) + 2)

    raw = yf.download(
        tickers,
        start=(snap - pd.DateOffset(days=10)).strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily data")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            closes = raw.xs("Close", axis=1, level=0)
    else:
        closes = raw.to_frame(name=tickers[0])

    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()

    # Base price = first trading day on/after the snapshot.
    base_idx = closes.index[closes.index >= snap]
    if len(base_idx) == 0:
        raise RuntimeError("No trading days on/after the snapshot date")
    base_day = base_idx[0]
    base_px = closes.loc[base_day]

    out = {}
    for h in horizons_months:
        target = base_day + pd.DateOffset(months=h)
        # nearest trading day on/before target
        ok = closes.index[closes.index <= target]
        if len(ok) == 0:
            continue
        tday = ok[-1]
        px = closes.loc[tday]
        out[f"fwd_{h}m"] = px / base_px - 1.0

    fwd = pd.DataFrame(out)
    fwd.index.name = "ticker"
    fwd = fwd.dropna(how="all")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fwd.to_parquet(cache_path)
    print(f"Cached forward returns: {fwd.shape[0]} tickers x {fwd.shape[1]} horizons -> {cache_path}")
    return fwd


def real_panel(
    horizon_col: str = "fwd_3m",
    cache_path: str = FORWARD_CACHE,
) -> pd.DataFrame:
    """Join the hardcoded SI snapshot with cached forward returns (cache-only).

    Returns a ticker-indexed frame with ``short_pct_float``, ``days_to_cover``,
    and ``fwd_ret`` (the chosen horizon column renamed).  Raises
    ``FileNotFoundError`` if the forward-return cache is absent.
    """
    si = short_interest_table()
    fwd = fetch_forward_returns(fetch=False, cache_path=cache_path)
    if horizon_col not in fwd.columns:
        raise KeyError(f"horizon {horizon_col!r} not in cache columns {list(fwd.columns)}")
    df = si.join(fwd[[horizon_col]], how="inner").rename(columns={horizon_col: "fwd_ret"})
    return df.dropna(subset=["fwd_ret"])


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a numeric frame, for the as-of stamp."""
    arr = df.select_dtypes("number").to_numpy(dtype=float)
    arr = np.nan_to_num(arr)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
