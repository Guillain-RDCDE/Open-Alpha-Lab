"""Data layer for Study 317 (Fed-Balance-Sheet).

The slogan under test is **"Don't fight the Fed"**: when the Federal Reserve is
*expanding* its balance sheet (quantitative easing, QE) stocks supposedly rip, and
when it is *shrinking* it (quantitative tightening, QT) stocks supposedly struggle.
The honest version of the test sorts daily SPY returns by which balance-sheet regime
the Fed was in on that day.

The ideal regime variable is the weekly Fed balance sheet (FRED series ``WALCL``,
*Total Assets of the Federal Reserve*). That series is **network-blocked** in this
environment, so we encode the regime *itself* — not the dollar level — as a small,
auditable, hand-built table of (start, end, regime) windows transcribed from the Fed's
own announced programmes (QE1/2/3, the 2017–19 runoff, COVID QE, the 2022→ QT). This is
a deliberate, stated simplification: we test the **announced direction** of the balance
sheet, which is exactly the information a "don't fight the Fed" trader claims to act on.

Two tapes, one shape (a tz-naive daily price frame with a ``regime`` label):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``qe_edge`` knob plants
  the only thing a QE/QT timing rule can harvest: a difference in mean drift between the
  QE days and the QT days. ``qe_edge = 0`` is the null (regime carries no information);
  ``qe_edge > 0`` is the positive control. The harness must bank the planted edge and
  find nothing when nothing is planted.
- ``load_real`` — the real SPY daily tape, **cache-first**: it tries the shared
  ``quantlab.data`` parquet cache, then this study's own ``_cache/``, and only touches
  the network on an explicit ``fetch=True``. Each day is then labelled with its regime
  from the hard-coded table. Graceful fallback: if no cache and no network, it raises a
  clear ``FileNotFoundError`` so callers can drop to the synthetic tape.

No look-ahead is baked in here — the regime label of day *t* is the regime in force on
day *t* (the programme direction is public, calendar-known information). The one
execution lag for a *trading* rule lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# The hard-coded QE / QT regime table (announced-direction proxy for WALCL)
# ---------------------------------------------------------------------------
# Each row: (start_inclusive, end_exclusive, regime). Regimes:
#   "QE"   — Fed balance sheet announced *expanding* (asset purchases)
#   "QT"   — Fed balance sheet announced *shrinking* (runoff / sales)
#   "FLAT" — neither (pause / steady reinvestment)
# Dates transcribed from the Fed's announced programmes; see docs/references.md.
# This is the *direction*, not the dollar level (WALCL is network-blocked here).
REGIME_TABLE: list[tuple[str, str, str]] = [
    ("1993-01-01", "2008-11-25", "FLAT"),  # pre-QE: conventional policy
    ("2008-11-25", "2010-03-31", "QE"),    # QE1 (MBS/agency + Treasuries)
    ("2010-03-31", "2010-11-03", "FLAT"),  # gap between QE1 and QE2
    ("2010-11-03", "2011-06-30", "QE"),    # QE2
    ("2011-06-30", "2012-09-13", "FLAT"),  # Operation Twist era (size ~flat)
    ("2012-09-13", "2014-10-29", "QE"),    # QE3 (open-ended purchases)
    ("2014-10-29", "2017-10-01", "FLAT"),  # QE3 ends; full reinvestment
    ("2017-10-01", "2019-08-01", "QT"),    # the 2017–19 balance-sheet runoff
    ("2019-08-01", "2020-03-15", "FLAT"),  # runoff halted; organic growth
    ("2020-03-15", "2022-03-09", "QE"),    # COVID QE (huge expansion)
    ("2022-03-09", "2022-06-01", "FLAT"),  # taper to the QT start
    ("2022-06-01", "2099-01-01", "QT"),    # 2022→ QT (runoff)
]


def label_regime(index: pd.DatetimeIndex) -> pd.Series:
    """Label each date in ``index`` with its QE/QT/FLAT regime from REGIME_TABLE.

    The label is calendar-known (the programme direction is announced), so there is
    no look-ahead in attaching the regime of day *t* to day *t*.
    """
    out = pd.Series("FLAT", index=index, dtype=object)
    for start, end, regime in REGIME_TABLE:
        mask = (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))
        out[mask] = regime
    out.name = "regime"
    return out


def regime_windows() -> pd.DataFrame:
    """The regime table as a tidy DataFrame (for display / auditing)."""
    rows = [{"start": pd.Timestamp(s), "end": pd.Timestamp(e), "regime": r}
            for s, e, r in REGIME_TABLE]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (control + null)
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6000,
    qe_edge: float = 0.0,
    base_drift: float = 0.0003,
    daily_vol: float = 0.011,
    start: str = "1998-01-02",
    seed: int = 317,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily SPY-like tape whose drift depends on the Fed regime.

    The session calendar is built with ``bdate_range`` and labelled by the *same*
    REGIME_TABLE used on the real tape, so the synthetic QE/QT split is realistic in
    shape. Daily log-return on day *t*::

        r_t = base_drift + qe_edge * s(regime_t) + eps_t,   eps_t ~ N(0, daily_vol)

    where ``s(QE) = +1``, ``s(QT) = -1``, ``s(FLAT) = 0``. So:

    - ``qe_edge = 0``  → the **null**: every regime has the same expected return; a
      QE/QT timing rule must be a fair coin here.
    - ``qe_edge > 0``  → the **positive control**: QE days really do drift up and QT
      days really do drift down, by ``qe_edge`` per day each way — the exact structure
      "don't fight the Fed" assumes.

    Returns ``(frame, truth)``; ``frame`` has columns ``close`` and ``regime``.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    regime = label_regime(sessions)

    sign = regime.map({"QE": 1.0, "QT": -1.0, "FLAT": 0.0}).to_numpy(dtype=float)
    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = base_drift + qe_edge * sign + eps
    close = 100.0 * np.exp(np.cumsum(log_ret))

    frame = pd.DataFrame(
        {"close": close, "regime": regime.to_numpy()},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "qe_edge": qe_edge,
        "base_drift": base_drift,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
        "n_qe": int((sign > 0).sum()),
        "n_qt": int((sign < 0).sum()),
        "n_flat": int((sign == 0).sum()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — SPY daily, cache-first (shared quantlab cache → study cache → net)
# ---------------------------------------------------------------------------
def _study_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def _shared_cache_candidates(ticker: str, mode: str = "split_only") -> list[str]:
    """Possible locations of the shared quantlab parquet for ``ticker``.

    ``quantlab.data.CACHE_DIR`` is a *relative* ``_cache`` path (resolved against the
    caller's cwd), so a notebook running from ``notebooks/`` would miss it. We therefore
    probe both the cwd-relative location and the repo-root ``_cache`` absolutely.
    """
    fname = f"{ticker}_{mode}.parquet"
    cands = []
    try:
        from quantlab import data as qdata
        cands.append(str(qdata.CACHE_DIR / fname))
    except Exception:
        pass
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    cands.append(os.path.join(repo_root, "_cache", fname))
    return cands


def _try_quantlab_cache(ticker: str, mode: str = "split_only") -> pd.DataFrame | None:
    """Read the shared ``quantlab/_cache`` parquet for ``ticker`` if it exists.

    Cache-only: never goes to the network here. Returns a frame with a ``close``
    column, or ``None`` if the shared cache is absent / unreadable.
    """
    cache_path = next((p for p in _shared_cache_candidates(ticker, mode)
                       if os.path.exists(p)), None)
    if cache_path is None:
        return None
    try:
        df = pd.read_parquet(cache_path)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    if "close" not in cols:
        return None
    out = df.rename(columns={cols["close"]: "close"})[["close"]].copy()
    return out


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily SPY close, labelled by Fed regime. **Cache-first**, offline-safe.

    Resolution order (network only on an explicit ``fetch=True``):

    1. the shared ``quantlab`` parquet cache (``SPY_split_only.parquet``);
    2. this study's own ``_cache/bars_SPY_daily.parquet``;
    3. if ``fetch=True``, download via ``quantlab.data.fetch`` and write the study cache;
    4. otherwise raise ``FileNotFoundError`` (callers fall back to the synthetic tape).

    The returned frame has columns ``close`` and ``regime`` (price-only / split-only —
    this is **not** a total-return series, and is labelled as such everywhere).
    """
    study_path = _study_cache_path(ticker, cache_dir)
    bars = _try_quantlab_cache(ticker)

    if bars is None and os.path.exists(study_path):
        bars = pd.read_parquet(study_path)

    if bars is None and fetch:
        from quantlab import data as qdata  # lazy: only on a real cache miss
        raw = qdata.fetch(ticker, start=start, mode="split_only", use_cache=True)
        bars = raw.rename(columns={c: c.lower() for c in raw.columns})[["close"]].copy()
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(study_path)

    if bars is None:
        raise FileNotFoundError(
            f"No cached SPY tape for {ticker} (shared quantlab cache or {study_path}). "
            f"Call load_real(fetch=True) once online to populate the cache, or use "
            f"synthetic_daily() for the offline core."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars = bars.loc[bars.index >= pd.Timestamp(start)].copy()
    bars.index.name = "date"
    bars["regime"] = label_regime(bars.index)
    return bars[["close", "regime"]]


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
