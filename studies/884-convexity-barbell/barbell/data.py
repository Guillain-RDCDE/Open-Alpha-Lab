"""Data layer for Study 884 — Convexity Barbell.

The claim under test: a **duration-matched barbell** (a mix of a short-duration bond and a
long-duration bond, weighted so its duration equals a middle **bullet**'s) carries **more
convexity** than the bullet, so at *equal* duration it should out-earn the bullet whenever
yields move a lot — because the second-order term of the price/yield relation,
``+½·C·(Δy)²``, is bigger for the barbell (positive for both, larger for the more convex
book). This is the textbook "barbells are convex" structure (Fabozzi; Ilmanen).

We rebuild it from three liquid iShares Treasury ETFs plus a cash leg:

    bullet  = IEF  (7-10y Treasury)          -- the belly
    barbell = w·SHY (1-3y) + (1-w)·TLT (20y+) -- short + long ends, duration-matched to IEF
    cash    = BIL  (1-3m T-bills)             -- the risk-free leg for excess-vs-excess races

The barbell weight ``w`` is chosen data-drivenly each day so the barbell's **empirical
duration** (its beta to a common Treasury "rates" factor) equals the bullet's — the two
books then share the same first-order rate exposure, and any surviving difference is the
second-order (convexity) term net of the yield/carry the market charges for it.

Two tapes, one schema (a dict ``{ticker: DataFrame[Close]}`` of daily total-return closes):

* ``synthetic_panel`` — a deterministic, offline generator. Three bond assets driven by a
  common **fat-tailed** yield-change factor with a planted duration ladder and a planted
  convexity ladder (convexity ∝ duration², as for real bonds), plus a carry curve. The
  ``edge`` knob controls whether the barbell's convexity is **underpriced**: ``edge = 0``
  is the null (convexity exactly paid for by a carry give-up ⇒ zero net spread in
  expectation); ``edge > 0`` makes the convexity a genuine free pickup.
* ``fetch`` — the real yfinance tape (SHY, IEF, TLT, BIL daily closes, ``auto_adjust=True``
  total-return), cache-first into this study's own ``_cache/`` so the reproducible core
  never needs the network.

No look-ahead: the duration-match weight on day ``t`` is built from betas known at the
close of ``t-1`` (one ``shift``); the book is held on day ``t``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Three Treasury ETFs (short / belly / long) + a T-bill cash leg.
BOND_TICKERS = ["SHY", "IEF", "TLT"]
CASH_TICKER = "BIL"
TICKERS = BOND_TICKERS + [CASH_TICKER]

START = "2010-01-04"        # BIL trades from 2007-05; SHY/IEF/TLT from 2002; 2010 = clean common start
AS_OF = "2026-06-30"        # last complete calendar month at publication (drop the partial month)

CACHE_PATH = os.path.join(CACHE_DIR, "barbell_etf_closes.parquet")

__all__ = [
    "BOND_TICKERS", "CASH_TICKER", "TICKERS", "START", "AS_OF",
    "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_panel",
    "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily total-return closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None, retries: int = 4,
          sleep_s: float = 2.0) -> pd.DataFrame:
    """Download the four ETF total-return close series; cache the parquet.

    Retries up to ``retries`` times with a short sleep on a transient yfinance failure.
    ``auto_adjust=True`` gives dividend-reinvested (total-return) closes. The frame is a
    tz-naive daily DatetimeIndex with one column per ticker; cached under this study's own
    ``_cache/``.
    """
    import yfinance as yf  # lazy — never imported by the offline core

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no daily bars")
            closes = (
                raw["Close"].copy()
                if isinstance(raw.columns, pd.MultiIndex)
                else raw[["Close"]].copy()
            )
            closes = closes[[c for c in TICKERS if c in closes.columns]]
            if closes.shape[1] < len(TICKERS):
                raise RuntimeError(f"missing tickers: got {list(closes.columns)}")
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            closes = closes.dropna(how="any")
            if closes.empty:
                raise RuntimeError("no fully-overlapping rows across the four ETFs")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as e:  # noqa: BLE001 — retry any transient failure
            last_err = e
            if attempt < retries - 1:
                time.sleep(sleep_s)
    raise RuntimeError(f"yfinance fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_series(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily total-return closes as a DataFrame (index=date, columns=tickers),
    sliced to ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance."""
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call fetch() once to populate."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[[c for c in TICKERS if c in df.columns]].sort_index()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    df = df[(df.index >= lo) & (df.index <= hi)].dropna(how="any")
    return df


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Close]}``, sliced to ``[start, asof]``."""
    closes = load_series(start, asof)
    return {c: closes[[c]].rename(columns={c: "Close"}) for c in closes.columns}


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the close panel, for the as-of stamp."""
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted duration + convexity ladders (the positive control)
# --------------------------------------------------------------------------- #
# Plausible durations (years): SHY ~1.9, IEF ~7.5, TLT ~16.5. Convexity ∝ D² for bonds.
# The control deliberately EXAGGERATES convexity (k above the real ~1.0) and quiets the
# idiosyncratic noise so the planted convexity is cleanly recoverable — it is a machinery
# proof, disclosed as such, never cited in support of the real-tape stamp.
SYN_DUR = np.array([1.9, 7.5, 16.5])       # SHY, IEF, TLT
SYN_CONV_K = 3.0                           # convexity = SYN_CONV_K * duration**2 (exaggerated)


def synthetic_panel(
    edge: float = 0.0,
    seed: int = 884,
    n_days: int = 2200,
    start: str = "2010-01-04",
    yield_vol: float = 0.0007,             # daily stdev of the common yield-change factor
    tail_df: float = 3.5,                  # Student-t dof -> fat-tailed moves (convexity bites)
    idio_vol: float = 0.00015,
    base_carry_bps: float = 0.9,           # daily carry of the belly (IEF), in bps
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded three-bond panel with a planted convexity structure.

    A single latent, **fat-tailed** yield-change factor ``dy_t`` (Student-t, so big rate
    moves happen) drives all three bonds through the textbook price identity

        r[i,t] = carry_i  -  D_i · dy_t  +  ½ · C_i · dy_t²  +  idio_i,t

    with a monotone duration ladder :data:`SYN_DUR` (1.9 → 16.5) and convexity
    ``C_i = SYN_CONV_K · D_i²`` (convex, and rising with maturity — the real bond shape).

    The **carry curve** is set so that, at ``edge = 0``, the barbell's expected convexity
    pickup is exactly cancelled by a carry give-up: the barbell (short+long ends, duration
    matched to the belly) yields *less* than the bullet by precisely
    ``½·(C_barbell − C_bullet)·Var(dy)`` per day, so the expected net spread is **zero**
    (convexity fairly priced — the null). ``edge > 0`` adds ``edge`` bps/day to the
    barbell's carry on top, i.e. the convexity is **underpriced** and the barbell is a
    genuine free pickup (the positive control fires). Business-day index, well below the
    pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    D = SYN_DUR
    C = SYN_CONV_K * D ** 2

    # Fat-tailed yield-change factor, scaled to the target daily stdev.
    t = rng.standard_t(tail_df, n_days)
    t *= yield_vol / t.std()
    var_dy = float(np.var(t))

    # Duration-match weight on the short leg so barbell duration == bullet (IEF) duration.
    w_short = (D[2] - D[1]) / (D[2] - D[0])          # SHY weight; TLT weight = 1 - w_short
    C_barbell = w_short * C[0] + (1.0 - w_short) * C[2]
    conv_gain = 0.5 * (C_barbell - C[1]) * var_dy    # per-day convexity pickup of barbell

    # Carry curve (bps/day). Belly = base; ends chosen so the DURATION-MATCHED barbell's
    # carry = belly carry - conv_gain + edge  (edge in bps/day). Anchor the short leg to a
    # low bill-like carry and solve the long leg to hit the barbell carry target.
    base = base_carry_bps / 1e4
    edge_d = edge / 1e4
    carry_short = 0.35 * base                          # short end yields well below the belly
    barbell_carry_target = base - conv_gain + edge_d
    carry_long = (barbell_carry_target - w_short * carry_short) / (1.0 - w_short)
    carry = np.array([carry_short, base, carry_long])  # SHY, IEF, TLT

    panel: dict[str, pd.DataFrame] = {}
    for i, tkr in enumerate(BOND_TICKERS):
        idio = rng.normal(0.0, idio_vol, n_days)
        r = carry[i] - D[i] * t + 0.5 * C[i] * t ** 2 + idio
        close = 100.0 * np.cumprod(1.0 + r)
        panel[tkr] = pd.DataFrame({"Close": close}, index=idx)

    # A near-constant cash leg (T-bill): tiny positive daily carry, negligible vol.
    cash_r = np.full(n_days, 0.30 * base) + rng.normal(0.0, 1e-5, n_days)
    panel[CASH_TICKER] = pd.DataFrame(
        {"Close": 100.0 * np.cumprod(1.0 + cash_r)}, index=idx
    )
    return panel
