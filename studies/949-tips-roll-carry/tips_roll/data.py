"""Data layer for Study 949 — Riding the TIPS Curve.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the four TIPS maturity buckets (VTIP, SCHP,
  TIP, LTPZ), the nominal Treasury funds used as duration matches (SHY, IEI, IEF, TLT)
  and the cash leg (BIL, the 1-3 month T-bill ETF). ``fetch`` touches the network and
  caches parquet under the shared ``studies/_cache`` (retry up to 4x); ``load_prices``
  reads that cache **offline** and never imports yfinance. The whole test-suite runs
  with NO cache present (synthetic only), so CI is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  common duration factor drives a nominal fund; the inflation-linked fund loads on the
  same factor with a beta plus a **planted residual carry** whose size is set by the
  ``signal_strength`` knob. At ``signal_strength=0`` there is no residual carry at all
  (the null: the detector must stay quiet); at ``signal_strength=1`` there is a real
  one the duration-hedged regression should recover. Seeds are fixed.

The claim under test: extending along the *real* yield curve is supposed to pay a
**roll-down carry** — as a linker ages it slides down a (normally upward-sloping) real
curve and is repriced at a lower real yield, a gain over and above the running real
coupon. If that is true, the longer buckets should out-earn cash by more than the short
one, and the part of a TIPS fund's return that is *not* explained by a duration-matched
nominal Treasury fund should be reliably positive.

**Identification caveat (stated up front, and again in the README).** A long-TIPS /
short-nominal residual is, by construction, a **long-breakeven** position: it earns
realised inflation minus the breakeven priced at purchase. Roll-down in real yields and
the inflation surprise are *not separately identifiable* from fund total returns alone.
This study measures the residual and is explicit that a positive residual is consistent
with either — which is precisely why the era cut around the 2021-2023 inflation shock
carries so much of the argument.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the hedge
beta is estimated on data through day ``t-1`` and applied at day ``t``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The four inflation-linked buckets, shortest real duration first. Labels are the
# funds' published effective real durations (a PROXY: they come from the sponsors'
# fact sheets, not from the tape, and they drift; nothing in the analysis depends on
# their exact value — every hedge ratio is estimated from returns).
TIPS_LEGS = ("VTIP", "SCHP", "TIP", "LTPZ")
TIPS_LABEL = {
    "VTIP": "short 0-5y linkers (~2.5y real duration)",
    "SCHP": "broad TIPS (~7y real duration)",
    "TIP": "broad TIPS, the incumbent (~7y real duration)",
    "LTPZ": "long 15y+ linkers (~20y real duration)",
}

# Duration-matched nominal Treasury funds. This PAIRING IS AN ASSUMPTION (sponsor
# durations, not tape) and is swept in `strategy.pairing_sweep`; the hedge ratio
# itself is always estimated from returns, so the pairing only sets the factor.
NOMINAL_MATCH = {"VTIP": "SHY", "SCHP": "IEF", "TIP": "IEF", "LTPZ": "TLT"}
ALT_MATCH = {"VTIP": ("SHY", "IEI", "IEF"), "SCHP": ("IEF", "IEI"),
             "TIP": ("IEF", "IEI"), "LTPZ": ("TLT", "IEF")}

CASH = "BIL"

TICKERS = ("VTIP", "SCHP", "TIP", "LTPZ", "SHY", "IEI", "IEF", "TLT", "BIL")

# The eight columns the headline fingerprint covers (IEI is a robustness leg only).
HEADLINE_COLS = ("VTIP", "SCHP", "TIP", "LTPZ", "SHY", "IEF", "TLT", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time, so the sample
# never creeps between reruns.
AS_OF = "2026-06-30"

# The inflation shock that dominates this sample, used for the era cut.
SHOCK_START = "2021-01-01"
SHOCK_END = "2023-12-31"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2003-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted total return — indispensable
    here, because a bond fund's whole return is coupons and (for linkers) the
    inflation accrual paid out as distributions. A price-only close would understate
    every leg and would flatter or damn the buckets unevenly by yield.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    dropna: bool = True,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a date-indexed frame with one column per ticker, sliced to ``asof`` so the
    sample never creeps, and (by default) restricted to the common window on which
    every leg trades — VTIP's 2012-10 inception gates it. Raises ``FileNotFoundError``
    if any ticker is missing; the offline core and the tests never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call tips_roll.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    if dropna:
        df = df.dropna()
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp.

    Canonical by construction (ISO dates + values rounded to 6 dp, columns sorted), so
    it matches :func:`quantlab.repro.fingerprint` on the same frame.
    """
    h = hashlib.sha256()
    for ts in prices.index:
        h.update(str(pd.Timestamp(ts).date()).encode())
        h.update(b"\x1f")
    for c in sorted(prices.columns):
        h.update(b"\x1e")
        h.update(str(c).encode())
        for v in prices[c].to_numpy():
            token = "nan" if pd.isna(v) else f"{round(float(v), 6):.6f}"
            h.update(token.encode())
            h.update(b",")
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted residual carry)
# --------------------------------------------------------------------------- #
def _cash_index(n: int, cash_rate_ann: float) -> np.ndarray:
    daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    return np.cumprod(np.full(n, daily))


def synthetic_daily(
    n_years: int = 14,
    beta_true: float = 0.85,        # linker's loading on the duration factor
    carry_ann: float = 0.020,       # planted residual carry at signal_strength = 1
    factor_vol_ann: float = 0.065,  # annualised vol of the common duration factor
    idio_vol_ann: float = 0.020,    # annualised idiosyncratic vol of the linker
    nominal_idio_vol_ann: float = 0.010,
    cash_rate_ann: float = 0.020,
    signal_strength: float = 1.0,
    start: str = "2012-10-16",
    seed: int = 949,
):
    """One linker / nominal / cash tape with a *planted* duration-hedged carry.

    The world: a common duration factor ``f`` (the level of real and nominal rates)
    drives the nominal Treasury fund one-for-one; the inflation-linked fund loads on it
    with ``beta_true`` and additionally earns ``carry_ann * signal_strength`` per year
    of residual carry that a duration hedge should isolate.

    - ``signal_strength = 1`` → a genuine, recoverable residual carry.
    - ``signal_strength = 0`` → **no** residual carry at all (the null); the
      duration-hedged alpha must be indistinguishable from zero.

    Returns ``(prices, truth)``. ``prices`` carries ``tips``, ``nominal`` and ``cash``
    total-return close levels; ``truth`` records the planted parameters. Deterministic
    given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    # bdate_range with a few thousand bars stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n)

    sf = factor_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    si = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    sn = nominal_idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    carry_d = carry_ann * float(signal_strength) / TRADING_DAYS_PER_YEAR

    f = rng.normal(0.0, sf, n)
    ex_nom = f + rng.normal(0.0, sn, n)
    ex_tips = beta_true * f + carry_d + rng.normal(0.0, si, n)

    cash_d = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    prices = pd.DataFrame(
        {
            "tips": 100.0 * np.cumprod(1.0 + cash_d + ex_tips),
            "nominal": 100.0 * np.cumprod(1.0 + cash_d + ex_nom),
            "cash": 100.0 * _cash_index(n, cash_rate_ann),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength),
        "beta_true": beta_true,
        "carry_ann": carry_ann,
        "carry_ann_planted": carry_ann * float(signal_strength),
        "factor_vol_ann": factor_vol_ann,
        "idio_vol_ann": idio_vol_ann,
        "cash_rate_ann": cash_rate_ann,
        "n_years": n_years, "n_days": n, "seed": seed,
    }
    return prices, truth


def synthetic_panel(
    n_years: int = 14,
    durations=(1.0, 3.0, 8.0),      # relative duration of the three buckets
    carry_ann: float = 0.020,
    signal_strength: float = 1.0,
    factor_vol_ann: float = 0.020,  # per unit of relative duration
    idio_vol_ann: float = 0.010,
    cash_rate_ann: float = 0.020,
    start: str = "2012-10-16",
    seed: int = 949,
):
    """A three-bucket ladder: linker and nominal fund at each of ``durations``.

    The same common factor drives every bucket, scaled by its relative duration, so the
    panel reproduces the real tape's defining feature: the long buckets carry several
    times the volatility of the short one. Each linker earns the *same* planted residual
    carry ``carry_ann * signal_strength`` — so a ladder race that shows carry rising with
    duration on this panel would be an artefact, not a finding.

    Returns ``(prices, truth)`` with columns ``tips_0..tips_k``, ``nom_0..nom_k`` and
    ``cash``, plus the planted parameters and the per-bucket relative durations.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n)
    carry_d = carry_ann * float(signal_strength) / TRADING_DAYS_PER_YEAR
    cash_d = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    sf = factor_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    si = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    f = rng.normal(0.0, sf, n)
    out = {}
    for k, d in enumerate(durations):
        ex_nom = d * f + rng.normal(0.0, si * np.sqrt(d), n)
        ex_tips = d * f + carry_d + rng.normal(0.0, si * np.sqrt(d), n)
        out[f"nom_{k}"] = 100.0 * np.cumprod(1.0 + cash_d + ex_nom)
        out[f"tips_{k}"] = 100.0 * np.cumprod(1.0 + cash_d + ex_tips)
    out["cash"] = 100.0 * _cash_index(n, cash_rate_ann)

    prices = pd.DataFrame(out, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": float(signal_strength),
        "carry_ann": carry_ann,
        "carry_ann_planted": carry_ann * float(signal_strength),
        "durations": tuple(float(d) for d in durations),
        "n_buckets": len(durations),
        "cash_rate_ann": cash_rate_ann,
        "n_years": n_years, "n_days": n, "seed": seed,
    }
    return prices, truth
