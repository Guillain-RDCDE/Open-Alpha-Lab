"""Data layer for Study 952 — After-Tax Equivalent (munis vs taxable credit, after tax).

The claim under test is an arithmetic one dressed up as an investment one: because
municipal coupon income escapes federal income tax, a taxable-account investor above some
marginal rate should prefer municipal bonds to taxable credit. The question this study
answers is *which* rate — the **break-even bracket** — and whether the after-tax win, once
it exists, is large enough to be distinguishable from noise on the real tape.

Two flavours of the same daily closes are cached, because the tax story lives entirely in
the **income** leg and income is not directly quoted:

- ``prices_<TICKER>_1d.parquet`` — **total-return** closes (``yfinance``,
  ``auto_adjust=True``): price plus reinvested distributions, net of fund fees.
- ``praw_<TICKER>_1d.parquet`` — **price-only** closes (``auto_adjust=False``).

The monthly **income (distribution) return** is then *reconstructed* as
``total_return - price_return``. That is the only leg the tax code touches, so it is
measured rather than assumed. The reconstruction is noisy at the month boundary (an
ex-dividend date landing either side of a month end), which shows up as a handful of
slightly negative "income" months; the loader floors income at zero by default and
:func:`~after_tax.strategy.income_floor_sensitivity` re-runs the headline unfloored.

The tape:

- **Muni legs** — ``MUB`` (iShares national IG muni), ``VTEB`` (Vanguard national IG muni,
  the cheap twin), ``SUB`` (iShares short-maturity muni), ``HYD`` (VanEck high-yield muni).
- **Taxable legs** — ``AGG`` (US Aggregate: Treasury/MBS-heavy), ``LQD`` (long IG
  corporate), ``VCIT`` (intermediate IG corporate — the closest duration match to MUB).
- **Cash** — ``BIL`` (1-3 month T-bills), the tradable risk-free leg. Its income is
  federally taxable but **state-exempt**, and the excess-of-cash races subtract the
  *after-tax* BIL leg so the comparison is apples-to-apples inside a taxable account.

``fetch`` touches the network and populates the shared cache under ``studies/_cache``
(retry up to 4x); ``load_prices`` / ``load_price_only`` read that cache **offline** and
never import ``yfinance``. The whole test-suite runs with NO cache present (synthetic
only), so CI is green on a fresh checkout.

``synthetic_panel`` is the deterministic offline control: a monthly world with a planted
pre-tax yield gap and a tunable ``signal_strength`` knob, used to prove the break-even
solver and the HAC/bootstrap machinery recover a planted effect and stay quiet on the null.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache, two levels up (studies/_cache) — not a study-local copy.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

MONTHS_PER_YEAR = 12

MUNI_TICKERS = ("MUB", "VTEB", "SUB", "HYD")
TAXABLE_TICKERS = ("LQD", "AGG", "VCIT")
CASH_TICKER = "BIL"
TICKERS = MUNI_TICKERS + TAXABLE_TICKERS + (CASH_TICKER,)

# Study-wide as-of: the last COMPLETE calendar month at build time. Partial months are
# dropped so the monthly sample never creeps between reruns.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Cache plumbing
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _cache_path(ticker: str, cache_dir: str, kind: str = "prices") -> str:
    """``kind='prices'`` = total-return closes; ``kind='praw'`` = price-only closes."""
    return os.path.join(cache_dir, f"{kind}_{_safe(ticker)}_1d.parquet")


def _download(ticker: str, start: str, end, auto_adjust: bool, retries: int) -> pd.DataFrame:
    import yfinance as yf  # lazy: only when we actually go to the network

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(
                ticker, start=start, end=end, interval="1d",
                auto_adjust=auto_adjust, progress=False,
            )
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    df = raw[["close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.dropna(subset=["close"])


def fetch(
    tickers=TICKERS,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> None:
    """Download BOTH the total-return and the price-only daily closes; cache as parquet.

    Network-only; run once to populate the shared cache. The pair is mandatory: the
    income (distribution) return — the only leg the tax code touches — is recovered as
    ``total_return - price_return``, so a total-return tape alone cannot answer this
    study's question.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        _download(tk, start, end, True, retries).to_parquet(
            _cache_path(tk, cache_dir, "prices"))
        _download(tk, start, end, False, retries).to_parquet(
            _cache_path(tk, cache_dir, "praw"))


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff BOTH tapes are cached for every ticker (offline-testable)."""
    return all(
        os.path.exists(_cache_path(tk, cache_dir, k))
        for tk in tickers for k in ("prices", "praw")
    )


def _load(kind: str, tickers, cache_dir: str, asof: str) -> pd.DataFrame:
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, kind)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached {kind} tape for {tk} at {path}. "
                f"Call after_tax.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_prices(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily **total-return** closes, offline, sliced to ``asof``."""
    return _load("prices", tickers, cache_dir, asof)


def load_price_only(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                    asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily **price-only** closes, offline, sliced to ``asof``."""
    return _load("praw", tickers, cache_dir, asof)


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a numeric frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Daily closes -> monthly total / price / income returns
# --------------------------------------------------------------------------- #
def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns from daily closes, indexed by monthly ``Period``.

    Grouping on ``index.to_period('M')`` (rather than ``resample``) keeps the code
    identical across pandas 2.x and 3.x, where the resample alias for month-end changed.
    """
    per = daily.index.to_period("M")
    last = daily.groupby(per).last()
    out = last.pct_change(fill_method=None).dropna(how="all")
    out.index.name = "month"
    return out


def decompose(
    total_daily: pd.DataFrame,
    price_daily: pd.DataFrame,
    floor_income: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split monthly total return into a **price** leg and an **income** leg.

    ``income = total - price``. Both tapes come from the same vendor and the same closes,
    so the difference is the reinvested distribution — the leg the tax code taxes.
    Ex-dividend dates that straddle a month end make a handful of months come out very
    slightly negative; ``floor_income=True`` clips those to zero (a taxpayer never gets a
    coupon refund). :func:`after_tax.strategy.income_floor_sensitivity` re-runs the
    headline with the floor off so the choice is visible, not hidden.

    Returns a dict with ``total``, ``price`` and ``income`` monthly frames on a common
    index.
    """
    tot = to_monthly(total_daily)
    pri = to_monthly(price_daily)
    cols = [c for c in tot.columns if c in pri.columns]
    idx = tot.index.intersection(pri.index)
    tot = tot.loc[idx, cols]
    pri = pri.loc[idx, cols]
    inc = tot - pri
    if floor_income:
        inc = inc.clip(lower=0.0)
    return {"total": tot, "price": pri, "income": inc}


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline control
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 40,
    muni_income_ann: float = 0.030,     # planted muni coupon yield (tax-exempt)
    taxable_income_ann: float = 0.045,  # planted taxable coupon yield (fully taxed)
    cash_income_ann: float = 0.020,     # planted T-bill yield
    duration_vol_ann: float = 0.045,    # shared rate-driven price vol
    idio_vol_ann: float = 0.010,        # leg-specific price vol
    beta_spread: float = 0.05,          # duration mismatch between the two legs
    price_drift_ann: float = 0.0,       # bonds pull to par: no structural price drift
    signal_strength: float = 1.0,       # 0 = twin legs (the null), 1 = full planted gap
    start: str = "1990-01",
    seed: int = 952,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A monthly muni / taxable / cash world with a PLANTED pre-tax yield gap.

    Three legs (``muni``, ``taxable``, ``cash``) share a rate factor (so their price legs
    co-move like real bond funds) and each carries a constant coupon yield. The
    ``signal_strength`` knob blends both the coupon-yield gap *and* the duration mismatch
    toward zero:

    - ``signal_strength = 1`` → the full planted world: the taxable leg yields 150 bp more
      pre-tax and carries a little more duration. The theoretical break-even effective
      rate is then ``(y_tax - y_muni) / y_tax`` = 33.3% with the defaults, and the solver
      must recover it from the tape alone.
    - ``signal_strength = 0`` → the **null**: identical coupon yields and identical
      duration, so the two legs are statistical twins. The break-even must collapse to ~0
      and the *pre-tax* difference must be indistinguishable from zero. (The after-tax
      difference at a positive bracket is *not* zero even here — that is the point: it is
      pure tax arithmetic, ``tau x yield``, and no evidence of any market edge.)

    Returns ``(panel, truth)`` where ``panel`` mirrors :func:`decompose`'s output
    (``total`` / ``price`` / ``income`` monthly frames) and ``truth`` records the planted
    parameters, including the theoretical break-even rate. Deterministic given ``seed``.

    The index is a monthly ``period_range`` — no huge ``date_range``, so nothing can
    overflow the nanosecond Timestamp horizon on pandas 2.x.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS_PER_YEAR
    idx = pd.period_range(start=start, periods=n, freq="M")
    idx.name = "month"

    mid = 0.5 * (muni_income_ann + taxable_income_ann)
    muni_y = (1 - signal_strength) * mid + signal_strength * muni_income_ann
    tax_y = (1 - signal_strength) * mid + signal_strength * taxable_income_ann
    spread = signal_strength * beta_spread

    rate_shock = rng.normal(0.0, duration_vol_ann / np.sqrt(MONTHS_PER_YEAR), n)
    drift_m = price_drift_ann / MONTHS_PER_YEAR
    idio = idio_vol_ann / np.sqrt(MONTHS_PER_YEAR)

    price = pd.DataFrame(
        {
            # The muni leg carries slightly less duration than the taxable leg.
            "muni": drift_m + (1.0 - spread) * rate_shock + rng.normal(0.0, idio, n),
            "taxable": drift_m + (1.0 + spread) * rate_shock + rng.normal(0.0, idio, n),
            "cash": np.zeros(n),
        },
        index=idx,
    )
    income = pd.DataFrame(
        {
            "muni": np.full(n, muni_y / MONTHS_PER_YEAR),
            "taxable": np.full(n, tax_y / MONTHS_PER_YEAR),
            "cash": np.full(n, cash_income_ann / MONTHS_PER_YEAR),
        },
        index=idx,
    )
    total = price + income

    truth = {
        "signal_strength": signal_strength,
        "muni_income_ann": float(muni_y),
        "taxable_income_ann": float(tax_y),
        "cash_income_ann": cash_income_ann,
        "n_months": int(n),
        "n_years": n_years,
        "seed": seed,
        # Theoretical break-even from the PLANTED parameters (the price legs are drift-free,
        # so the whole gap is the coupon gap): tau* = (y_tax - y_muni) / y_tax.
        "planted_breakeven": float((tax_y - muni_y) / tax_y),
        "planted_yield_gap_ann": float(tax_y - muni_y),
    }
    return {"total": total, "price": price, "income": income}, truth
