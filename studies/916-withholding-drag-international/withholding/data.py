"""Data layer for Study 916 — Withholding Drag on US-domiciled international funds.

Three tapes, one shape (a date-indexed daily frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the broad developed-ex-US funds
  (VEA, IEFA, EFA, VXUS), the single-country funds that make up most of that market
  (EWJ, EWG, EWU) and a cash proxy (BIL). Cached in the **shared** desk cache
  ``studies/_cache`` under the house convention ``prices_<TICKER>_1d.parquet``.

- ``fetch`` / ``load_distributions`` — the *price-only* leg: ``auto_adjust=False``
  closes (split-adjusted, **not** dividend-adjusted) together with the per-share
  ``dividend`` and ``capital_gains`` cash amounts, cached as
  ``divs_<TICKER>_1d.parquet``. This second tape is what makes the study possible:
  the gap between the total-return leg and the price-only leg *is* the realised
  distribution the fund actually paid — the income that reached a US holder's
  account, already **net** of foreign withholding at the fund level.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators.
  ``synthetic_panel`` builds a small panel of funds on a common market factor, each
  paying a quarterly dividend out of a known **gross** yield after a known
  withholding rate and a known expense ratio. The ``signal_strength`` knob scales the
  planted *difference* in withholding between the broad fund and the single-country
  benchmark: at ``signal_strength=0`` every fund suffers the same withholding (the
  null — the estimator must find no gap); at ``signal_strength=1`` the broad fund
  loses a further, recoverable slice of its dividend. Seeds are fixed, so the whole
  test-suite is deterministic and runs with **no cache present**.

What is and is not on the tape
------------------------------
The tape gives the **net** distribution exactly. It does *not* give the gross
dividend the underlying companies declared, so the gross/net split — the withholding
rate itself — is **never observed**. Any number this study puts on "withholding drag"
is therefore an *inference* conditional on an assumed effective rate, and is swept
across a grid rather than asserted. Expense ratios are likewise a hardcoded,
publicly-verifiable **ASSUMPTION** (fund fact sheets), not tape, and are swept too.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the blend
weights known through day ``t`` are traded at day ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Broad US-domiciled developed-ex-US (or total-ex-US) funds — the things under test.
BROAD = ("VEA", "IEFA", "EFA", "VXUS")
# Single-country funds covering the three largest developed-ex-US markets. All are
# US-domiciled iShares MSCI funds, so they suffer the *same* treaty withholding —
# which is exactly why they are a composition benchmark, not a gross-yield oracle.
SINGLE = ("EWJ", "EWG", "EWU")
CASH = "BIL"
TICKERS = BROAD + SINGLE + (CASH,)

# --------------------------------------------------------------------------- #
# NON-TAPE ASSUMPTIONS (labelled, swept in strategy.py)
# --------------------------------------------------------------------------- #
# Net prospectus expense ratios, decimal per year, from the funds' own fact sheets
# (2026). These are an ASSUMPTION, not tape: distributions are paid net of expenses,
# so a fee gap masquerades as an income-yield gap unless it is added back.
# ANACHRONISM, named: these are TODAY's fees applied to a 19-year window, and ETF fees
# fell over it (VEA was well above 10 bp at its 2007 launch vs 3 bp now; the iShares
# single-country funds barely moved). The time-averaged fee gap is therefore SMALLER
# than the 47 bp the headline adds back, so the fee-adjusted gap is an upper bound —
# it flatters the withholding hypothesis, which needs a positive gap.
# Swept in strategy.fee_gap_sweep.
EXPENSE_RATIO = {
    "VEA": 0.0003, "IEFA": 0.0007, "EFA": 0.0033, "VXUS": 0.0005,
    "EWJ": 0.0050, "EWG": 0.0050, "EWU": 0.0050, "BIL": 0.0013,
}

# Static blend weights for the single-country benchmark, roughly the Japan / UK /
# Germany shares of MSCI EAFE renormalised to 1. An ASSUMPTION (index fact sheets),
# swept in strategy.blend_weight_sweep.
BLEND_WEIGHTS = {"EWJ": 0.50, "EWU": 0.30, "EWG": 0.20}

# Effective withholding a US fund typically suffers on dividends from these markets:
# Japan 15% and Germany 15% by treaty (Germany's 26.375% statutory rate is reduced to
# the treaty 15% by reclaim, which funds do not always fully recover); the UK 0% is
# NOT a treaty rate — the UK simply levies no dividend withholding tax under its own
# domestic law. An ASSUMPTION used only for the *inference* step; swept across a grid.
TREATY_RATE = {"EWJ": 0.15, "EWG": 0.15, "EWU": 0.00}
ASSUMED_EFFECTIVE_WITHHOLDING = 0.12   # blended EAFE central case (ASSUMPTION)

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Cache plumbing
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _tr_path(ticker: str, cache_dir: str) -> str:
    """Total-return close, house convention (shared with every other study)."""
    return os.path.join(cache_dir, f"prices_{_safe(ticker)}_1d.parquet")


def _div_path(ticker: str, cache_dir: str) -> str:
    """Price-only close + per-share dividend / capital-gain cash amounts."""
    return os.path.join(cache_dir, f"divs_{_safe(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1996-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download both legs for ``tickers`` and cache them as parquet.

    Leg 1 (``auto_adjust=True``) is the daily **total-return** close — the house
    convention every other study reads. Leg 2 (``auto_adjust=False, actions=True``)
    is the **price-only** close plus the per-share ``dividend`` and
    ``capital_gains`` cash amounts. The difference between the two legs is the
    realised distribution — the whole point of this study.

    Network-only; run once to populate the shared cache. Retries up to ``retries``.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        # --- leg 1: total return -------------------------------------------- #
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no total-return data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        tr = raw[["close"]].copy()
        tr.index = pd.to_datetime(tr.index)
        tr.index.name = "date"
        tr = tr.dropna(subset=["close"])
        tr.to_parquet(_tr_path(tk, cache_dir))

        # --- leg 2: price-only + distributions ------------------------------ #
        raw2 = None
        for _ in range(retries):
            try:
                raw2 = yf.download(tk, start=start, end=end, interval="1d",
                                   auto_adjust=False, actions=True, progress=False)
                if raw2 is not None and len(raw2) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw2 is None or len(raw2) == 0:
            raise RuntimeError(f"yfinance returned no price-only data for {tk}")
        if isinstance(raw2.columns, pd.MultiIndex):
            raw2.columns = raw2.columns.get_level_values(0)
        raw2 = raw2.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
        px = pd.DataFrame({"close": raw2["close"]})
        px["dividend"] = raw2["dividends"] if "dividends" in raw2 else 0.0
        px["capital_gains"] = raw2["capital_gains"] if "capital_gains" in raw2 else 0.0
        px.index = pd.to_datetime(px.index)
        px.index.name = "date"
        px = px.dropna(subset=["close"]).fillna({"dividend": 0.0, "capital_gains": 0.0})
        px.to_parquet(_div_path(tk, cache_dir))

        out[tk] = px
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff both legs of every ticker are cached (offline-testable)."""
    return all(
        os.path.exists(_tr_path(tk, cache_dir)) and os.path.exists(_div_path(tk, cache_dir))
        for tk in tickers
    )


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily **total-return** closes OFFLINE into one aligned frame.

    One column per ticker, sliced to ``asof`` so the sample never creeps. Raises
    ``FileNotFoundError`` if any ticker is missing — the offline core and the whole
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _tr_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached total-return prices for {tk} at {path}. "
                f"Call withholding.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_distributions(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> dict[str, pd.DataFrame]:
    """Read the cached **price-only** leg OFFLINE: close, dividend, capital_gains.

    Returns one frame per ticker (they have different histories, so no forced
    alignment here — ``strategy.py`` intersects the windows it needs).
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _div_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached distribution leg for {tk} at {path}. "
                f"Call withholding.data.fetch() once to populate the shared cache."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        out[tk] = df[df.index <= pd.Timestamp(asof)].sort_index()
    return out


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — deterministic offline core (a planted withholding gap)
# --------------------------------------------------------------------------- #
def _one_fund(
    rng: np.random.Generator,
    n_days: int,
    factor: np.ndarray,
    beta: float,
    idio_vol: float,
    gross_yield: float,
    withhold: float,
    expense: float,
    ex_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (price_only, total_return, dividend_per_share) paths for one fund.

    The fund earns ``gross_yield`` per year on its holdings, loses ``withhold`` of it
    to foreign tax and ``expense`` (as a fraction of NAV per year) to fees, and
    distributes the remainder in ``252 / ex_period`` equal instalments. The price
    leg drops by the cash distributed on each ex-date; the total-return leg does not.
    So ``r_tr - r_px`` summed over a year recovers exactly the **net** distributed
    yield — and never the gross one.
    """
    idio = rng.normal(0.0, idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    # Capital return excludes the income component (income is paid out, not retained).
    cap = beta * factor + idio - expense / TRADING_DAYS_PER_YEAR

    net_yield = gross_yield * (1.0 - withhold) - expense
    net_yield = max(net_yield, 0.0)
    per_pay = net_yield * ex_period / TRADING_DAYS_PER_YEAR

    px = np.empty(n_days)
    tr = np.empty(n_days)
    div = np.zeros(n_days)
    px[0] = 100.0
    tr[0] = 100.0
    for i in range(1, n_days):
        grow = np.exp(cap[i])
        p_pre = px[i - 1] * grow
        pay = p_pre * per_pay if (i % ex_period == 0) else 0.0
        px[i] = p_pre - pay
        div[i] = pay
        tr[i] = tr[i - 1] * (px[i] + pay) / px[i - 1]
    return px, tr, div


def synthetic_panel(
    n_years: int = 18,
    signal_strength: float = 1.0,
    gross_yield: float = 0.032,
    base_withholding: float = 0.10,
    extra_broad_withholding: float = 0.08,
    market_vol: float = 0.16,
    market_drift: float = 0.05,
    idio_vol: float = 0.06,
    ex_period: int = 63,
    start: str = "2008-01-02",
    seed: int = 916,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A panel of funds on one market factor, with a planted withholding difference.

    Four funds are built:

    - ``broad``     — the fund under test. Withholding
      ``base_withholding + signal_strength * extra_broad_withholding``.
    - ``bench_a/b/c`` — the single-country benchmark constituents, all at
      ``base_withholding``, all with the same gross yield.

    Every fund carries the *same* expense ratio here, so the only planted difference
    is the tax leak. At ``signal_strength = 0`` the broad fund's withholding equals
    the benchmark's — the **null**: the estimator must report a gap of ~0 bp/yr. At
    ``signal_strength = 1`` the planted drag is
    ``gross_yield * extra_broad_withholding`` in bp/yr, which the estimator must
    recover.

    Returns ``(tr, px, truth)``: two date-indexed frames of total-return and
    price-only closes with matching columns, plus the planted parameters and the
    implied true gap.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a modest bdate_range, well inside pandas' ns Timestamp horizon.
    dates = pd.bdate_range(start=start, periods=n_days)

    factor = rng.normal(
        market_drift / TRADING_DAYS_PER_YEAR,
        market_vol / np.sqrt(TRADING_DAYS_PER_YEAR),
        n_days,
    )
    w_broad = base_withholding + signal_strength * extra_broad_withholding
    expense = 0.0015

    specs = [
        ("broad", 1.00, w_broad),
        ("bench_a", 1.05, base_withholding),
        ("bench_b", 0.95, base_withholding),
        ("bench_c", 1.00, base_withholding),
    ]
    px_cols, tr_cols = {}, {}
    for k, (name, beta, w) in enumerate(specs):
        p, t, _ = _one_fund(
            rng, n_days, factor, beta, idio_vol, gross_yield, w, expense, ex_period
        )
        px_cols[name] = p
        tr_cols[name] = t

    idx = pd.DatetimeIndex(dates, name="date")
    px = pd.DataFrame(px_cols, index=idx)
    tr = pd.DataFrame(tr_cols, index=idx)

    truth = {
        "signal_strength": signal_strength,
        "gross_yield": gross_yield,
        "base_withholding": base_withholding,
        "w_broad": w_broad,
        "extra_broad_withholding": signal_strength * extra_broad_withholding,
        "expense": expense,
        "true_gap_bp": gross_yield * signal_strength * extra_broad_withholding * 1e4,
        "net_yield_bench_bp": (gross_yield * (1 - base_withholding) - expense) * 1e4,
        "n_years": n_years, "n_days": n_days, "seed": seed,
        "ex_period": ex_period, "bench_cols": ["bench_a", "bench_b", "bench_c"],
    }
    return tr, px, truth


def synthetic_daily(
    n_years: int = 18,
    signal_strength: float = 1.0,
    seed: int = 916,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Single-fund convenience view of :func:`synthetic_panel`.

    Returns one frame with ``tr`` (total-return close) and ``px`` (price-only close)
    for the broad fund, plus the same ``truth`` dict — the shape a one-fund income
    measurement test wants.
    """
    tr, px, truth = synthetic_panel(
        n_years=n_years, signal_strength=signal_strength, seed=seed, **kwargs
    )
    out = pd.DataFrame({"tr": tr["broad"], "px": px["broad"]})
    return out, truth
