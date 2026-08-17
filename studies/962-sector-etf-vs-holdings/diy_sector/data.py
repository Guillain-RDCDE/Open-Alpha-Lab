"""Data layer for Study 962 — Do It Yourself (a sector ETF vs its own top holdings).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for three SPDR Select Sector funds (XLK, XLE,
  XLF), the single names that make up their published top-10 holdings today *and*
  circa January 2011, and a cash proxy (BIL, the 1-3 month T-bill ETF). ``fetch``
  touches the network and caches parquet under the **shared** ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI is
  green on a fresh checkout.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. A
  one-factor sector world: ``n_names`` large constituents plus a long tail of small
  ones, a cap-weighted "fund" leg that charges an explicit annual fee, and a cash
  accrual leg. The ``signal_strength`` knob scales that fee: at ``signal_strength=1``
  the fund skims a real fee the do-it-yourself basket can recover; at
  ``signal_strength=0`` the fee is zero and there is nothing to recover (the null).

The claim under test: a sector ETF is mostly a handful of mega-caps, so you can hold
the top N names directly, skip the (small) expense ratio, and pocket the difference.
The two things that can spoil it are **tracking error** (the tail you left out) and
**rebalancing cost** (the fund rebalances for free inside its own NAV; you pay).

### Non-tape inputs — labelled PROXY / ASSUMPTION

1. **The holdings lists themselves are hardcoded, not scraped.** Yahoo! Finance does
   not serve a point-in-time holdings history, so both name lists below are typed in
   from public fund fact sheets and treated as a fixed input:
   - ``top_2026`` — the top-10 names and approximate cap weights as published near the
     study as-of date (2026-06-30). Using *today's* list over the *whole* sample is a
     deliberate, named **look-ahead**: it is the hindsight portfolio a reader would
     accidentally build. It is the thing this study measures, not something it hides.
   - ``top_2011`` — the top-10 names as published circa January 2011, held **equal
     weight, fixed from the first day of the sample**. This is the honest,
     contemporaneous control for that look-ahead.
2. **Survivorship in the 2011 list.** Two genuine January-2011 constituents (Anadarko
   Petroleum and Marathon Oil, both since acquired) have no retrievable Yahoo! tape —
   the ``APC`` symbol now resolves to an unrelated new listing — so the 2011 energy
   list below is **survivor-only**: the two dead names are replaced by the next
   *surviving* names down the same fact sheet, which is itself a selection on having
   survived. More generally, **every** name in **every** 2011 list is by construction a
   name with a retrievable tape today. That biases the control basket in the DIY
   investor's *favour*; it is named on the Signal axis rather than papered over.
3. **Ticker continuity.** ``GOOGL`` and ``BRK-B`` stand in for the share lines a 2011
   investor actually held (Google traded as a single ``GOOG`` line until the April-2014
   class split). Yahoo!'s back-adjusted series is the closest retrievable total return;
   it is a labelled **PROXY**, not the literal 2011 instrument.
4. **Nothing is taxed and nothing is spread-charged twice.** Dividends are reinvested
   frictionlessly at the close on *both* legs (``auto_adjust=True``), and no capital-gains
   tax is modelled anywhere. That assumption runs in the **DIY basket's favour**: an ETF
   sheds appreciated stock through in-kind redemption, while a private investor
   rebalancing ten names realises the gain. The single ``cost_bps`` one-way charge is the
   only friction on the DIY side and stands in for commission *and* spread together.
5. **Expense ratios** (0.08%/yr for the Select Sector SPDRs at the as-of date) are
   quoted for context only. They are *not* added anywhere: the fund's own
   total-return close already carries its fee (and any securities-lending revenue it
   earns back), so the race compares like with like.

No look-ahead is baked into the mechanics here — that discipline lives in
``strategy.py`` (weights formed at a month-end close ``t`` are traded at ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# The sample opens in 2011 so the January-2011 control list is contemporaneous with
# its own start, and so every fund has a long common window with the cash leg.
SAMPLE_START = "2011-01-03"

CASH = "BIL"

# --------------------------------------------------------------------------- #
# The hardcoded holdings lists (PROXY — see the module docstring)
# --------------------------------------------------------------------------- #
# ``top_2026``: (ticker, approximate published cap weight in % of fund) near 2026-06-30.
# Weights are renormalised inside the basket, so only their *ratios* matter.
# ``top_2011``: the published top-10 circa 2011-01, held EQUAL weight from day one.
SECTORS: dict[str, dict] = {
    "XLK": {
        "label": "Technology Select Sector SPDR",
        "expense_ratio": 0.0008,
        "top_2026": [
            ("NVDA", 16.0), ("MSFT", 14.0), ("AAPL", 12.0), ("AVGO", 6.0),
            ("ORCL", 4.0), ("CRM", 3.0), ("AMD", 3.0), ("CSCO", 3.0),
            ("IBM", 3.0), ("ACN", 2.0),
        ],
        # XLK still contained the telecoms (T, VZ) in 2011 — it did until the 2018
        # GICS reshuffle. Keeping them is the point of a contemporaneous list.
        "top_2011": ["AAPL", "MSFT", "IBM", "GOOGL", "T", "ORCL", "INTC",
                     "CSCO", "QCOM", "VZ"],
    },
    "XLE": {
        "label": "Energy Select Sector SPDR",
        "expense_ratio": 0.0008,
        "top_2026": [
            ("XOM", 22.0), ("CVX", 17.0), ("COP", 7.0), ("WMB", 5.0),
            ("EOG", 4.5), ("KMI", 4.0), ("OKE", 3.5), ("SLB", 3.5),
            ("PSX", 3.0), ("MPC", 3.0),
        ],
        # Survivor-only: Anadarko (APC) and Marathon Oil (MRO) were genuine 2011
        # top-10 names but have no retrievable tape. Named, not hidden.
        "top_2011": ["XOM", "CVX", "SLB", "COP", "OXY", "HAL", "APA", "NOV",
                     "DVN", "EOG"],
    },
    "XLF": {
        "label": "Financial Select Sector SPDR",
        "expense_ratio": 0.0008,
        "top_2026": [
            ("BRK-B", 13.0), ("JPM", 10.0), ("V", 7.5), ("MA", 6.5),
            ("BAC", 4.5), ("WFC", 4.0), ("GS", 3.0), ("SPGI", 2.8),
            ("AXP", 2.5), ("MS", 2.4),
        ],
        "top_2011": ["JPM", "WFC", "BRK-B", "C", "BAC", "GS", "USB", "AXP",
                     "MET", "PNC"],
    },
}

FUNDS = tuple(SECTORS.keys())


def _sector_tickers(fund: str) -> list[str]:
    names = [t for t, _ in SECTORS[fund]["top_2026"]] + SECTORS[fund]["top_2011"]
    out: list[str] = []
    for t in names:
        if t not in out:
            out.append(t)
    return out


def all_tickers() -> tuple[str, ...]:
    """Every symbol this study needs: the three funds, the cash leg, all constituents."""
    out: list[str] = list(FUNDS) + [CASH]
    for f in FUNDS:
        for t in _sector_tickers(f):
            if t not in out:
                out.append(t)
    return tuple(out)


TICKERS = all_tickers()


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2006-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
    skip_cached: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted total return — mandatory here,
    because the whole race is a fund's total return against a basket of single-name
    total returns, and the dividend yields differ enormously across the three sectors
    (energy and financials pay, technology mostly does not).
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if skip_cached and os.path.exists(path):
            out[tk] = pd.read_parquet(path)
            continue
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
        df.to_parquet(path)
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline check)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    start: str = SAMPLE_START,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    One column per ticker, indexed by date, sliced to ``[start, asof]`` so the sample
    never creeps. Columns are *not* forward-filled and *not* inner-joined: a name that
    had not listed yet (PSX, 2012) or that has a shorter tape simply carries NaN, and
    ``strategy.basket_returns`` renormalises the basket over whatever is live that day.
    Raises ``FileNotFoundError`` if any ticker is missing — the offline core and the
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call diy_sector.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(asof))]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def basket_weights(fund: str, n: int, scheme: str = "cap2026") -> pd.Series:
    """Target weights for the top-``n`` DIY basket of ``fund``.

    Three schemes, and the third one exists to keep the study honest:

    - ``"cap2026"`` — the first ``n`` names of the as-of-2026 published list at their
      (renormalised) published cap weights. The full **look-ahead** basket.
    - ``"eq2011"`` — the first ``n`` names of the January-2011 list, equal weight. The
      contemporaneous control.
    - ``"eq2026"`` — today's names at **equal** weight. Neither headline basket on its
      own identifies the look-ahead, because ``cap2026`` and ``eq2011`` differ in *two*
      things at once (which names, and how they are weighted). ``eq2026`` holds the
      weighting scheme fixed at equal weight so the gap ``eq2026 − eq2011`` is the
      **name-vintage** (look-ahead) term and ``cap2026 − eq2026`` is the separate
      **weighting-scheme** term. Reporting only the sum would credit the look-ahead with
      a cap-weighting effect that has nothing to do with hindsight about membership.

    Weights sum to 1.
    """
    if scheme == "cap2026":
        pairs = SECTORS[fund]["top_2026"][:n]
        w = pd.Series({t: float(x) for t, x in pairs}, dtype=float)
    elif scheme == "eq2026":
        names = [t for t, _ in SECTORS[fund]["top_2026"]][:n]
        w = pd.Series(1.0, index=names, dtype=float)
    elif scheme == "eq2011":
        names = SECTORS[fund]["top_2011"][:n]
        w = pd.Series(1.0, index=names, dtype=float)
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted fee to recover)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 20,
    n_names: int = 10,            # the "top holdings" a DIY investor would buy
    n_tail: int = 40,             # the small names only the fund holds
    tail_weight: float = 0.10,    # share of the fund sitting in that tail
    fee_ann: float = 0.0200,      # the fund's annual fee at signal_strength=1
    factor_vol: float = 0.18,     # annualised sector-factor vol
    idio_vol: float = 0.18,       # annualised single-name idiosyncratic vol
    drift_ann: float = 0.08,
    signal_strength: float = 1.0,  # 1 = full fee to recover, 0 = no fee (the null)
    start: str = "2006-01-03",
    seed: int = 962,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """A one-factor sector world: ``n_names`` big holdings, a small-cap tail, a fund, cash.

    Every name loads on one common sector factor plus its own idiosyncratic noise. The
    ``fund`` column is the cap-weighted total return of *all* ``n_names + n_tail``
    constituents **minus** a daily fee accrual of ``fee_ann * signal_strength``. So:

    - ``signal_strength = 1`` → the fund skims a real fee; a basket of the ``n_names``
      big holdings should out-return it by roughly that fee (net of whatever tracking
      error the omitted tail contributes, which is zero-mean).
    - ``signal_strength = 0`` → no fee at all; the DIY basket must show **no** reliable
      out-performance (the null).

    The defaults plant a **deliberately fat** fee (2%/yr) against a **deliberately thin**
    omitted tail (10% of the fund), because that is the only regime in which the
    detector has any power at all. With a realistic 8 bps fee and the 5-10%/yr tracking
    error the real tape produces, no test of this shape could ever resolve the fee —
    which is itself one of this study's findings, not a defect of the harness.

    Returns ``(prices, truth)``: ``prices`` carries one column per big name
    (``name_00`` …), plus ``fund`` and ``cash``; ``truth`` records the planted fee, the
    basket weights and the tail share. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    f_vol = factor_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    i_vol = idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift_ann / TRADING_DAYS_PER_YEAR

    n_all = n_names + n_tail
    factor = rng.normal(mu, f_vol, n_days)
    betas = np.concatenate([
        rng.uniform(0.85, 1.15, n_names),   # the mega-caps are the sector
        rng.uniform(0.70, 1.40, n_tail),    # the tail is noisier
    ])
    idio = rng.normal(0.0, i_vol, (n_days, n_all))
    log_ret = factor[:, None] * betas[None, :] + idio          # daily log returns
    levels = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    # Cap weights: the big names share (1 - tail_weight), the tail shares the rest.
    w_big = rng.uniform(0.5, 1.5, n_names)
    w_big = w_big / w_big.sum() * (1.0 - tail_weight)
    w_tail = rng.uniform(0.5, 1.5, n_tail)
    w_tail = w_tail / w_tail.sum() * tail_weight
    w_all = np.concatenate([w_big, w_tail])

    simple = np.exp(log_ret) - 1.0
    fee_daily = (fee_ann * signal_strength) / TRADING_DAYS_PER_YEAR
    fund_ret = simple @ w_all - fee_daily
    fund_level = 100.0 * np.cumprod(1.0 + fund_ret)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = 100.0 * np.cumprod(np.full(n_days, cash_daily))

    cols = {f"name_{i:02d}": levels[:, i] for i in range(n_names)}
    cols["fund"] = fund_level
    cols["cash"] = cash_idx
    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))

    weights = pd.Series(w_big / w_big.sum(),
                        index=[f"name_{i:02d}" for i in range(n_names)])
    truth = {
        "signal_strength": signal_strength,
        "fee_ann": fee_ann,
        "planted_fee_ann": fee_ann * signal_strength,
        "tail_weight": tail_weight,
        "n_names": n_names,
        "n_tail": n_tail,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "weights": weights,
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Alias of :func:`synthetic_panel` (the desk's single-tape naming convention)."""
    return synthetic_panel(**kwargs)
