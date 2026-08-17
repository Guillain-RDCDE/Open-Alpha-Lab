"""Data layer for Study 931 — the closed-end-fund IPO hole.

Three tapes, one shape (date-indexed daily closes):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for a hardcoded list of closed-end funds that
  came to market between 2012 and 2022, plus one asset-class benchmark ETF per fund.
  BIL is cached alongside them as the desk's cash reference but is **deliberately never
  read**: every number in this study is a difference of two fully invested legs or a
  self-financing spread, so the cash rate cancels on both sides and there is no funded /
  unfunded Sharpe race to correct. ``fetch`` touches the network and caches parquet under the
  **shared** ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache
  **offline** and never imports yfinance.

- ``load_raw_closes`` — the same funds' **unadjusted** closes, cached under the
  ``rawclose_`` prefix. They are used for exactly one thing: the day-one gap between
  the hardcoded **offering price** and the first traded close, so the study can price
  the return of a genuine IPO *subscriber* (who pays the offering price) rather than
  the return of a buyer at the day-one close. Nothing else reads them.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator of
  a fund/benchmark panel with a planted post-IPO decay. The ``signal_strength`` knob
  scales the planted decay: at ``0`` funds track their benchmark (the null — the
  detector must stay quiet), at ``1`` a full ~10%/year first-year slide is planted.

The whole test-suite runs with NO cache present (synthetic only), so CI is green on a
fresh checkout.

**Assumptions and proxies, named here and in the README.**

- ``offer`` (the IPO offering price, $20 or $25) is a hardcoded ASSUMPTION. It is
  validated against the tape: for 21 of the 28 funds the first traded close lands
  within 1% of it (worst gap -7.2%, RIV), which is what a syndicate-supported CEF IPO
  looks like. The whole study is re-run against a day-one-close entry as a sweep.
- The underwriting load (~3-6% of the offering price) is quoted for context and is
  **never subtracted anywhere** — the price tape already contains its consequences.
  Subtracting it would double-count.
- The benchmark mapping (fund -> asset-class ETF) is a hand-made ASSUMPTION; the
  study re-runs the headline under an all-equity and an all-60/40 mapping as a sweep.
  Subtracting *one* unit of that benchmark is a second assumption (beta = 1), stressed
  by ``strategy.beta_adjusted_table``.
- **The universe is a hand-built convenience sample, not a census.** These 28 tickers
  are large, well-known US closed-end fund launches that Yahoo still carries under
  their IPO ticker. Several hundred closed-end funds came to market in this window;
  no exhaustive listing-date screen is available on a free tape, so this list cannot
  be shown to be representative from inside the study. Two vintages, **2017 and 2018,
  contribute nothing at all** — the gap is visible in the vintage counts and is a
  property of the sample, not of the market.
- **Named exclusions**, both decided on data grounds before any result was computed:
  ``BIGZ`` (2021 vintage) left the tape when BlackRock converted it to an ETF in 2025
  and yfinance returns nothing for it; ``XFLT`` (2017 vintage — the only one available
  to us) is dropped because Yahoo's split factors for it are visibly corrupt (a raw
  first close of $54 for a $10 issue). Dropping a delisted fund is **survivorship**: a
  list of funds that still trade under their IPO ticker biases the measured decay
  *toward zero*, because the funds that fared worst are the ones that get merged,
  converted or wound up.
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

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# The hardcoded CEF IPO list
# --------------------------------------------------------------------------- #
# (ticker, offering price [ASSUMPTION], sleeve, benchmark ETF [ASSUMPTION])
# Offering prices are the standard US CEF syndicate prices: $20.00 for equity and
# multi-sector credit funds, $25.00 for preferred-share funds. Vintages 2012-2022.
CEF_IPOS: tuple[tuple[str, float, str, str], ...] = (
    # --- equity / thematic ---
    ("BST",  20.0, "equity-tech",     "XLK"),
    ("BSTZ", 20.0, "equity-tech",     "XLK"),
    ("NBXG", 20.0, "equity-tech",     "XLK"),
    ("AIO",  20.0, "equity-tech",     "XLK"),
    ("BMEZ", 20.0, "equity-health",   "XLV"),
    ("THQ",  20.0, "equity-health",   "XLV"),
    ("THW",  20.0, "equity-health",   "XLV"),
    ("ASGI", 20.0, "infrastructure",  "IGF"),
    ("MEGI", 20.0, "infrastructure",  "IGF"),
    # --- multi-asset / allocation ---
    ("BCAT", 20.0, "multi-asset",     "AOR"),
    ("ECAT", 20.0, "multi-asset",     "AOR"),
    ("RIV",  20.0, "multi-asset",     "AOR"),
    # --- leveraged loans ---
    ("BGB",  20.0, "loans",           "BKLN"),
    ("ARDC", 20.0, "loans",           "BKLN"),
    # --- high yield / multi-sector credit ---
    ("KIO",  20.0, "credit",          "HYG"),
    ("ECC",  20.0, "credit",          "HYG"),
    ("EIC",  20.0, "credit",          "HYG"),
    ("OPP",  20.0, "credit",          "HYG"),
    ("FTHY", 20.0, "credit",          "HYG"),
    ("WDI",  20.0, "credit",          "HYG"),
    ("PDO",  20.0, "credit",          "HYG"),
    ("PAXS", 20.0, "credit",          "HYG"),
    ("PDI",  25.0, "credit",          "HYG"),
    # --- investment-grade / financial credit ---
    ("FINS", 20.0, "ig-credit",       "LQD"),
    # --- preferred shares ---
    ("LDP",  25.0, "preferred",       "PFF"),
    ("DFP",  25.0, "preferred",       "PFF"),
    ("PTA",  25.0, "preferred",       "PFF"),
    ("NPFD", 25.0, "preferred",       "PFF"),
)

FUND_TICKERS = tuple(t for t, _, _, _ in CEF_IPOS)
BENCH_TICKERS = ("XLK", "XLV", "IGF", "AOR", "BKLN", "HYG", "LQD", "PFF", "SPY", "BIL")
TICKERS = FUND_TICKERS + BENCH_TICKERS

OFFER = {t: p for t, p, _, _ in CEF_IPOS}
SLEEVE = {t: s for t, _, s, _ in CEF_IPOS}
BENCH = {t: b for t, _, _, b in CEF_IPOS}

# Alternative benchmark mappings, for the mapping sweep (an ASSUMPTION being stressed).
ALT_MAPPINGS = {
    "asset-class": BENCH,
    "all-equity": {t: "SPY" for t in FUND_TICKERS},
    "all-60-40": {t: "AOR" for t in FUND_TICKERS},
}

# The underwriting load band quoted in CEF prospectuses (a context PROXY only —
# never subtracted from any return in this study).
LOAD_BAND_PCT = (3.0, 4.5, 6.0)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, cache-only by default
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"prices_{_safe(ticker)}_1d.parquet")


def _raw_cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"rawclose_{_safe(ticker)}_1d.parquet")


def _download(tk: str, start: str, end: str | None, adjust: bool, retries: int):
    import yfinance as yf  # lazy: only when we actually go to the network

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tk, start=start, end=end, interval="1d",
                              auto_adjust=adjust, progress=False)
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
    return df.dropna(subset=["close"])


def fetch(
    tickers=TICKERS,
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache them as parquet.

    Writes ``prices_<TK>_1d.parquet`` (total return, ``auto_adjust=True``) for every
    ticker and, for the funds only, ``rawclose_<TK>_1d.parquet`` (unadjusted close,
    used solely for the day-one offering-price gap). Network-only; run once.
    """
    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        df = _download(tk, start, end, True, retries)
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
        if tk in FUND_TICKERS:
            rawdf = _download(tk, start, end, False, retries)
            rawdf.to_parquet(_raw_cache_path(tk, cache_dir))
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's total-return parquet (and each fund's raw close) is cached."""
    ok = all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)
    return ok and all(os.path.exists(_raw_cache_path(tk, cache_dir)) for tk in FUND_TICKERS)


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
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call cef_ipo.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_raw_closes(
    tickers=FUND_TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached **unadjusted** fund closes OFFLINE (day-one offering-price gap only)."""
    cols = {}
    for tk in tickers:
        path = _raw_cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached raw closes for {tk} at {path}. Call cef_ipo.data.fetch()."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp.

    Hashed on daily **returns**, not levels, and rounded to 1e-9. Yahoo rescales an
    adjusted-close history retroactively every time a fund pays a distribution, which
    changes every level in the file by a constant factor while changing no return and
    no number in this study. A level-based stamp therefore churns for reasons that are
    not about the data the study used; a return-based one moves only when a return moves.
    """
    r = prices.astype(float).pct_change(fill_method=None)
    arr = np.ascontiguousarray(np.round(r.to_numpy(dtype=float), 9))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted post-IPO decay)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_days: int = 800,
    decay_pct_year1: float = 10.0,
    bench_drift: float = 0.07,
    bench_vol: float = 0.14,
    idio_vol: float = 0.10,
    beta: float = 1.0,
    signal_strength: float = 1.0,
    start: str = "2015-01-02",
    seed: int = 931,
) -> tuple[pd.DataFrame, dict]:
    """One synthetic (fund, benchmark) pair with a planted first-year decay.

    The fund is the benchmark times ``beta`` plus idiosyncratic noise, minus a
    *planted* linear slide of ``signal_strength * decay_pct_year1`` percent spread over
    the first 252 trading days and flat thereafter — a caricature of the price drifting
    from the offering price down to a seasoned discount. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=int(n_days))

    b_d = bench_drift / TRADING_DAYS_PER_YEAR
    b_s = bench_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    i_s = idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    r_bench = rng.normal(b_d, b_s, int(n_days))
    r_idio = rng.normal(0.0, i_s, int(n_days))

    slide_total = signal_strength * decay_pct_year1 / 100.0
    year1 = min(TRADING_DAYS_PER_YEAR, int(n_days))
    slide = np.zeros(int(n_days))
    slide[:year1] = -slide_total / year1

    r_fund = beta * r_bench + r_idio + slide
    prices = pd.DataFrame(
        {
            "fund": 20.0 * np.exp(np.cumsum(r_fund)),
            "bench": 100.0 * np.exp(np.cumsum(r_bench)),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "decay_pct_year1": decay_pct_year1,
        "planted_slide": slide_total,
        "beta": beta,
        "n_days": int(n_days),
        "seed": seed,
    }
    return prices, truth


def synthetic_panel(
    n_funds: int = 28,
    n_days: int = 800,
    decay_pct_year1: float = 10.0,
    signal_strength: float = 1.0,
    stagger_days: int = 40,
    start: str = "2013-01-02",
    beta: float = 1.0,
    seed: int = 931,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A panel of ``n_funds`` staggered synthetic IPOs, each with its own benchmark.

    Fund *i* starts ``i * stagger_days`` business days after ``start``, mirroring the
    real list's staggered vintages, and carries the same planted first-year slide. Each
    entry of the returned dict is a ``(fund, bench)`` frame keyed ``F00, F01, ...``.
    ``signal_strength = 0`` plants nothing (the null the detector must stay quiet on).

    ``beta`` plants **leverage**: at ``beta = 1.5`` with nothing else planted, every fund
    is 1.5x its benchmark and drifts *above* it in a rising market, so a beta-1 abnormal
    return is biased even though no IPO effect exists. That is the world
    ``strategy.beta_adjusted_table`` has to survive, and the test-suite makes it.
    """
    panel: dict[str, pd.DataFrame] = {}
    base = pd.Timestamp(start)
    for i in range(int(n_funds)):
        s = (base + pd.tseries.offsets.BDay(i * stagger_days)).strftime("%Y-%m-%d")
        px, _ = synthetic_daily(
            n_days=n_days, decay_pct_year1=decay_pct_year1,
            signal_strength=signal_strength, start=s, beta=beta, seed=seed + i,
        )
        panel[f"F{i:02d}"] = px
    truth = {
        "n_funds": int(n_funds),
        "signal_strength": signal_strength,
        "decay_pct_year1": decay_pct_year1,
        "planted_slide": signal_strength * decay_pct_year1 / 100.0,
        "stagger_days": int(stagger_days),
        "beta": float(beta),
        "n_days": int(n_days),
        "seed": seed,
    }
    return panel, truth
