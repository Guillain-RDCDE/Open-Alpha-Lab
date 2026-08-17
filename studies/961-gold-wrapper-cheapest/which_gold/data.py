"""Data layer for Study 961 — Which Gold (five physically-backed gold wrappers, one metal).

Two tapes, one shape (a date-indexed daily close frame):

- ``fetch`` / ``load_prices`` — daily closes from Yahoo! Finance (``yfinance``,
  ``auto_adjust=True``) for the five US physically-backed gold trusts — **GLD, IAU,
  GLDM, SGOL, BAR** — plus a cash proxy (``BIL``) for the excess-of-cash ownership
  race. ``fetch`` touches the network and caches parquet under the shared
  ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache **offline** and
  never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic only), so CI is green on a fresh checkout.

  **Price-only vs total-return.** ``auto_adjust=True`` gives split- and
  distribution-adjusted closes. None of these five grantor trusts has ever paid a
  distribution — they sell bullion to pay the sponsor fee, so the leak shows up in the
  share price itself — which means for them the adjusted close *is* the price close and
  the two labels coincide. ``BIL`` is genuine total return (it distributes monthly) and
  is used only for the excess-of-cash Sharpe race.

- ``load_dollar_volume`` — daily close x volume per wrapper, cached as
  ``dvol_<TICKER>_1d.parquet``. This is the honest counterweight to the fee number: the
  cheapest wrapper is only the right answer if you can actually buy it in size.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. A
  planted world of ``n_funds`` wrappers on one bullion price, each shaved by its own fee,
  each carrying its own AR(1) premium/discount, quoted against a benchmark observed at a
  slightly different strike. The ``signal_strength`` knob collapses the planted fee
  dispersion: at ``signal_strength=0`` every wrapper charges the cohort-average fee, so
  the fee-rank test faces an informative-looking fee sheet with nothing behind it (the
  null); at ``signal_strength=1`` the planted ladder is the real one. Seed is fixed ->
  tests are deterministic.

The measurement problem, and why this tape is a friendly one: all five wrappers hold the
same metal and strike at the same 16:00 New York close, so a wrapper-minus-wrapper
difference cancels the gold price almost exactly and leaves a residual of ~6 bp/day, most
of it a mean-reverting premium/discount. A 30 bp/yr fee gap is 0.12 bp/day — invisible
day by day, and perfectly legible once aggregated to non-overlapping months.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the wrapper
choice formed on data through month ``t`` is acted on at the next session's close).
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
MONTHS_PER_YEAR = 12

# --------------------------------------------------------------------------- #
# The cohort
# --------------------------------------------------------------------------- #
# Five US physically-backed gold trusts. Every one of them holds allocated London bullion
# in a vault, publishes a bar list, and differs from the others in essentially one
# contractual respect: the sponsor fee.
COHORT = ("GLD", "IAU", "GLDM", "SGOL", "BAR")

CASH = "BIL"  # 1-3 month T-bill total return, for the excess-of-cash race
TICKERS = COHORT + (CASH,)

# **Survivorship, named.** These are five wrappers that still exist in 2026. The gold-ETF
# graveyard is real (OUNZ's peers, several currency-hedged and levered bullion notes) and
# closed funds are absent from this panel by construction. That biases the *level* of the
# cohort-mean tracking difference upward; it cannot manufacture the cross-sectional fee
# ordering, which is the claim under test.

# --------------------------------------------------------------------------- #
# The fee sheet — an ASSUMPTION, not tape data
# --------------------------------------------------------------------------- #
# Published sponsor fees in basis points per year, read off the funds' fee schedules at
# build time. THIS IS A PROXY / ASSUMPTION: it is not measured on the tape, it carries
# whatever hindsight today's fee sheet carries, and a sponsor can change it (several of
# these did). It enters ONLY the fee-rank and pass-through tests; the headline
# cheapest-minus-priciest spread is pure tape and never touches it.
FEE_BPS = {
    "GLD": 40.00,    # SPDR Gold Shares — the 2004 original, never cut
    "IAU": 25.00,    # iShares Gold Trust
    "GLDM": 10.00,   # SPDR Gold MiniShares  <- the fee-cheapest of the cohort
    "SGOL": 17.00,   # abrdn Physical Gold Shares
    "BAR": 17.49,    # GraniteShares Gold Trust
}

# In-window fee changes, as (effective date, new fee in bp/yr). Also an ASSUMPTION, and a
# more fragile one than the current sheet: fee-cut dates are announcement-driven and are
# not visible on a price tape. Only one change falls inside the common window — GLDM's cut
# from its 18 bp launch fee — which is exactly why the study also quotes a **fee-stable
# sub-window** (2021-01-01 onward) where today's sheet holds for all five.
FEE_CHANGES = {
    "GLDM": [("2018-06-26", 18.00), ("2020-10-01", 10.00)],
}

# The date from which today's published sheet applies to every one of the five.
FEE_STABLE_FROM = "2021-01-01"

# Study-wide as-of: the last COMPLETE calendar month at build time. Partial months are
# dropped so the monthly estimator never eats a stub and the sample never creeps.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str, kind: str = "prices") -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"{kind}_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
    with_volume: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download daily adjusted closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted. None of the five wrappers
    distributes, so for them adjusted == price; ``BIL`` genuinely compounds its coupon.
    With ``with_volume`` the daily **dollar volume** (close x volume) of each cohort
    member is cached alongside as ``dvol_<TICKER>_1d.parquet`` — the liquidity leg of the
    counterweight.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
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
        if with_volume and tk in COHORT and "volume" in raw.columns:
            dv = (raw["close"] * raw["volume"]).rename("dollar_volume").to_frame()
            dv.index = pd.to_datetime(dv.index)
            dv.index.name = "date"
            dv.dropna().to_parquet(_cache_path(tk, cache_dir, kind="dvol"))
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def have_volume(tickers=COHORT, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cohort member's dollar-volume parquet is present."""
    return all(os.path.exists(_cache_path(tk, cache_dir, kind="dvol")) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so the
    sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call which_gold.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_dollar_volume(
    tickers=COHORT,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily dollar volume OFFLINE into one aligned frame (one column per fund)."""
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, kind="dvol")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached dollar volume for {tk} at {path}. "
                f"Call which_gold.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["dollar_volume"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def cohort_frame(prices: pd.DataFrame, cohort=COHORT) -> pd.DataFrame:
    """The five wrappers on their common window (every column present, no gaps)."""
    return prices[list(cohort)].dropna()


def blended_fee_bps(ticker: str, start, end) -> float:
    """The time-weighted average published fee over ``[start, end]``, in bp/yr — an ASSUMPTION.

    Walks ``FEE_CHANGES`` for the ticker (falling back to the flat ``FEE_BPS`` when it has
    none) and averages the applicable fee day by day. The fee-change dates are announcement
    facts, not tape facts, so every headline that uses them is also quoted on the
    fee-stable sub-window and on rank order alone.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    steps = FEE_CHANGES.get(ticker)
    if not steps or end <= start:
        return float(FEE_BPS[ticker])
    bounds = sorted((pd.Timestamp(d), float(f)) for d, f in steps)
    days = pd.date_range(start, end, freq="D")
    fee = np.full(len(days), bounds[0][1], dtype=float)  # before the first step
    for d0, f in bounds:
        fee[days >= d0] = f
    return float(fee.mean())


def fee_vector(cohort=COHORT, blended_over: tuple | None = None) -> dict:
    """The fee ASSUMPTION as a dict — today's sheet, or time-blended over a window."""
    if blended_over is None:
        return {t: float(FEE_BPS[t]) for t in cohort}
    start, end = blended_over
    return {t: blended_fee_bps(t, start, end) for t in cohort}


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted fee ladder)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: float = 8.0,
    fees_bps=(40.0, 25.0, 10.0, 17.0, 17.5),
    bench_vol: float = 0.16,        # annualised vol of bullion
    bench_drift: float = 0.08,      # annualised drift (irrelevant to a tracking difference)
    strike_stub_bp: float = 8.0,    # daily sd of the benchmark's different-strike mismatch
    premium_innov_bp: float = 4.0,  # daily innovation of each wrapper's premium/discount
    premium_phi: float = 0.55,      # AR(1) persistence of that premium
    signal_strength: float = 1.0,   # 0 = every wrapper charges the cohort-average fee (null)
    start: str = "2018-06-26",
    seed: int = 961,
    cash_rate_ann: float = 0.03,
) -> tuple[pd.DataFrame, dict]:
    """A planted cohort of gold wrappers on one bullion price, each shaved by its own fee.

    Construction, mirroring the real measurement problem:

    - a bullion log price random walk (the *true* metal);
    - each wrapper's log price = bullion - fee_i * t + an AR(1) premium/discount;
    - the **observed** benchmark carries an iid mean-zero ``strike_stub_bp`` per day (a
      London-fix-versus-16:00-New-York mismatch), which cancels out of every
      wrapper-minus-wrapper difference and does not out of a wrapper-minus-benchmark one.

    The ``signal_strength`` knob shrinks the planted fee dispersion toward its mean:

    - ``signal_strength = 1`` -> the full planted ladder; the fee rank *is* the tracking
      difference rank, and the estimators must recover it;
    - ``signal_strength = 0`` -> every wrapper charges the same fee while the published
      sheet still shows dispersion, so the rank test and the pair spread must stay quiet.

    Returns ``(prices, truth)``. ``prices`` has one column per wrapper (``F0``..), a
    ``bench`` column (the *observed*, stubbed bullion price) and a ``cash`` accrual leg.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(round(n_years * TRADING_DAYS_PER_YEAR))
    # OOB-safe: a few thousand business days stays well inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    fees = np.asarray(fees_bps, dtype=float)
    fees_eff = (1.0 - signal_strength) * fees.mean() + signal_strength * fees
    n_funds = fees_eff.size

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    d_bench = (bench_drift - 0.5 * bench_vol ** 2) * dt
    s_bench = bench_vol * np.sqrt(dt)
    true_log = np.cumsum(rng.normal(d_bench, s_bench, n_days)) + np.log(120.0)

    t_years = np.arange(n_days, dtype=float) * dt
    cols = {}
    prem_sd = premium_innov_bp * 1e-4
    for i in range(n_funds):
        eps = rng.normal(0.0, prem_sd, n_days)
        prem = np.empty(n_days)
        prem[0] = eps[0]
        for k in range(1, n_days):
            prem[k] = premium_phi * prem[k - 1] + eps[k]
        cols[f"F{i}"] = np.exp(true_log - fees_eff[i] * 1e-4 * t_years + prem)

    stub = rng.normal(0.0, strike_stub_bp * 1e-4, n_days)
    cols["bench"] = np.exp(true_log + stub)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cols["cash"] = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "fees_bps": fees.tolist(),
        "fees_eff_bps": fees_eff.tolist(),
        "fund_cols": [f"F{i}" for i in range(n_funds)],
        "signal_strength": float(signal_strength),
        "n_funds": int(n_funds),
        "n_days": int(n_days),
        "n_years": float(n_years),
        "bench_vol": float(bench_vol),
        "strike_stub_bp": float(strike_stub_bp),
        "premium_innov_bp": float(premium_innov_bp),
        "premium_phi": float(premium_phi),
        "seed": int(seed),
        "cash_rate_ann": float(cash_rate_ann),
        "fee_spread_bps": float(fees_eff.max() - fees_eff.min()),
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Two-wrapper convenience shim over :func:`synthetic_panel` (cheapest vs priciest)."""
    kwargs.setdefault("fees_bps", (40.0, 10.0))
    return synthetic_panel(**kwargs)
