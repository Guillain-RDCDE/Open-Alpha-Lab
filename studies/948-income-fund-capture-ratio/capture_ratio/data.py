"""Data layer for Study 948 — Capture Ratio (income ETFs vs their beta benchmark).

Two tapes, one shape (monthly simple total returns, benchmark-aligned):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for fourteen US income ETFs, the three equity
  benchmarks they are marketed against (SPY, QQQ, IWM) and a cash leg (BIL, the 1-3
  month T-bill ETF). ``fetch`` touches the network and writes parquet into the shared
  ``studies/_cache`` (retry up to 4×); ``load_prices`` reads that cache **offline** and
  never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic-only), so CI is green on a fresh checkout. ``load_prices`` also runs an
  **unadjusted-corporate-action screen** (``repair_split_artifacts``) — the vendor's
  "adjusted" tape misses NUSI's 2025-02-18 1-for-2 reverse split, which arrives as an
  exact +100% one-day return and, left in, manufactures a fake positive capture spread
  and a fake +186 bps/month alpha for that fund.

- ``synthetic_panel`` / ``synthetic_daily`` — *deterministic, offline* generators. A
  benchmark tape plus a cross-section of funds with a planted piecewise payoff:
  up-beta ``b + s*c`` in benchmark-up months, down-beta ``b - s*c`` in down months, so
  the planted capture spread is exactly ``2*s*c``. The ``signal_strength`` knob ``s``
  blends the world toward the null: at ``s = 0`` every fund is plain linear beta and the
  true capture spread of every fund is **zero**; at ``s = 1`` one fund is genuinely
  convex, several are genuinely concave. Seeds are fixed → tests are deterministic.

The claim under test is the income-fund sales pitch: a covered-call or high-dividend
fund "gives up a little upside for a lot of downside protection", i.e. it should capture
*more* of the benchmark's up months than of its down months. That is one number — the
**capture spread**, up-capture minus down-capture — and it is the number the industry's
own scorecard implies but never quotes with an interval. (Whether it is a *good* estimator
of payoff shape is itself one of the study's questions; see ``strategy.py``, where a fund
with a positive alpha and no curvature at all turns out to read "convex".) This module
supplies the tape; the measurement, the inference and the (single) execution lag live in
``strategy.py``.

**Non-tape inputs are assumptions, and are labelled as such.** The fund → benchmark map
below is hand-assigned from each fund's own stated reference index; it is an ASSUMPTION
and is swept in ``examples/verify.py`` (everything re-run with SPY for every fund). BIL
is a PROXY for cash. Borrow and trading costs enter only the traded arm in
``strategy.py``, and are swept there.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

MONTHS_PER_YEAR = 12
TRADING_DAYS_PER_YEAR = 252

# ASSUMPTION — the fund → benchmark map. Each fund is paired with the index it is
# *marketed against* (the one its own literature names), not with a fitted best-R²
# benchmark: pairing a fund with whatever index happens to explain it best would let the
# analyst choose the answer. Swept in verify.py by re-running with SPY for every fund.
FUND_BENCH = {
    # NASDAQ-100 option-writers
    "QYLD": "QQQ",   # Global X NASDAQ-100 Covered Call
    "JEPQ": "QQQ",   # JPMorgan NASDAQ Equity Premium Income
    "NUSI": "QQQ",   # Nationwide NASDAQ-100 Risk-Managed Income (net-credit collar)
    # S&P 500 option-writers
    "XYLD": "SPY",   # Global X S&P 500 Covered Call
    "JEPI": "SPY",   # JPMorgan Equity Premium Income
    "PBP": "SPY",    # Invesco S&P 500 BuyWrite
    "DIVO": "SPY",   # Amplify CWP Enhanced Dividend Income
    "SPYI": "SPY",   # NEOS S&P 500 High Income
    # S&P 500 dividend-tilt funds (no options written — the control group)
    "SCHD": "SPY",   # Schwab US Dividend Equity
    "VYM": "SPY",    # Vanguard High Dividend Yield
    "DVY": "SPY",    # iShares Select Dividend
    "SPHD": "SPY",   # Invesco S&P 500 High Dividend Low Volatility
    "NOBL": "SPY",   # ProShares S&P 500 Dividend Aristocrats
    # Russell 2000 option-writer
    "RYLD": "IWM",   # Global X Russell 2000 Covered Call
}

# The option-writing subset — the funds whose payoff is *designed* to be concave.
OPTION_WRITERS = ("QYLD", "JEPQ", "NUSI", "XYLD", "JEPI", "PBP", "DIVO", "SPYI", "RYLD")

FUNDS = tuple(FUND_BENCH)
BENCHES = ("SPY", "QQQ", "IWM")
CASH = "BIL"          # PROXY for the risk-free leg (1-3 month T-bill total return)
TICKERS = FUNDS + BENCHES + (CASH,)

# Study-wide as-of: the last COMPLETE calendar month at build time. The partial current
# month is dropped so the sample never creeps between reruns.
AS_OF = "2026-06-30"

# --- Unadjusted-corporate-action screen -------------------------------------
# ``auto_adjust=True`` is supposed to back-adjust splits, but the vendor tape does miss
# them (NUSI's 1-for-2 reverse split of 2025-02-18 arrives as an exact +100.0000% "return").
# A one-day price ratio that lands on a real share-count factor is a corporate action,
# not a return, and leaving it in manufactures a fake up-month and a fake alpha.
#
# The factor list is a deliberately SHORT whitelist of the ratios ETFs actually split on
# (3:2, 2:1, 3:1, 4:1, 5:1, 10:1 and their reverse twins). A long list of every n/m would
# tile the number line so densely that a genuine −40% day would "match" something and get
# smoothed away — the screen must be able to say *no*. Anything past the jump threshold
# that matches nothing is left in the tape and flagged for the reader.
#
# The 3% tolerance is wide enough that a split day carrying its own genuine ±3% move is
# still recognised, and — because the whitelist is sparse (the tightest neighbouring pair,
# 1.5 and 2.0, is 33% apart) — still narrow enough that no ordinary return can land on one.
SPLIT_JUMP_THRESHOLD = 0.35
SPLIT_RATIO_TOL = 0.03
SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0,
                 1.0 / 1.5, 0.5, 1.0 / 3.0, 0.25, 0.2, 0.1)


def _split_factor(ratio: float) -> float:
    """Nearest whitelisted share-count factor to ``ratio``, or ``nan`` if none is close.

    Relative tolerance ``SPLIT_RATIO_TOL``. A factor of 1 is never returned (nothing to
    adjust), so an ordinary large move cannot be mistaken for a corporate action.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return float("nan")
    best, best_err = float("nan"), np.inf
    for f in SPLIT_FACTORS:
        err = abs(ratio / f - 1.0)
        if err < best_err:
            best, best_err = f, err
    return best if best_err <= SPLIT_RATIO_TOL else float("nan")


def split_artifacts(prices: pd.DataFrame) -> list:
    """Every unadjusted-split suspect in a daily close frame, as a list of dicts.

    One entry per (ticker, date) whose one-day price ratio exceeds the jump threshold AND
    lands on a simple share-count fraction. Reported, never silently swallowed.
    """
    out = []
    for col in prices.columns:
        s = prices[col].dropna()
        ratio = (s / s.shift(1)).dropna()
        for dt, r in ratio[(ratio - 1.0).abs() > SPLIT_JUMP_THRESHOLD].items():
            f = _split_factor(float(r))
            out.append({"ticker": col, "date": pd.Timestamp(dt), "ratio": float(r),
                        "factor": f, "repairable": bool(np.isfinite(f))})
    return out


def repair_split_artifacts(prices: pd.DataFrame) -> tuple:
    """Back-adjust unadjusted share-count events. Returns ``(repaired_frame, log)``.

    For a jump of factor ``f`` on date ``d``, every close **strictly before** ``d`` is
    multiplied by ``f`` so the level series becomes continuous and the return on ``d``
    collapses to the day's genuine move. Jumps that do not land on a simple fraction are
    left untouched and flagged ``repairable = False`` in the log — a real crash is a real
    return and this function is not licensed to smooth it away.
    """
    log = split_artifacts(prices)
    if not log:
        return prices, log
    out = prices.copy()
    for ev in log:
        if not ev["repairable"]:
            continue
        col, dt, f = ev["ticker"], ev["date"], ev["factor"]
        mask = out.index < dt
        out.loc[mask, col] = out.loc[mask, col] * f
    return out, log


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
    ``close`` column is split- and **distribution-adjusted total return** — non-negotiable
    here, because an income fund pays out most of its return as distributions. On a
    price-only tape QYLD looks like a −40% disaster and every capture ratio is nonsense.
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
    repair_splits: bool = True,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    One column per ticker, sliced to ``asof`` so the sample never creeps. Columns are
    ragged by construction (SPYI starts in 2022, DVY in 2003) — every downstream
    routine intersects per fund rather than dropping to the shortest common window,
    which would throw away eighteen years of PBP to accommodate a 2022 launch.
    Raises ``FileNotFoundError`` if any ticker is missing: the offline core and the
    test-suite never touch the network.

    ``repair_splits=True`` (the default) runs :func:`repair_split_artifacts` over the
    frame, because the vendor's adjusted tape misses at least one share-count event in
    this universe. Pass ``False`` to reproduce the raw, contaminated tape — the study
    quotes both.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call capture_ratio.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    if repair_splits:
        df, _ = repair_split_artifacts(df)
    return df


def to_monthly_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Daily total-return closes → **monthly simple total returns**, partial month dropped.

    Capture ratios are a monthly measure by convention (Morningstar computes them on
    monthly returns), and monthly sampling is what keeps the up/down partition stable:
    on daily data roughly half the observations are "down days" and the ratios are
    dominated by microstructure noise. The last month is kept only if ``asof`` is a
    month end, which the study-wide as-of always is.
    """
    m = prices.resample("ME").last()
    # Resample labels are month *ends*, so an incomplete trailing month carries a label
    # strictly after ``asof`` and is dropped by this one comparison.
    m = m[m.index <= pd.Timestamp(asof)]
    return m.pct_change()


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted capture spread)
# --------------------------------------------------------------------------- #
# The planted cross-section. ``beta`` is the average of the two half-betas; ``convex`` is
# the half-spread ``c``, so the true capture spread is ``2 * signal_strength * c``.
# One genuinely convex fund, three genuinely concave ones (the option-writer caricature),
# the rest flat. ``alpha_bps`` is a monthly intercept in basis points.
PLANTED_FUNDS = (
    #  name,      beta,  convex,  alpha_bps
    ("CONVEX",    0.60,   0.15,     0.0),
    ("FLAT_A",    0.80,   0.00,     0.0),
    ("FLAT_B",    0.55,   0.00,     0.0),
    ("FLAT_C",    0.90,   0.00,     0.0),
    ("CONCAVE_A", 0.55,  -0.15,    35.0),
    ("CONCAVE_B", 0.70,  -0.10,    25.0),
    ("CONCAVE_C", 0.65,  -0.20,    45.0),
)


def synthetic_panel(
    n_months: int = 240,
    signal_strength: float = 1.0,
    bench_mu_ann: float = 0.08,
    bench_vol_ann: float = 0.16,
    cash_rate_ann: float = 0.02,
    idio_vol_ann: float = 0.05,
    start: str = "2006-01",
    seed: int = 948,
) -> tuple:
    """A monthly panel: one benchmark, one cash leg, and a cross-section of funds.

    Each fund's excess return is a **piecewise-linear** function of the benchmark's
    excess return, kinked at zero::

        e_fund = alpha + (beta + s*c) * max(e_bench, 0) + (beta - s*c) * min(e_bench, 0) + eps

    so the fund's *true* up-capture is ``beta + s*c``, its true down-capture is
    ``beta - s*c``, and its true **capture spread** is ``2*s*c``. With
    ``signal_strength = 0`` every ``c`` is switched off: all funds are plain linear beta
    and every true capture spread is exactly zero (the null the machinery must stay quiet
    on). With ``signal_strength = 1`` one fund is genuinely convex, three genuinely
    concave, three flat — a planted cross-section the scorecard must rank correctly.

    Returns ``(returns, truth)`` where ``returns`` is a month-end-indexed frame of simple
    **total** returns with columns ``BENCH``, ``CASH`` and one per planted fund, and
    ``truth`` records the planted parameters (including the exact capture spreads).
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    s = float(signal_strength)
    # period_range → to_timestamp keeps us far inside pandas' ns Timestamp horizon even
    # for large n_months (a plain date_range with a huge periods= overflows on pandas 2.x).
    idx = pd.period_range(start=start, periods=n_months, freq="M").to_timestamp(how="end")
    idx = pd.DatetimeIndex(idx.normalize(), name="date")

    r_cash = np.full(n_months, (1.0 + cash_rate_ann) ** (1.0 / MONTHS_PER_YEAR) - 1.0)
    e_bench = rng.normal(bench_mu_ann / MONTHS_PER_YEAR,
                         bench_vol_ann / np.sqrt(MONTHS_PER_YEAR), n_months)

    up = np.maximum(e_bench, 0.0)
    dn = np.minimum(e_bench, 0.0)
    idio_sd = idio_vol_ann / np.sqrt(MONTHS_PER_YEAR)

    cols = {"BENCH": r_cash + e_bench, "CASH": r_cash}
    planted = {}
    for name, beta, convex, alpha_bps in PLANTED_FUNDS:
        b_up = beta + s * convex
        b_dn = beta - s * convex
        eps = rng.normal(0.0, idio_sd, n_months)
        e_fund = alpha_bps * 1e-4 + b_up * up + b_dn * dn + eps
        cols[name] = r_cash + e_fund
        planted[name] = {
            "beta": beta, "b_up": b_up, "b_dn": b_dn,
            "true_spread": b_up - b_dn, "alpha_bps": alpha_bps,
        }

    returns = pd.DataFrame(cols, index=idx)
    truth = {
        "signal_strength": s,
        "n_months": n_months,
        "seed": seed,
        "bench_mu_ann": bench_mu_ann,
        "bench_vol_ann": bench_vol_ann,
        "cash_rate_ann": cash_rate_ann,
        "funds": planted,
        "fund_names": [n for n, _, _, _ in PLANTED_FUNDS],
        "bench_map": {n: "BENCH" for n, _, _, _ in PLANTED_FUNDS},
    }
    return returns, truth


def synthetic_daily(
    n_years: int = 20,
    signal_strength: float = 1.0,
    beta: float = 0.60,
    convex: float = 0.15,
    alpha_bps_month: float = 0.0,
    bench_mu_ann: float = 0.08,
    bench_vol_ann: float = 0.16,
    cash_rate_ann: float = 0.02,
    idio_vol_ann: float = 0.05,
    start: str = "2006-01-02",
    seed: int = 948,
) -> tuple:
    """A single fund/benchmark/cash trio as **daily total-return price levels**.

    Same piecewise construction as :func:`synthetic_panel`, but kinked on the *monthly*
    benchmark move and then spread across that month's business days, and returned as
    price *levels* so the daily → monthly resampling path in
    :func:`to_monthly_returns` is exercised end-to-end offline. Columns: ``FUND``,
    ``BENCH``, ``CASH``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    s = float(signal_strength)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    r_cash_d = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    e_bench_d = rng.normal(bench_mu_ann / TRADING_DAYS_PER_YEAR,
                           bench_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    e_bench = pd.Series(e_bench_d, index=dates)

    # Month key → that month's cumulated benchmark excess move decides which half-beta
    # applies to every day of the month (a payoff kinked on the monthly, not daily, move).
    key = e_bench.index.to_period("M")
    monthly_sum = e_bench.groupby(key).transform("sum")
    b_up, b_dn = beta + s * convex, beta - s * convex
    beta_d = np.where(monthly_sum.to_numpy() >= 0.0, b_up, b_dn)

    n_months_eff = max(len(set(key)), 1)
    alpha_d = alpha_bps_month * 1e-4 * n_months_eff / n_days
    eps = rng.normal(0.0, idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    e_fund_d = alpha_d + beta_d * e_bench_d + eps

    prices = pd.DataFrame(
        {
            "FUND": 100.0 * np.cumprod(1.0 + r_cash_d + e_fund_d),
            "BENCH": 100.0 * np.cumprod(1.0 + r_cash_d + e_bench_d),
            "CASH": 100.0 * np.cumprod(np.full(n_days, 1.0 + r_cash_d)),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": s, "beta": beta, "convex": convex,
        "b_up": b_up, "b_dn": b_dn, "true_spread": b_up - b_dn,
        "n_days": n_days, "n_years": n_years, "seed": seed,
        "alpha_bps_month": alpha_bps_month,
    }
    return prices, truth
