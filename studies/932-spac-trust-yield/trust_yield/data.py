"""Data layer for Study 932 — the pre-deal SPAC trust yield.

Three tapes, one shape (date-indexed daily closes):

- ``fetch`` / ``load_spac_closes`` — daily **unadjusted** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=False``, cached under the ``rawclose_`` prefix) for a
  hardcoded list of 2020-2022 vintage SPACs. A pre-deal SPAC is a $10 trust account
  with a redemption right; its price only means anything **in dollars**, so the split-
  and dividend-adjusted total-return series is the wrong tape here — a post-deal
  reverse split would rescale the pre-deal $10 into $200 and destroy the comparison.
  Each SPAC is read off its **successor ticker**, because Yahoo carries the pre-deal
  SPAC history forward under the de-SPACed company's symbol.

- ``load_cash`` — daily **total-return** closes (``auto_adjust=True``, ``prices_``
  prefix) for BIL (the 1-3 month T-bill ETF, the cash leg and the trust-accrual proxy),
  SGOV (a second bill ETF, cross-check) and ^IRX (the 13-week bill discount rate, for
  the yield chart only).

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator of a
  pre-deal SPAC panel with a planted below-trust discount. The ``signal_strength`` knob
  scales the plant: at ``0`` prices oscillate around trust with zero mean discount (the
  null — the detector must stay quiet), at ``1`` a full persistent discount is planted.

The whole test-suite runs with NO cache present (synthetic only), so CI is green on a
fresh checkout.

**Assumptions and proxies, named here and in the README.**

- ``TRUST0 = $10.00`` per share at IPO is a hardcoded ASSUMPTION. It is the standard US
  SPAC trust funding for the 2020-2021 vintage; a minority of deals were over-funded at
  $10.10-$10.20. The headline is swept over $9.90/$10.00/$10.10/$10.20.
- The **trust accrual** is a PROXY: the trust is assumed to earn BIL's total return from
  the first day of the SPAC's tape, less an optional fee/tax drag (swept 0/10/25 bps a
  year). Real trusts hold T-bills or a government money-market fund, so this is close,
  but it is not the filed figure.
- The **deal close dates** are hardcoded ASSUMPTIONS (public de-SPAC calendar). The
  redemption *election deadline* sits days-to-weeks before the close, so the study exits
  at ``close - BUFFER_DAYS`` (default 30 calendar days) and sweeps the buffer. This
  matters: several names in the list de-anchor violently from trust in the fortnight
  before closing, precisely because the redemption put has already expired.
- Deadline **extensions** are not modelled. Every SPAC here is exited at its (buffered)
  *final* close date. Holders could in reality have redeemed at an earlier extension
  vote, so the modelled holding period is an upper bound and the modelled annualised
  yield a lower bound — but the modelled *duration risk* is an upper bound too.
- The list is a **survivorship-shaped** one: it contains SPACs whose successor ticker
  still trades, so it under-represents liquidations and delistings, and it excludes
  every name that did a post-deal reverse split (their pre-deal dollar tape is not
  recoverable from Yahoo). Named again on the Signal axis of the README.
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
DAYS_PER_YEAR = 365.25

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# The hardcoded SPAC list
# --------------------------------------------------------------------------- #
# (successor ticker on Yahoo, SPAC symbol as it traded, deal close date [ASSUMPTION])
#
# The pre-deal window for each name is read off the successor ticker's own history,
# which begins at the SPAC's listing. Only names whose pre-deal dollar tape survives
# intact (no post-deal reverse split) can be used — see the survivorship note above.
SPACS: tuple[tuple[str, str, str], ...] = (
    # --- 2021 closes: the zero-rate era, when trust yield was ~0 ---
    ("CLOV", "IPOC", "2021-01-07"),
    ("HIMS", "OAC",  "2021-01-20"),
    ("BFLY", "LGVW", "2021-02-12"),
    ("ASTS", "NPA",  "2021-04-06"),
    ("SKIN", "FEAC", "2021-06-08"),
    ("NAUT", "ARYA", "2021-06-09"),
    ("MVST", "THCB", "2021-07-23"),
    ("JOBY", "RTP",  "2021-08-10"),
    ("RKLB", "VACQ", "2021-08-25"),
    ("CIFR", "GWAC", "2021-08-27"),
    ("RDW",  "GNPK", "2021-09-02"),
    ("IONQ", "DMYI", "2021-09-30"),
    ("MIR",  "GSAH", "2021-10-20"),
    ("NN",   "TMTS", "2021-10-28"),
    ("AUR",  "RTPY", "2021-11-03"),
    ("GRAB", "AGC",  "2021-12-01"),
    ("PL",   "DMYQ", "2021-12-07"),
    ("SLDP", "DCRC", "2021-12-08"),
    # --- 2022 closes: the hiking cycle, when the discount opened up ---
    ("SATL", "CFV",  "2022-01-25"),
    ("SGHC", "SEAH", "2022-01-27"),
    ("SES",  "IVAN", "2022-02-03"),
    ("SYM",  "SVFB", "2022-06-07"),
    ("GETY", "CCNB", "2022-07-22"),
    ("RUM",  "CFVI", "2022-09-16"),
    # --- 2023-2024 closes: the long tail of extended, deeply-redeemed shells ---
    ("ALTI", "CFFE", "2023-01-03"),
    ("LUNR", "IPAX", "2023-02-13"),
    ("NPWR", "XPDB", "2023-06-08"),
    ("WEST", "AACT", "2023-06-30"),
    ("CXAI", "CXAC", "2023-07-25"),
    ("DJT",  "DWAC", "2024-03-26"),
    ("OKLO", "ALCC", "2024-05-10"),
)

SPAC_TICKERS = tuple(t for t, _, _ in SPACS)
SPAC_SYMBOL = {t: s for t, s, _ in SPACS}
DEAL_CLOSE = {t: d for t, _, d in SPACS}

CASH_TICKERS = ("BIL", "SGOV", "^IRX")

# --------------------------------------------------------------------------- #
# The non-tape assumptions (all swept in strategy.py)
# --------------------------------------------------------------------------- #
TRUST0 = 10.00          # ASSUMPTION: trust per share at IPO, USD
TRUST0_GRID = (9.90, 10.00, 10.10, 10.20)
BUFFER_DAYS = 30        # ASSUMPTION: redemption election lands this far before the close
BUFFER_GRID = (15, 30, 60, 90)
TRUST_FEE_BPS = 0.0     # ASSUMPTION: annual drag on trust accrual (fees / franchise tax)
TRUST_FEE_GRID = (0.0, 10.0, 25.0)
# Quotes implying a discount deeper than this are treated as suspect (a stale print, a
# wide two-sided market, or a redemption deadline that has already quietly passed) and
# are not tradable. Swept — dropping the guard is the aggressive case.
MAX_DISCOUNT = 0.12
MAX_DISCOUNT_GRID = (0.06, 0.12, 0.25, 1.00)
# A redeemable trust share cannot rationally trade at a fraction of its trust; a quote
# under this fraction of $10 inside the pre-deal window is a bad print, not a market.
PRINT_FLOOR_FRAC = 0.60


# --------------------------------------------------------------------------- #
# Real tape — cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str, prefix: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"{prefix}_{safe}_1d.parquet")


def _download(tk: str, start: str, end, auto_adjust: bool, retries: int) -> pd.DataFrame:
    import yfinance as yf  # lazy: only when we actually go to the network

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(
                tk, start=start, end=end, interval="1d",
                auto_adjust=auto_adjust, progress=False, threads=False, actions=False,
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
    return df.dropna(subset=["close"])


def fetch(
    spac_tickers=SPAC_TICKERS,
    cash_tickers=CASH_TICKERS,
    start: str = "2019-01-01",
    cash_start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> None:
    """Download and cache both tapes. Network-only; run once to populate the cache.

    SPAC legs are cached **unadjusted** (``rawclose_``) because dollars are the unit of
    account against a $10 trust; cash legs are cached **total-return**
    (``auto_adjust=True``, ``prices_``) because the cash race must credit the coupon.
    The cash legs live in the **shared** desk cache and other studies read them, so they
    are fetched from ``cash_start`` (2004) rather than this study's own 2019 window — a
    re-run must never shorten a file somebody else depends on.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for tk in spac_tickers:
        df = _download(tk, start, end, auto_adjust=False, retries=retries)
        df.to_parquet(_cache_path(tk, cache_dir, "rawclose"))
    for tk in cash_tickers:
        df = _download(tk, cash_start, end, auto_adjust=True, retries=retries)
        df.to_parquet(_cache_path(tk, cache_dir, "prices"))


def have_real(spac_tickers=SPAC_TICKERS, cash_tickers=CASH_TICKERS,
              cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every leg's parquet is present in the shared cache (offline-testable)."""
    ok = all(os.path.exists(_cache_path(t, cache_dir, "rawclose")) for t in spac_tickers)
    return ok and all(os.path.exists(_cache_path(t, cache_dir, "prices")) for t in cash_tickers)


def load_spac_closes(spac_tickers=SPAC_TICKERS, cache_dir: str = DEFAULT_CACHE,
                     asof: str = AS_OF) -> pd.DataFrame:
    """Read cached **unadjusted** SPAC closes OFFLINE into one aligned frame."""
    cols = {}
    for tk in spac_tickers:
        path = _cache_path(tk, cache_dir, "rawclose")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached raw closes for {tk} at {path}. "
                f"Call trust_yield.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_cash(cash_tickers=CASH_TICKERS, cache_dir: str = DEFAULT_CACHE,
              asof: str = AS_OF) -> pd.DataFrame:
    """Read cached **total-return** cash-leg closes OFFLINE (BIL, SGOV, ^IRX)."""
    cols = {}
    for tk in cash_tickers:
        path = _cache_path(tk, cache_dir, "prices")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call trust_yield.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk.replace("^", "")] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def clean_quotes(closes: pd.DataFrame, spacs=SPACS, buffer_days: int = BUFFER_DAYS,
                 floor_frac: float = PRINT_FLOOR_FRAC,
                 trust0: float = TRUST0) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace bad **pre-deal** prints with the previous good quote; report the count.

    A pre-deal SPAC share is redeemable for ~$10 of Treasuries, so a quote at a fraction
    of that is a vendor artefact (a cancelled print, a one-lot cross, a corporate-action
    mis-stamp), not a market anyone could have hit. The rule is applied **only inside each
    SPAC's pre-deal window** — after the de-SPAC a collapse to pennies is perfectly real
    and must not be touched. Exactly one name in this list is affected; the per-name count
    is printed in ``docs/results.md`` so the cleaning is auditable, and the headline is
    re-run with that name dropped entirely as a sensitivity.
    """
    floor = floor_frac * trust0
    out = closes.copy()
    counts: dict[str, int] = {}
    for tk, _sym, close in spacs:
        if tk not in out.columns:
            continue
        deadline = pd.Timestamp(close) - pd.Timedelta(days=buffer_days)
        col = out[tk]
        win = (col.index <= deadline) & col.notna()
        bad = win & (col < floor).fillna(False).to_numpy()
        n = int(bad.sum())
        if n:
            counts[tk] = n
            col = col.mask(pd.Series(bad, index=col.index)).ffill()
            out[tk] = col
    return out, counts


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted below-trust discount)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_spacs: int = 24,
    n_days: int = 900,
    signal_strength: float = 1.0,
    discount_mean: float = 0.030,     # planted average below-trust discount at ss=1
    discount_halflife: int = 180,     # days over which the discount pulls to zero
    noise_bps: float = 45.0,          # daily quote noise
    cash_rate_ann: float = 0.035,     # the trust's accrual rate
    start: str = "2021-01-04",
    seed: int = 932,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """A panel of pre-deal SPACs, in a world where the redemption right does or does not bind.

    The ``signal_strength`` knob switches the *mechanism*, not merely its size — which is
    the only null that means anything here. Buying below a line and being paid that line
    is an arithmetic identity; the thing that can fail in the real world is the line.

    - ``signal_strength = 1`` → the trust put binds. The quote is
      ``trust_t * (1 - d_t)`` with ``d_t`` decaying to zero at the deadline, and the
      terminal payoff **is** the accrued trust. Buying below trust must earn a positive
      excess over cash.
    - ``signal_strength = 0`` → the trust put is a fiction. The quote is a driftless
      random walk that happens to start at $10 and has no anchor to the trust line, and
      the terminal payoff is whatever the last quote happens to be. "Below trust" then
      carries no information about the terminal value, so the rule must earn nothing in
      excess of cash (a hair *below* cash once the entry cost is charged).

    Returns ``(prices, cash, meta, truth)``: a wide close frame, a cash accrual index, a
    per-SPAC meta frame (``start``, ``deadline``, ``payoff``) and the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # OOB-safe: a few thousand business days stays far inside pandas' ns horizon.
    dates = pd.bdate_range(start=start, periods=n_days)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(np.cumprod(np.full(n_days, cash_daily)),
                     index=pd.DatetimeIndex(dates, name="date"), name="cash")
    cash_arr = cash.to_numpy()

    prices = pd.DataFrame(index=cash.index, dtype=float)
    rows = []
    for i in range(n_spacs):
        tk = f"SPAC{i:02d}"
        i0 = int(rng.integers(0, max(1, n_days // 3)))
        life = int(rng.integers(300, 620))
        i1 = min(n_days - 1, i0 + life)
        if i1 - i0 < 200:
            i1 = min(n_days - 1, i0 + 260)
        idx = np.arange(i0, i1 + 1)
        trust = TRUST0 * (cash_arr[idx] / cash_arr[i0])

        # --- the "put binds" world: a discount that decays to zero at the deadline ---
        d0 = float(rng.normal(discount_mean, discount_mean * 0.35))
        ttm = (idx[-1] - idx).astype(float)
        planted = d0 * (1.0 - np.exp(-ttm / float(discount_halflife)))
        anchored = trust * (1.0 - planted)

        # --- the "put is a fiction" world: a martingale walk with no anchor to trust.
        # It accrues at the cash rate in expectation (a zero-alpha asset), so a quiet
        # null is a mean excess of ~zero, not a mechanical loss to the cash leg.
        step = rng.normal(0.0, noise_bps * 1e-4, size=idx.size)
        drift = np.cumsum(step) - 0.5 * (noise_bps * 1e-4) ** 2 * np.arange(idx.size)
        walk = trust * np.exp(drift)

        wobble = rng.normal(0.0, noise_bps * 1e-4 * 0.5, size=idx.size) * TRUST0
        px = signal_strength * anchored + (1.0 - signal_strength) * walk + wobble
        payoff = signal_strength * float(trust[-1]) + (1.0 - signal_strength) * float(px[-1])

        col = pd.Series(np.nan, index=cash.index, name=tk, dtype=float)
        col.iloc[idx] = px
        prices[tk] = col
        rows.append({"ticker": tk, "start": cash.index[i0], "deadline": cash.index[i1],
                     "payoff": payoff})

    meta = pd.DataFrame(rows).set_index("ticker")
    truth = {
        "signal_strength": signal_strength,
        "discount_mean": discount_mean,
        "discount_halflife": discount_halflife,
        "noise_bps": noise_bps,
        "cash_rate_ann": cash_rate_ann,
        "n_spacs": n_spacs,
        "n_days": n_days,
        "seed": seed,
        "trust0": TRUST0,
    }
    return prices, cash, meta, truth


def synthetic_daily(signal_strength: float = 1.0, seed: int = 932, **kw):
    """Single-SPAC convenience wrapper over :func:`synthetic_panel` (n_spacs=1)."""
    kw.setdefault("n_spacs", 1)
    return synthetic_panel(signal_strength=signal_strength, seed=seed, **kw)
