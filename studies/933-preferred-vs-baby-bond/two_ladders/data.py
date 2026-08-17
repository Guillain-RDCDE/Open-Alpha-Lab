"""Data layer for Study 933 — Same Issuer, Two Ladders.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for eight issuers that list **both** a
  **preferred** and an exchange-traded **baby bond** (retail-denominated $25-par
  notes, NYSE/Nasdaq listed), plus **PFF** (the preferred-sector ETF) and **BIL**
  (the cash leg). ``fetch`` touches the network and caches parquet in the **shared**
  ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache **offline**
  and never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic-only).

  **The rungs are not uniform** — see ``RUNG_STRUCTURE``. Only two of the eight pairs
  (B. Riley, Babcock & Wilcox) are the textbook "perpetual preferred vs *senior* note".
  In four the note is itself **junior subordinated** or a **perpetual subordinated**
  note, and in two the "preferred" is a **dated term preferred** with a mandatory
  redemption date — so the seniority step is only one rung, sometimes less, and it is
  not the same step in every pair. That heterogeneity is part of the answer, not a
  footnote to it.

- ``synthetic_panel`` — a *deterministic, offline* generator. A panel of issuers,
  each with a common credit factor driving two legs of one balance sheet: a senior
  (baby-bond) leg and a junior (preferred) leg with a larger factor loading. The
  ``signal_strength`` knob plants a **junior carry premium**: at ``signal_strength=0``
  the pairwise (preferred − bond) return difference has mean exactly zero (the null);
  at ``signal_strength=1`` the preferred leg earns the full planted premium.

The claim under test: for one and the same balance sheet, the preferred sits *below*
the baby bond in the capital stack (it is equity for accounting and regulatory purposes,
its coupon is discretionary for many issuers, and it is usually perpetual). If the market prices
seniority correctly, the preferred should pay a **higher return** and carry **more risk**,
leaving the risk-adjusted race roughly a wash. If it pays *more* return **without** more
risk-adjusted cost — or the reverse — one rung of the ladder is mis-priced relative to the
other, and the pair is a clean, issuer-neutral test because the credit is identical.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (weights formed
on data through day ``t`` are acted on at day ``t+1``).
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

# --------------------------------------------------------------------------- #
# The eight issuer pairs. Each entry: (issuer label, preferred ticker, baby-bond
# ticker, sector). Every pair is ONE balance sheet: same obligor, two listed rungs.
# Yahoo's hyphen convention for preferreds: "CMS-PB" = CMS Energy Series B preferred.
# --------------------------------------------------------------------------- #
PAIRS = (
    ("CMS Energy",             "CMS-PB",  "CMSC",   "utility"),
    ("Duke Energy",            "DUK-PA",  "DUKB",   "utility"),
    ("Brookfield Renewable",   "BEP-PA",  "BEPH",   "infrastructure"),
    ("Brookfield Infra.",      "BIP-PA",  "BIPH",   "infrastructure"),
    ("B. Riley Financial",     "RILYP",   "RILYZ",  "financial"),
    ("Oxford Lane Capital",    "OXLCP",   "OXLCZ",  "closed-end credit"),
    ("Eagle Point Credit",     "ECC-PD",  "ECCC",   "closed-end credit"),
    ("Babcock & Wilcox",       "BW-PA",   "BWNB",   "industrial"),
)

PREF_TICKERS = tuple(p for _, p, _, _ in PAIRS)
BOND_TICKERS = tuple(b for _, _, b, _ in PAIRS)

# --------------------------------------------------------------------------- #
# What each rung ACTUALLY is. The headline framing ("junior perpetual preferred vs
# senior dated note") is only exact for two of the eight pairs; naming the rest is
# part of the study's answer, because a small seniority step cannot pay a large
# seniority premium. ``step`` is the qualitative size of the subordination gap.
# --------------------------------------------------------------------------- #
RUNG_STRUCTURE = {
    "CMS Energy":           ("cumulative convertible preferred", "junior subordinated notes 2078", "narrow"),
    "Duke Energy":          ("perpetual preferred", "junior subordinated debentures 2078", "narrow"),
    "Brookfield Renewable": ("perpetual preferred units", "PERPETUAL subordinated notes", "narrow"),
    "Brookfield Infra.":    ("perpetual preferred units", "PERPETUAL subordinated notes", "narrow"),
    "B. Riley Financial":   ("perpetual preferred", "senior notes 2028", "wide"),
    "Oxford Lane Capital":  ("DATED term preferred 2027", "senior notes 2031", "narrow"),
    "Eagle Point Credit":   ("DATED term preferred 2031", "senior notes 2031", "narrow"),
    "Babcock & Wilcox":     ("perpetual preferred", "senior notes 2026", "wide"),
}

# The two pairs that carry both the return spread and the risk step — and the only two
# where the seniority step is the textbook one. Used by the concentration probe.
DISTRESSED = ("B. Riley Financial", "Babcock & Wilcox")

# Sector benchmark (the fallback comparison) + the cash leg.
BENCH = ("PFF",)      # iShares Preferred & Income Securities ETF
CASH = "BIL"          # 1-3 month T-bill ETF — the cash leg of every excess race

# Second-instrument cross-checks: an alternative rung for four of the issuers, used to
# show the result is not an artefact of which specific CUSIP we happened to pick.
ALT_TICKERS = ("CMSA", "CMSD", "BEPI", "BIPI")

# Two REDEEMED Sachem Capital baby bonds (SACC matured 2024-12, SCCB called 2024-06) kept
# only to *measure* the survivorship hole the live panel cannot see. Not in the headline.
DEAD_TICKERS = ("SACH-PA", "SACC", "SCCB")

TICKERS = PREF_TICKERS + BOND_TICKERS + BENCH + (CASH,) + ALT_TICKERS + DEAD_TICKERS

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# The common window: the last pair leg to list (BIPI/OXLCZ, January 2022) gates the panel.
PANEL_START = "2022-02-01"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    ``close`` column dividend-adjusted total return — indispensable here, because both
    rungs of the ladder are *coupon instruments*: a price-only race between a 5.9% and a
    7.4% coupon is meaningless. Total return is what we compare, everywhere.

    Yahoo's adjustment for thin $25-par tickers is itself a **PROXY** for a clean
    total-return series: the closes are often stale prints (see
    ``strategy.staleness_table``), which is why every *risk* claim in the study is
    re-checked at weekly and monthly frequency before it is believed.
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
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

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
                f"Call two_ladders.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def panel_returns(
    prices: pd.DataFrame,
    pairs=PAIRS,
    start: str = PANEL_START,
    asof: str = AS_OF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cached close frame into aligned (preferred, baby-bond) daily return panels.

    Both frames share one index and one column order (the issuer labels), restricted to the
    common window in which **every** leg of **every** pair trades. Simple (arithmetic)
    daily total returns.
    """
    labels = [lbl for lbl, _, _, _ in pairs]
    pref = prices[[p for _, p, _, _ in pairs]].copy()
    bond = prices[[b for _, _, b, _ in pairs]].copy()
    pref.columns = labels
    bond.columns = labels
    win = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(asof))
    pref, bond = pref[win], bond[win]
    keep = pref.notna().all(axis=1) & bond.notna().all(axis=1)
    pref, bond = pref[keep], bond[keep]
    return pref.pct_change().dropna(), bond.pct_change().dropna()


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted junior carry premium)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_issuers: int = 8,
    n_years: int = 10,
    junior_premium_ann: float = 0.06,    # planted preferred-minus-bond carry, annualised
    factor_vol_ann: float = 0.08,        # common credit-factor vol
    bond_beta: float = 0.80,             # senior leg's loading on the issuer credit factor
    pref_beta: float = 1.15,             # junior leg's loading (further down the stack)
    idio_vol_ann: float = 0.035,         # per-leg idiosyncratic vol
    signal_strength: float = 1.0,        # 0 = exact null, 1 = full planted premium
    start: str = "2019-01-02",
    seed: int = 933,
    cash_rate_ann: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """A synthetic panel of issuers, each with a senior and a junior listed rung.

    Every issuer draws a common market credit factor plus its own issuer factor; the two
    legs of the same balance sheet load on the *same* issuer factor with different betas
    (``pref_beta`` > ``bond_beta``, because the preferred sits lower in the stack), and
    each leg adds its own idiosyncratic noise.

    The ``signal_strength`` knob scales the **planted junior carry premium**:

    - ``signal_strength = 1`` → the preferred leg earns ``junior_premium_ann`` more per
      year than the bond leg, on top of its higher beta; the pairwise estimator must
      recover it.
    - ``signal_strength = 0`` → the premium is exactly zero; the pairwise difference has
      mean zero by construction and the estimator must stay quiet (the null).

    Returns ``(pref_returns, bond_returns, cash_returns, truth)``. Deterministic given
    ``seed``. Daily simple returns, one column per issuer.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a modest bdate_range (<= a few thousand bars) stays inside pandas' ns range.
    dates = pd.DatetimeIndex(pd.bdate_range(start=start, periods=n_days), name="date")

    f_market = rng.normal(0.0, factor_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    prem_d = signal_strength * junior_premium_ann / TRADING_DAYS_PER_YEAR
    s_idio = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_issuer = 0.5 * factor_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    labels = [f"issuer_{i:02d}" for i in range(n_issuers)]
    pref_cols, bond_cols = {}, {}
    for i, lbl in enumerate(labels):
        f_issuer = rng.normal(0.0, s_issuer, n_days)
        f = f_market + f_issuer
        base = 0.045 / TRADING_DAYS_PER_YEAR      # a shared coupon-like drift
        bond_cols[lbl] = base + bond_beta * f + rng.normal(0.0, s_idio, n_days)
        pref_cols[lbl] = base + prem_d + pref_beta * f + rng.normal(0.0, s_idio, n_days)

    pref = pd.DataFrame(pref_cols, index=dates)[labels]
    bond = pd.DataFrame(bond_cols, index=dates)[labels]
    cash = pd.Series(
        np.full(n_days, cash_rate_ann / TRADING_DAYS_PER_YEAR), index=dates, name="cash"
    )
    truth = {
        "signal_strength": signal_strength,
        "junior_premium_ann": junior_premium_ann,
        "planted_premium_ann": signal_strength * junior_premium_ann,
        "bond_beta": bond_beta,
        "pref_beta": pref_beta,
        "factor_vol_ann": factor_vol_ann,
        "idio_vol_ann": idio_vol_ann,
        "n_issuers": n_issuers,
        "n_days": n_days,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
    }
    return pref, bond, cash, truth


def synthetic_daily(**kwargs):
    """Single-pair convenience wrapper around :func:`synthetic_panel` (one issuer)."""
    kwargs.setdefault("n_issuers", 1)
    return synthetic_panel(**kwargs)
