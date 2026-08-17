"""Data layer for Study 955 — ADR Catch-Up.

Two tapes, one shape (a long panel of daily returns, one row per ADR per date):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for eight large cross-listed ADRs, the three
  home indices they belong to (``^N225``, ``^GDAXI``, ``^FTSE``), the three currency
  pairs that translate a home move into dollars (``JPY=X``, ``EURUSD=X``, ``GBPUSD=X``)
  and a cash leg (``^IRX``, the 13-week T-bill discount yield, with ``BIL`` as a
  cross-check). ``fetch`` touches the network and writes parquet into the **shared**
  cache at ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache
  **offline** and never imports yfinance.

- ``synthetic_panel`` — a *deterministic, offline* generator. Home index moves, an FX
  move and an ADR that loads on the home move — except that a fraction of the loading
  arrives **one day late**. The ``signal_strength`` knob is exactly that fraction: at
  ``signal_strength=1`` there is a genuine, detectable catch-up lag; at
  ``signal_strength=0`` the ADR prices the home move in full the same day (the null),
  and no catch-up rule may show an edge. Seed is fixed → tests are deterministic.

The whole test-suite runs with NO cache present (synthetic-only), so CI is green on a
fresh checkout where ``studies/_cache/`` is git-ignored and absent.

**The coarse-proxy warning, up front.** Genuine ADR catch-up is an *intraday* effect:
Tokyo closes at 02:00 New York time, London and Frankfurt at 11:30, and the question is
whether the ADR's *US session* absorbs that information. A daily-close tape cannot see a
session; it sees only close-to-close. What this study can honestly test is the residue
that survives to the ADR's own close — whether yesterday's home move still predicts
today's ADR return. That is a weaker, but entirely well-posed, question, and it is the
one we answer. This limitation is named on the Signal axis.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal
formed from closes through day ``t`` is acted on at day ``t+1``).
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

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# The sample opens where the FX tapes do (EURUSD=X / GBPUSD=X start 2003-12).
SAMPLE_START = "2004-01-02"

# --------------------------------------------------------------------------- #
# The ADR universe: (home index, FX pair, invert_fx, region)
#
# ``invert_fx`` is True when the Yahoo pair is quoted *local per USD* (JPY=X is
# USD/JPY, ~159 yen per dollar) rather than *USD per local* (EURUSD=X, GBPUSD=X).
# Inverting turns it into the dollar value of one unit of the home currency, which is
# what a US holder of the ADR actually earns on the currency leg.
# --------------------------------------------------------------------------- #
ADR_PAIRS: dict[str, tuple[str, str, bool, str]] = {
    "TM":   ("^N225",  "JPY=X",    True,  "Japan"),
    "SONY": ("^N225",  "JPY=X",    True,  "Japan"),
    "SAP":  ("^GDAXI", "EURUSD=X", False, "Europe"),
    "NVO":  ("^GDAXI", "EURUSD=X", False, "Europe"),
    "SHEL": ("^FTSE",  "GBPUSD=X", False, "UK"),
    "BP":   ("^FTSE",  "GBPUSD=X", False, "UK"),
    "HSBC": ("^FTSE",  "GBPUSD=X", False, "UK"),
    "RIO":  ("^FTSE",  "GBPUSD=X", False, "UK"),
}

ADRS = tuple(ADR_PAIRS)
HOME_INDICES = ("^N225", "^GDAXI", "^FTSE")
FX = ("JPY=X", "EURUSD=X", "GBPUSD=X")
CASH = ("^IRX", "BIL")

TICKERS = ADRS + HOME_INDICES + FX + CASH

# Home-market close times in New York hours, for the honest overlap accounting. Tokyo
# closes thirteen hours before the ADR does; London and Frankfurt close *inside* the US
# session, so most of their move is already in the ADR's own close-to-close window.
HOME_CLOSE_ET = {"^N225": "02:00", "^GDAXI": "11:30", "^FTSE": "11:30"}

# NVO is Danish, and no Copenhagen index sits in the shared cache long enough to run the
# main 2004 sample, so the headline panel pairs it with ^GDAXI/EURUSD=X — a **PROXY** for
# a euro-area home market (the krone is pegged to the euro inside a +/-2.25% band). The
# genuine Danish pair is available from 2016-12 and is run as a cross-check.
NVO_CROSS_CHECK = {"NVO": ("^OMXC25", "DKK=X", True, "Denmark")}
CROSS_CHECK_TICKERS = ("NVO", "^OMXC25", "DKK=X", "^GDAXI", "EURUSD=X")
CROSS_CHECK_START = "2016-12-20"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. Uses ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted **total return** for the ADRs — which
    matters here, because these are high-payout names (BP, HSBC, Shell) whose price-only
    tape would understate their return by several points a year. The home indices
    (``^N225``, ``^GDAXI``, ``^FTSE``) are **price-only** by construction: Yahoo publishes
    the price index, not the total-return index, for all three. That asymmetry is harmless
    for this study — the home index is used only as a same-day *explanatory* move, never
    raced against the ADR — but it is labelled everywhere.
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
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    start: str = SAMPLE_START,
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
                f"Call adr_catchup.data.fetch() once to populate the shared cache."
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


def cash_returns(prices: pd.DataFrame, col: str = "^IRX") -> pd.Series:
    """Daily simple cash return from the 13-week T-bill discount yield (``^IRX``).

    ``^IRX`` quotes an annualised **discount yield in percent**; we convert it to a daily
    simple accrual (``yield/100/252``) held over the following session. This is a *proxy*
    for a tradable cash leg — it ignores the small discount-to-bond-equivalent conversion
    and any bill-fund expense ratio — and is labelled as such. ``BIL`` (a real, costed
    T-bill ETF, but only from 2007) is cached as the cross-check.
    """
    y = prices[col].astype(float) / 100.0 / TRADING_DAYS_PER_YEAR
    return y.shift(1).rename("r_cash")


# --------------------------------------------------------------------------- #
# The panel — one row per ADR per date
# --------------------------------------------------------------------------- #
def build_panel(prices: pd.DataFrame, pairs: dict | None = None) -> pd.DataFrame:
    """Turn the close frame into the long return panel this study regresses on.

    Columns, per ADR ``i`` and date ``t``:

    - ``a``  — the ADR's US total-return close-to-close return, ``t-1 -> t``.
    - ``h``  — the home index's **price-only** close-to-close return in *local* currency.
    - ``f``  — the dollar return of one unit of the home currency (``JPY=X`` inverted).
    - ``x``  — the home move **in dollars**, ``(1+h)(1+f) - 1``: what a US investor would
      have earned on the home line over the same calendar day.

    Dates are the intersection of the ADR, home-index and FX calendars, so a Tokyo or
    London holiday drops the row rather than smearing a two-day home move onto one ADR
    day. Rows are returned sorted by date (then ADR), which is what the pooled HAC and
    the Fama-MacBeth pass both expect.

    Returns are formed by explicit division (``p_t / p_{t-1} - 1``) rather than
    ``pct_change()`` **on purpose**: the close frame is a *union* of three calendars, so it
    carries NaNs wherever one market was shut, and ``pct_change``'s legacy default
    (``fill_method="pad"``, still the default on pandas 2.x) would forward-fill the missing
    close, print a spurious 0% on the holiday and then smear a two-day home move onto the
    next ADR day — the exact contamination this function claims to avoid. Explicit division
    propagates the NaN, so both the holiday row and the row after it drop out, and the
    panel is identical on pandas 2.x and 3.x.
    """
    pairs = pairs or ADR_PAIRS
    frames = []

    def _ret(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return s / s.shift(1) - 1.0

    for adr, (home, fx, invert, region) in pairs.items():
        if adr not in prices.columns or home not in prices.columns or fx not in prices.columns:
            continue
        a = _ret(prices[adr])
        h = _ret(prices[home])
        f = _ret(prices[fx])
        if invert:
            # Quoted local-per-USD: the dollar value of the local unit moves the other way.
            f = 1.0 / (1.0 + f) - 1.0
        df = pd.DataFrame({"a": a, "h": h, "f": f}).dropna()
        df["x"] = (1.0 + df["h"]) * (1.0 + df["f"]) - 1.0
        df["adr"] = adr
        df["region"] = region
        frames.append(df)
    panel = pd.concat(frames)
    panel.index.name = "date"
    return panel.sort_values(["date", "adr"])


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted catch-up lag)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_names: int = 8,
    n_years: int = 20,
    signal_strength: float = 1.0,   # = the fraction of the home move that arrives LATE
    max_lag_share: float = 0.40,    # the planted late share at signal_strength = 1
    bounce_vol: float = 0.0,        # plain single-stock pricing noise (the CONFOUND)
    home_vol: float = 0.011,        # daily home-index vol
    fx_vol: float = 0.006,          # daily FX vol
    idio_vol: float = 0.013,        # daily ADR-specific vol
    drift_ann: float = 0.06,        # ADR annual drift
    beta_low: float = 0.40,
    beta_high: float = 0.90,
    start: str = "2006-01-02",
    seed: int = 955,
) -> tuple[pd.DataFrame, dict]:
    """A synthetic ADR panel with two *independently* plantable effects.

    **1. The catch-up lag** (``signal_strength``). Each name loads with coefficient
    ``b_i`` on its home market's dollar move ``x_t``, but only ``(1 - lag_share)`` of that
    loading is priced the same day; the remaining ``lag_share`` arrives on day ``t+1``:

        ``a_t = b_i * [ (1 - lag_share) x_t + lag_share x_{t-1} ] + u_t + (n_t - n_{t-1})``

    - ``signal_strength = 1`` → ``lag_share = max_lag_share``: a genuine stale-price
      effect. The stale-catchup regression must find a positive loading on ``x_{t-1}``,
      the home move must still predict once the ADR's own move is controlled for, and the
      costed catch-up book must earn gross.
    - ``signal_strength = 0`` → ``lag_share = 0``: the ADR prices the home move in full
      the same day (the null). Nothing catch-up-shaped may fire.

    **2. Plain pricing noise** (``bounce_vol``, the *confound*). A transient error ``n_t``
    that enters today's return and unwinds tomorrow — the textbook bid-ask bounce. It has
    nothing to do with the home market, but it makes *any* residual mean-revert. Planting
    it with ``signal_strength = 0`` is how the test-suite proves the residual-reversal rule
    is **not specific** to catch-up: it fires on pure bounce with zero stale information.
    That non-specificity is the methodological spine of this study.

    Returns ``(panel, truth)`` where ``panel`` has the same columns as
    :func:`build_panel` (``a``, ``h``, ``f``, ``x``, ``adr``, ``region``) and ``truth``
    records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)
    lag_share = float(np.clip(signal_strength, 0.0, 1.0)) * max_lag_share

    n_homes = max(1, n_names // 2)
    home_rets = rng.normal(0.0002, home_vol, size=(n_days, n_homes))
    fx_rets = rng.normal(0.0, fx_vol, size=(n_days, n_homes))

    mu = drift_ann / TRADING_DAYS_PER_YEAR
    frames = []
    betas = {}
    for i in range(n_names):
        j = i % n_homes
        h = home_rets[:, j]
        f = fx_rets[:, j]
        x = (1.0 + h) * (1.0 + f) - 1.0
        b = float(rng.uniform(beta_low, beta_high))
        betas[f"A{i:02d}"] = b
        x_lag = np.concatenate([[0.0], x[:-1]])
        u = rng.normal(mu, idio_vol, n_days)
        a = b * ((1.0 - lag_share) * x + lag_share * x_lag) + u
        if bounce_vol > 0:
            noise = rng.normal(0.0, bounce_vol, n_days + 1)
            a = a + (noise[1:] - noise[:-1])
        frames.append(pd.DataFrame(
            {"a": a, "h": h, "f": f, "x": x,
             "adr": f"A{i:02d}", "region": f"R{j}"},
            index=pd.DatetimeIndex(dates, name="date"),
        ))
    panel = pd.concat(frames).sort_values(["date", "adr"])
    truth = {
        "signal_strength": float(signal_strength),
        "lag_share": lag_share,
        "max_lag_share": max_lag_share,
        "bounce_vol": bounce_vol,
        "n_names": n_names,
        "n_years": n_years,
        "n_days": n_days,
        "home_vol": home_vol,
        "fx_vol": fx_vol,
        "idio_vol": idio_vol,
        "drift_ann": drift_ann,
        "betas": betas,
        "seed": seed,
    }
    return panel, truth


def synthetic_daily(**kwargs):
    """Alias kept for symmetry with the desk's single-asset studies."""
    return synthetic_panel(**kwargs)
