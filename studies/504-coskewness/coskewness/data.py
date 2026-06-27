"""Data layer for Study 504 — Coskewness (Harvey-Siddique 2000).

Two sources, both offline-friendly once the cache is built:

* **Real tape.** Daily adjusted closes for a fixed, long-listed S&P-100-style basket of US
  large-caps **plus the market proxy SPY** (yfinance, no key), cached under
  ``_cache/basket_prices.csv`` (a wide CSV indexed by date, one column per ticker; SPY is one
  of the columns and is split out as the market). Every month-end we measure each name's
  **coskewness with the market** over a **trailing 12-month window** — Harvey & Siddique's (2000)
  *direct* standardised coskewness,

  .. math:: \\beta_{SKD} = \\frac{E[\\varepsilon_i\\,\\varepsilon_m^2]}{\\sqrt{E[\\varepsilon_i^2]}\\;E[\\varepsilon_m^2]},

  where :math:`\\varepsilon_i` and :math:`\\varepsilon_m` are the demeaned daily stock and market
  returns. Coskewness is the part of a name's tail risk that **co-moves with the market's** own
  tail — a *systematic*, undiversifiable skew. Harvey-Siddique price it: names with **low (more
  negative)** coskewness pay down the market when the market crashes, so investors demand a
  premium; high-coskewness names are insurance and earn less. Sorting the cross-section and going
  **long the low-coskew quintile, short the high-coskew quintile** gives the coskewness factor.
  This large-cap basket is an explicit, transparent **proxy** for the CRSP universe Harvey-Siddique
  used (survivorship-tilted, named on the Signal axis everywhere).

* **Synthetic.** A deterministic, fixed-seed generator that builds a monthly *panel* of returns
  with a controllable **coskewness-premium edge** (``edge``): names that printed low (more
  negative) coskewness last month are made to out-perform next month by a planted amount.
  ``edge=0`` is the null — coskewness carries no forward information, so a long-low / short-high
  book is a coin; a large ``edge`` must light the harness up. This is the positive control and
  the null in one bottle. With ``edge=0`` the harness must NOT manufacture significance.

This is **distinct** from Study 503 (Expected Idiosyncratic Skewness): there the signal is the
skew of the *residual* (the market-orthogonal, *diversifiable* tail); here it is the *systematic*
co-movement of the name's tail with the market's tail — the opposite axis of the same regression.

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "basket_prices.csv")

MARKET = "SPY"  # market proxy column; the conditioning factor for coskewness

# A transparent, fixed S&P-100-style basket of large, long-listed US large-caps used as the
# cross-section for the coskewness sort. Chosen for long price history and sector spread; this is
# a PROXY for the CRSP universe Harvey-Siddique (2000) used. Survivorship is acknowledged on the
# Signal axis: a fixed surviving-large-cap basket is a thin, high-quality slice of the market.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "GE", "IBM",
    "CVX", "PFE", "MRK", "T", "VZ", "INTC", "CSCO", "HD", "MCD", "DIS",
    "BA", "CAT", "MMM", "HON", "UNH", "WFC", "C", "BAC", "ORCL", "PEP",
    "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "USB", "AXP", "DE", "DUK",
    "NKE", "SBUX", "QCOM", "AMD", "MU", "ADBE", "CRM", "NFLX", "AMZN",
    "GOOGL", "TMO", "DHR", "LIN", "ACN", "COP", "SLB", "EOG", "OXY", "MO",
    "PM", "CL", "GIS", "KMB", "SO", "D", "EXC", "AEP", "NEE", "F",
    "GM", "GILD", "BIIB", "REGN", "VRTX", "BK", "STT", "SCHW", "MS", "BLK",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "2005-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the basket + market proxy via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/basket_prices.csv``. Never imported by the offline
    notebook cells. Keeps only names with a long, mostly-complete history. Retries to guard
    yfinance flakiness.
    """
    import time

    import yfinance as yf

    tickers = sorted(set(BASKET) | {MARKET})
    raw = None
    for attempt in range(4):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and raw.shape[1] > 0:
                break
        except Exception as exc:  # noqa: BLE001 - network flakiness
            print(f"  yfinance attempt {attempt + 1} failed: {exc}")
        time.sleep(2.0 * (attempt + 1))
    if raw is None:
        raise RuntimeError("yfinance download failed after retries")
    raw = raw.dropna(how="all")
    # keep names present for >=50% of the window (long-listed survivors); always keep MARKET
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.50 or c == MARKET]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers incl. SPY)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def build_panel(prices: pd.DataFrame, window_months: int = 12, min_obs: int = 120) -> dict:
    """From daily prices, build the monthly coskewness panel.

    For each name and each month-end ``t`` we compute the **direct standardised coskewness** of
    the name's daily returns with the market's daily returns over the trailing ``window_months``
    calendar months (Harvey & Siddique, 2000):

        beta_SKD = E[e_i * e_m^2] / ( sqrt(E[e_i^2]) * E[e_m^2] )

    where ``e_i`` and ``e_m`` are the demeaned daily stock and market returns over the window.
    A **negative** beta_SKD means the name tends to be *down* when the market is *unusually
    volatile / falling hard* — it adds to the market's left tail, undiversifiable, so investors
    demand a premium. That coskewness, observed at the close of month ``t``, is our signal. We
    pair it with the name's return **the next calendar month** — so the panel is already lagged
    by construction (no look-ahead).

    Returns a dict with:
      * ``coskew``  — DataFrame (month-end index x ticker): trailing direct coskewness.
      * ``fwd_ret`` — DataFrame (month-end index x ticker): next-calendar-month simple return.
      * ``mret``    — DataFrame (month-end index x ticker): same-month simple return (for legs).
    """
    prices = prices.sort_index()
    daily = prices.pct_change()
    if MARKET not in daily.columns:
        raise KeyError(f"market proxy {MARKET!r} missing from cached prices")
    mkt = daily[MARKET]
    names = [c for c in daily.columns if c != MARKET]

    # calendar month-ends present in the data (signal stamps)
    me_index = daily.resample("ME").last().index

    coskew = pd.DataFrame(index=me_index, columns=names, dtype=float)
    win = pd.DateOffset(months=window_months)
    for t in me_index:
        lo = t - win
        wmkt = mkt.loc[(mkt.index > lo) & (mkt.index <= t)]
        if len(wmkt) < min_obs:
            continue
        mx = wmkt.to_numpy(dtype=float)
        mfin = np.isfinite(mx)
        emc = mx - np.nanmean(mx[mfin])
        em2 = emc * emc
        denom_m = float(np.nanmean(em2[mfin]))          # E[e_m^2] = market variance
        if denom_m <= 0:
            continue
        for nm in names:
            wr = daily[nm].loc[(daily.index > lo) & (daily.index <= t)]
            wr = wr.reindex(wmkt.index)
            mask = wr.notna().to_numpy() & mfin
            if mask.sum() < min_obs:
                continue
            y = wr.to_numpy(dtype=float)[mask]
            ei = y - y.mean()
            sd_i = ei.std()
            if sd_i <= 0:
                continue
            num = float((ei * em2[mask]).mean())        # E[e_i * e_m^2]
            coskew.loc[t, nm] = num / (sd_i * denom_m)

    # monthly total return = product of (1+daily) - 1, per calendar month
    grp = daily[names].groupby(pd.Grouper(freq="ME"))
    cnt = grp.count()
    mret = grp.apply(lambda d: (1.0 + d).prod() - 1.0)
    mret = mret.where(cnt >= 15)
    mret = mret.reindex(me_index)
    fwd = mret.shift(-1)
    return {"coskew": coskew, "fwd_ret": fwd, "mret": mret}


def load_real(path: str = DEFAULT_CACHE, **kw) -> dict:
    """Convenience: cached prices -> monthly coskewness panel in one call."""
    return build_panel(load_prices(path), **kw)


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of the price frame, for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_months: int = 240, n_names: int = 80, edge: float = 0.0,
                    seed: int = 504, mkt_vol: float = 0.045, idio_vol: float = 0.06,
                    mkt_drift: float = 0.006) -> dict:
    """Deterministic monthly panel with a *planted* coskewness-premium edge.

    Each name loads on a common monthly market factor plus an idiosyncratic shock. Every month
    we compute a stand-in **coskewness** for each name (a fixed per-name coskewness loading
    perturbed by that month's shock) and, if ``edge`` != 0, **add**
    ``edge`` × (cross-sectionally standardised *negative* coskewness) to each name's *next*-month
    return — i.e. low- (negative-) coskewness names are rewarded next month, exactly the risk
    premium the sort should detect (low coskewness → high subsequent return).

    * ``edge = 0``  → coskewness is pure noise w.r.t. forward returns; the long-low / short-high
      book is a coin. The harness must NOT print significance.
    * ``edge > 0``  → low-coskew names out-perform by a known amount; the spread must turn
      positive and (for a large edge) clear t = 2. The positive control.

    A *decorative* month-end index is built with ``pd.period_range`` (never a huge
    ``date_range`` span) to stay far under the ns-Timestamp overflow wall.

    Returns the same ``{"coskew", "fwd_ret", "mret"}`` shape as :func:`build_panel`.
    """
    rng = np.random.default_rng(seed)
    n, k = n_months, n_names
    idx = pd.period_range("2005-01", periods=n, freq="M").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")
    cols = [f"N{i:02d}" for i in range(k)]

    betas = rng.uniform(0.7, 1.3, size=k)
    cosk_load = rng.uniform(-1.0, 1.0, size=k)            # fixed per-name coskewness tendency
    mkt = rng.normal(mkt_drift, mkt_vol, size=n)
    idio = rng.normal(0.0, idio_vol, size=(n, k))

    # coskewness stand-in: the fixed per-name loading + a monthly perturbation
    ck = cosk_load[None, :] + 0.3 * rng.normal(0.0, 1.0, size=(n, k))

    # base monthly returns (market + idio)
    base = mkt[:, None] * betas[None, :] + idio

    # plant the coskewness premium: next month, LOW- (negative-) coskew names out-perform.
    # standardise coskewness cross-sectionally, then reward the low end (-z) next month.
    z = (ck - ck.mean(axis=1, keepdims=True)) / (ck.std(axis=1, keepdims=True) + 1e-9)
    bonus = np.zeros_like(base)
    bonus[1:, :] = edge * (-z[:-1, :])         # this month's LOW coskew rewards next month's return
    ret = base + bonus

    mret = pd.DataFrame(ret, index=idx, columns=cols)
    coskew = pd.DataFrame(ck, index=idx, columns=cols)
    fwd = mret.shift(-1)
    return {"coskew": coskew, "fwd_ret": fwd, "mret": mret}
