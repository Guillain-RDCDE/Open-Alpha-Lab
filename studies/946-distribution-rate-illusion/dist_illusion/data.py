"""Data layer for Study 946 — Distribution is not Return.

The claim under test: an income ETF's **advertised distribution rate** tells you something
about what you will *earn*. The sceptic's counter: a distribution is a transfer, not a
return — the cash leaves the NAV on the ex-date, so a higher payout should predict a lower
*price* path and say nothing at all about *total* return.

Two tapes, one shape (a date-indexed daily close frame, one column per ticker):

- ``fetch`` / ``load_prices`` — daily closes from Yahoo! Finance (``yfinance``) in **two
  flavours**, both cached as parquet under the shared ``studies/_cache``:

    * ``kind="tr"`` → ``prices_<TICKER>_1d.parquet``, ``auto_adjust=True``: the
      **total-return** close (every distribution reinvested).
    * ``kind="pr"`` → ``rawclose_<TICKER>_1d.parquet``, ``auto_adjust=False``: the
      **price-only** close (split-adjusted, *not* distribution-adjusted) — what the
      investor sees quoted, and what erodes on the ex-date.

  ``fetch`` touches the network and retries up to 4×; ``load_prices`` reads the cache
  **offline** and never imports yfinance. The test-suite runs with NO cache present
  (synthetic-only), so CI is green on a fresh checkout.

- ``synthetic_panel`` — a *deterministic, offline* generator of a fund panel with a
  **planted** relationship between payout and total return, controlled by
  ``signal_strength``. At ``signal_strength=0`` the payout is pure return-of-capital: it
  must show up one-for-one in the price leg and **not at all** in total return (the null).
  At ``signal_strength=1`` a genuine yield-to-return link is planted and the estimator has
  to find it (the positive control). Seed is fixed → tests are deterministic.

**The distribution rate is a PROXY, and it is labelled one everywhere.** yfinance does not
publish a fund's marketed "distribution rate". We *reconstruct* the realised payout from the
gap between the two tapes: for month ``t``,

    d_t = (1 + r_total_t) / (1 + r_price_t) − 1

and the trailing-12-month rate is the compounded product of the last twelve ``d_t``. This
differs from a fund's marketing sticker in three named ways: (a) it is *realised trailing*,
not *last-payment-annualised*; (b) it lumps **capital-gains distributions** in with income;
(c) it inherits whatever adjustment convention Yahoo! applied. Every one of those pushes in
the same direction — our measure is the honest cash-out-the-door rate, which is exactly the
quantity the "distribution is not return" claim is about.

**Corporate-action guard (an ASSUMPTION *applied with hindsight*, swept in the results).**
A reverse split that Yahoo! failed to adjust shows up as a huge same-size jump in *both*
tapes (it cancels out of ``d_t`` but wrecks the return). NUSI has exactly one — 2025-02-18,
a 1-for-2 the feed never applied. Rather than hardcode a fund-specific patch we drop any
fund-month whose total return exceeds ``guard`` in absolute value (default 0.50). Be
explicit about what that is: the filter reads the return of the month being *predicted*, so
a fund is dropped from the sort formed at ``t`` because of what its ``t+1`` print turns out
to be — **a hindsight filter, not a rule a live trader could run**. It fires exactly once in
the whole panel (NUSI, 2025-02), it is swept at 0.40/0.50/off, and the panel is also
re-run with NUSI deleted outright; the no-guard column is the honest live-tradable read and
it moves nothing that matters (price leg −57.0 bps, *t* = −2.79; total leg still a null).

**Survivorship, and one more forward-looking filter.** These are the high-payout ETFs that
gathered assets and are still quoted; the income products that closed (and the pre-2013
covered-call wrappers that never scaled) are absent. NUSI itself stopped trading in July
2026, just after the as-of. On top of that, :func:`strategy.sorted_legs` only ranks funds
whose *next*-month return exists, so a fund that stops printing is quietly dropped instead
of realised. Both push the same way: any read on the high-payout leg is an upper bound.

**The price leg is an identity, not a second experiment.** Because the payout is *defined*
as the total/price gap, the tercile spreads satisfy ``hml_price ≡ hml_total − hml_payout``
to machine precision (the realised correlation between the two sides is 0.99995). So the
"NAV erosion" *t* is not independent evidence: it is the payout-persistence *t* carried over
once the total-return spread contributes nothing. What the tape actually supplies is (a) a
strongly forecastable payout and (b) a total-return null — the erosion follows by
arithmetic. It is stated that way everywhere in this study.
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

# The cross-section: fifteen listed income funds spanning ~2 % to ~13 % trailing payout.
# Nine option-income / high-payout wrappers (the "yield sticker" cohort) ...
CORE_FUNDS = ("QYLD", "XYLD", "RYLD", "JEPI", "JEPQ", "SPYI", "DIVO", "NUSI", "PBP")
# ... plus a preferred-share fund and five dividend-equity funds, which supply the low-payout
# end of the sort. Without them there is no cross-section to rank.
WIDE_FUNDS = CORE_FUNDS + ("PFF", "SPHD", "SCHD", "VYM", "DVY", "NOBL")

FUNDS = WIDE_FUNDS
BENCH = "SPY"      # the risk benchmark for the CAPM control
CASH = "BIL"       # 1-3 month T-bill ETF — the cash leg of every excess-of-cash number
TICKERS = FUNDS + (BENCH, CASH)

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# ASSUMPTION: any |monthly total return| above this is an unadjusted corporate action,
# not a market move. Swept in docs/results.md.
GUARD = 0.50

__all__ = [
    "AS_OF", "BENCH", "CASH", "CORE_FUNDS", "DEFAULT_CACHE", "FUNDS", "GUARD",
    "MONTHS_PER_YEAR", "TICKERS", "WIDE_FUNDS",
    "fetch", "fingerprint", "have_real", "load_prices", "monthly_panel",
    "returns_fingerprint", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance, two flavours, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, kind: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    prefix = "prices" if kind == "tr" else "rawclose"
    return os.path.join(cache_dir, f"{prefix}_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2003-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download both the total-return and the price-only daily close for ``tickers``.

    Network-only; run once to populate the shared cache. Each ticker yields two parquet
    files with a single ``close`` column — ``prices_<T>_1d.parquet`` (``auto_adjust=True``,
    distributions reinvested) and ``rawclose_<T>_1d.parquet`` (``auto_adjust=False``, the
    quoted price). The *gap* between the two is the whole study.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        for kind, adjust in (("tr", True), ("pr", False)):
            raw = None
            for _ in range(retries):
                try:
                    raw = yf.download(
                        tk, start=start, end=end, interval="1d",
                        auto_adjust=adjust, progress=False, actions=False,
                    )
                    if raw is not None and len(raw) > 0:
                        break
                except Exception:
                    time.sleep(2.0)
            if raw is None or len(raw) == 0:
                raise RuntimeError(f"yfinance returned no data for {tk} ({kind})")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.lower)
            df = raw[["close"]].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.dropna(subset=["close"])
            df.to_parquet(_cache_path(tk, kind, cache_dir))
            out[f"{tk}:{kind}"] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff both tapes are cached for every ticker (offline-testable)."""
    return all(
        os.path.exists(_cache_path(tk, kind, cache_dir))
        for tk in tickers
        for kind in ("tr", "pr")
    )


def load_prices(
    tickers=TICKERS,
    kind: str = "tr",
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read one cached tape OFFLINE into an aligned close frame (``kind`` = ``tr``/``pr``).

    Sliced to ``asof`` so the sample never creeps between reruns. Raises
    ``FileNotFoundError`` if any ticker is missing — the offline core and the whole
    test-suite never touch the network.
    """
    if kind not in ("tr", "pr"):
        raise ValueError("kind must be 'tr' (total return) or 'pr' (price only)")
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, kind, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached {kind} prices for {tk} at {path}. "
                f"Call dist_illusion.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def returns_fingerprint(prices: pd.DataFrame) -> str:
    """The reproducible data stamp: a fingerprint of the tape's **returns**.

    A *level* fingerprint is not reproducible for the total-return tape, and that is a
    property of the feed, not of this study: ``auto_adjust=True`` back-adjusts the entire
    history every time a new distribution lands, so every past close is rescaled on each
    re-fetch (and ``studies/_cache`` is refreshed by whichever study fetches last). Returns
    are invariant to that rescaling, so this stamp reproduces across re-fetches while still
    moving the instant a single observation actually changes.
    """
    r = prices.sort_index().pct_change(fill_method=None).round(10)
    return fingerprint(r)


# --------------------------------------------------------------------------- #
# The monthly panel — where the distribution rate is reconstructed
# --------------------------------------------------------------------------- #
def monthly_panel(
    tr: pd.DataFrame,
    pr: pd.DataFrame,
    funds=FUNDS,
    guard: float | None = GUARD,
    lookback: int = 12,
) -> dict:
    """Build the monthly fund panel from the two daily tapes.

    Returns a dict of aligned month-end frames:

    ``total``      simple monthly **total** return per fund;
    ``price``      simple monthly **price-only** return per fund;
    ``dist``       the implied monthly distribution ``(1+total)/(1+price) − 1`` (a PROXY);
    ``dist_rate``  the trailing ``lookback``-month compounded distribution rate — the
                   ranking variable, known at the close of the month it is stamped on;
    ``cash``       the cash leg (BIL total return);
    ``bench``      the benchmark leg (SPY total return).

    The two tapes are intersected on their common daily index *before* resampling so the
    two flavours cannot drift apart by a session. Fund-months whose absolute total return
    exceeds ``guard`` are masked out of **both** legs (the unadjusted-corporate-action
    guard; pass ``guard=None`` to disable).
    """
    common = tr.index.intersection(pr.index)
    tr = tr.loc[common].sort_index()
    pr = pr.loc[common].sort_index()

    m_tr = tr.resample("ME").last().pct_change(fill_method=None)
    m_pr = pr.resample("ME").last().pct_change(fill_method=None)

    funds = [f for f in funds if f in m_tr.columns and f in m_pr.columns]
    total = m_tr[funds].copy()
    price = m_pr[funds].copy()
    if guard is not None:
        bad = total.abs() > guard
        total = total.mask(bad)
        price = price.mask(bad)

    dist = (1.0 + total) / (1.0 + price) - 1.0
    # Compounded trailing payout, computed in logs so the rolling window is a plain sum.
    dist_rate = np.exp(np.log1p(dist).rolling(lookback).sum()) - 1.0

    return {
        "total": total,
        "price": price,
        "dist": dist,
        "dist_rate": dist_rate,
        "cash": m_tr[CASH] if CASH in m_tr.columns else None,
        "bench": m_tr[BENCH] if BENCH in m_tr.columns else None,
        "funds": funds,
        "lookback": lookback,
        "guard": guard,
    }


# --------------------------------------------------------------------------- #
# Synthetic panel — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_funds: int = 12,
    n_months: int = 180,
    signal_strength: float = 0.0,
    yield_lo: float = 0.02,          # lowest planted annual distribution rate
    yield_hi: float = 0.13,          # highest planted annual distribution rate
    alpha_scale: float = 1.0,        # at signal_strength=1: extra annual total return per
                                     # 1.0 of annual yield (1.0 = a full one-for-one bonus)
    beta_slope: float = 0.0,         # market beta tilt across the yield sort (a confound knob)
    mkt_mean: float = 0.008,         # monthly market mean
    mkt_vol: float = 0.042,          # monthly market vol
    idio_vol: float = 0.015,         # monthly idiosyncratic vol
    payout_noise: float = 0.15,      # relative wobble on the monthly payout
    cash_annual: float = 0.02,
    start: str = "2008-01",
    seed: int = 946,
) -> tuple[dict, dict]:
    """A deterministic monthly fund panel with a PLANTED payout-to-return relationship.

    Each of ``n_funds`` funds is given a fixed annual distribution rate ``y_i`` spread evenly
    across ``[yield_lo, yield_hi]``. Its monthly **total** return is

        r_it = beta_i · mkt_t + eps_it + signal_strength · alpha_scale · (y_i − ȳ) / 12

    its monthly **payout** is ``y_i/12`` times a positive wobble, and its **price** return is
    whatever is left once the payout has been handed out:

        p_it = (1 + r_it) / (1 + d_it) − 1

    So the two knobs do different jobs:

    - ``signal_strength = 0`` — the **null**. The payout is pure return-of-capital: it is
      perfectly predictable, it erodes the price leg one-for-one, and it carries **no**
      information about total return. A sound estimator must find a big negative price
      slope, a big positive payout slope, and a *flat* total-return slope.
    - ``signal_strength = 1`` — the **positive control**. A genuine yield-to-return link of
      ``alpha_scale`` per unit of yield is planted; the high-minus-low total return must fire.
    - ``beta_slope > 0`` — a **confound** control: market beta declines across the yield sort,
      so the raw high-minus-low total return goes negative in an up-market even though no
      alpha was planted. The CAPM leg of ``strategy`` must absorb it.

    Returns ``(panel, truth)`` with ``panel`` shaped exactly like :func:`monthly_panel`'s
    output. The index is built with ``period_range`` (well inside pandas' ns horizon).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start=start, periods=n_months, freq="M").to_timestamp(how="end")
    idx = pd.DatetimeIndex(idx.normalize(), name="date")

    names = [f"F{i:02d}" for i in range(n_funds)]
    y = np.linspace(yield_lo, yield_hi, n_funds)          # annual planted payout
    y_bar = float(y.mean())
    beta = 1.0 - beta_slope * (y - y_bar) / max(y.std(ddof=0), 1e-12)

    mkt = rng.normal(mkt_mean, mkt_vol, n_months)
    eps = rng.normal(0.0, idio_vol, (n_months, n_funds))
    alpha_m = signal_strength * alpha_scale * (y - y_bar) / MONTHS_PER_YEAR

    total = mkt[:, None] * beta[None, :] + eps + alpha_m[None, :]
    wobble = np.abs(1.0 + rng.normal(0.0, payout_noise, (n_months, n_funds)))
    dist = (y[None, :] / MONTHS_PER_YEAR) * wobble
    price = (1.0 + total) / (1.0 + dist) - 1.0

    total_df = pd.DataFrame(total, index=idx, columns=names)
    price_df = pd.DataFrame(price, index=idx, columns=names)
    dist_df = pd.DataFrame(dist, index=idx, columns=names)
    dist_rate = np.exp(np.log1p(dist_df).rolling(MONTHS_PER_YEAR).sum()) - 1.0

    cash = pd.Series(
        np.full(n_months, (1.0 + cash_annual) ** (1.0 / MONTHS_PER_YEAR) - 1.0),
        index=idx, name=CASH,
    )
    bench = pd.Series(mkt, index=idx, name=BENCH) + cash

    panel = {
        "total": total_df, "price": price_df, "dist": dist_df, "dist_rate": dist_rate,
        "cash": cash, "bench": bench, "funds": names,
        "lookback": MONTHS_PER_YEAR, "guard": None,
    }
    truth = {
        "signal_strength": signal_strength, "alpha_scale": alpha_scale,
        "beta_slope": beta_slope, "n_funds": n_funds, "n_months": n_months,
        "yield_lo": yield_lo, "yield_hi": yield_hi, "yields": y, "betas": beta,
        "yield_mean": y_bar, "yield_sd": float(y.std(ddof=1)), "seed": seed,
        # The planted monthly total-return slope per 1 sd of annual yield.
        "planted_slope_per_sd": float(
            signal_strength * alpha_scale * y.std(ddof=1) / MONTHS_PER_YEAR
        ),
    }
    return panel, truth
