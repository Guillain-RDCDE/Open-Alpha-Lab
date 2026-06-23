"""Data layer for Study 395 (Quantum-Computing-Basket).

The "quantum-computing basket" is the archetypal **hype-cycle** trade: hold the handful of
pure-play quantum-computing stocks — IONQ, RGTI, QBTS, QUBT — equal-weight and ride the "next
big thing" instead of the boring index. On the realised tape these names have produced
*spectacular* headline returns (and *spectacular* drawdowns), so the data layer must supply a
**monthly return panel** (the pure-plays + the diversified ``QTUM`` quantum ETF as a sane-basket
control) alongside the market (SPY) and tech (QQQ) benchmarks, so the harness can ask the decisive
question: is this a real, risk-adjusted edge — or a high-volatility lottery whose raw CAGR is a
mirage once you look at Sharpe, drawdown, and the standard error of the spread?

Two tapes, one shape (a return panel + benchmark return series):

- ``synthetic_panel`` — a *deterministic, offline* generator. A single knob, ``edge_sharpe``,
  plants the one thing a real theme would have: a *persistent* risk-adjusted edge (positive mean
  net of the extra volatility) on a fixed, pre-specified basket. At ``edge_sharpe=0`` the basket
  has the *same* expected risk-adjusted return as the market but a much fatter tail (the hype
  signature), so its raw CAGR can look huge while its Sharpe and HAC *t* stay at the null — exactly
  the mirage the study exposes. At ``edge_sharpe>0`` the basket genuinely out-earns on a
  risk-adjusted basis and the HAC *t* lights up. This is the study's null and positive control in
  one.

- ``load_real`` — a real monthly panel of the quantum pure-plays + the QTUM ETF plus the SPY / QQQ
  benchmarks, via yfinance, cache-first (parquet) with a graceful offline contract.
  **Survivorship / look-ahead is named on the Signal axis.** The four pure-plays are *today's*
  surviving, exchange-listed quantum names — the de-SPAC cohort that *made it* to a quote; the many
  quantum SPACs that delisted or never listed are absent, a bias that runs *bullish* (we keep the
  survivors). Using the basket therefore requires ``allow_survivorship=True`` — the desk's opt-in —
  and the caveat travels with every published number.

No look-ahead is baked into the *return accounting*: a fixed basket's weights come from membership
that does not change; the survivorship the study measures is the *cohort selection* itself, which
is the subject. No network import lives at module top.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# The four pure-play quantum-computing stocks, as the "quantum basket" is marketed in 2024-2026:
# IonQ (trapped-ion), Rigetti (superconducting), D-Wave Quantum (annealing), and Quantum Computing
# Inc. (photonic). These are *current* survivors — the de-SPAC cohort that made it to a listing; a
# 2021 investor could only have named them in hindsight, and the many quantum SPACs that delisted
# are absent (survivorship, named on the Signal axis).
BASKET = ["IONQ", "RGTI", "QBTS", "QUBT"]

# The "sane basket" control: Defiance Quantum ETF (QTUM) — a ~70-name diversified quantum / disruptive
# tech index that *includes* the pure-plays but drowns them in mega-cap compute. It is the
# diversified way to "play quantum," and the cleanest test of whether the concentrated pure-play
# bet adds anything over a fund you could actually have bought from day one.
ETF = "QTUM"

# The full FIELD the harness loads: the pure-plays + the ETF.
FIELD = BASKET + [ETF]

BENCH = "SPY"     # the market (S&P 500 total-return proxy)
TECH = "QQQ"      # the tech tape (Nasdaq-100 total-return proxy)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_periods: int = 60,             # 5 years monthly, like the real pure-play window
    n_assets: int = 5,               # 4 pure-plays + 1 ETF-like name
    n_basket: int = 4,
    edge_sharpe: float = 0.0,
    market_vol: float = 0.045,
    hype_vol: float = 0.40,          # the pure-plays' fat idiosyncratic monthly vol (the hype tail)
    market_mean: float = 0.009,
    seed: int = 395,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible monthly panel with an optional *planted risk-adjusted* basket edge.

    Each asset's monthly return is::

        r_it = beta_i * mkt_t + mu_i + eps_it

    where ``mkt_t`` is a common market factor, ``eps_it`` is idiosyncratic, and the first
    ``n_basket`` names (the *hype basket*) carry a high beta **and** a very large idiosyncratic vol
    (``hype_vol``) — the lottery signature. Their per-name drift ``mu_i`` is set so that with
    ``edge_sharpe = 0`` each name's *expected excess over the market is exactly zero* (pure beta, no
    alpha): ``mu_i = (1 - beta_i) * market_mean``. The fat ``hype_vol`` then only inflates the
    *noise*, so the spread's HAC *t* sits at the null however large a lucky-draw CAGR looks.
    ``edge_sharpe > 0`` adds a genuine annualised-Sharpe worth of drift on top.

    - ``edge_sharpe = 0`` → the hype basket has **zero** expected excess over the market but a huge
      volatility. Its *raw CAGR* can look enormous in a lucky draw while its Sharpe sits at/below
      the market's and the HAC *t* of its spread sits at the null. This is the mirage: a big number
      with no edge.
    - ``edge_sharpe > 0`` → the basket genuinely out-earns on a risk-adjusted basis; the HAC *t*
      lights up. The forward-tradable case the headline basket is *not*.

    The benchmark is the **market factor** itself (a clean index leg), so a basket's spread over it
    is the object under test. Returns ``(returns_df, bench, truth)``: ``returns_df`` is
    (n_periods x n_assets) monthly simple returns, ``bench`` the (n_periods,) benchmark series,
    ``truth`` the planted parameters and the basket columns.
    """
    rng = np.random.default_rng(seed)

    betas = np.ones(n_assets)
    betas[:n_basket] = rng.uniform(1.6, 2.4, n_basket)        # hype names are high-beta
    idio = np.full(n_assets, 0.05)
    idio[:n_basket] = hype_vol                                # ...and very fat-tailed

    # Per-name drift. Null (edge_sharpe=0): each hype name's expected EXCESS over the market is
    # exactly zero, mu_i = (1 - beta_i) * market_mean, so E[r_i - mkt] = (beta_i - 1)*market_mean +
    # mu_i = 0. The fat hype_vol then only inflates the noise -> the spread's HAC t sits at the null.
    # Edge: add `edge_sharpe` annualised-Sharpe worth of monthly drift on the basket's total vol.
    mu = np.zeros(n_assets)
    base_total_vol = np.sqrt((betas[:n_basket] * market_vol) ** 2 + idio[:n_basket] ** 2)
    mu[:n_basket] = (1.0 - betas[:n_basket]) * market_mean
    mu[:n_basket] += edge_sharpe / np.sqrt(MONTHS_PER_YEAR) * base_total_vol

    mkt = rng.normal(market_mean, market_vol, n_periods)
    eps = rng.normal(0.0, 1.0, size=(n_periods, n_assets)) * idio[None, :]
    raw = mkt[:, None] * betas[None, :] + mu[None, :] + eps

    # Decorative monthly labels via period_range (NEVER date_range(periods=BIG, freq="M") — that
    # path overflows ns-Timestamps for long spans; see CI pitfalls).
    idx = pd.period_range("2021-05", periods=n_periods, freq="M").to_timestamp()
    didx = pd.DatetimeIndex(idx, name="date")
    cols = [f"H{i:02d}" for i in range(n_basket)] + [f"X{i:02d}" for i in range(n_assets - n_basket)]
    rdf = pd.DataFrame(raw, index=didx, columns=cols)

    bench = pd.Series(mkt, index=didx, name="bench")          # the pure market factor

    truth = {
        "n_periods": n_periods,
        "n_assets": n_assets,
        "n_basket": n_basket,
        "edge_sharpe": edge_sharpe,
        "market_vol": market_vol,
        "hype_vol": hype_vol,
        "seed": seed,
        "basket_cols": cols[:n_basket],
    }
    return rdf, bench, truth


# ---------------------------------------------------------------------------
# Real tape — quantum pure-plays + QTUM + SPY/QQQ via yfinance, cache-first
# ---------------------------------------------------------------------------
def _panel_cache(cache_dir: str) -> str:
    return os.path.join(cache_dir, "quantum_panel.parquet")


def _bench_cache(cache_dir: str) -> str:
    return os.path.join(cache_dir, "quantum_bench.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_panel_cache(cache_dir)) and os.path.exists(_bench_cache(cache_dir))


def load_real(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_survivorship: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real monthly return panel for the quantum field (pure-plays + QTUM) + SPY & QQQ benchmarks.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError`` (the
    offline contract) so the reproducible core and notebooks stay deterministic — they fall back to
    the frozen ``R`` numbers exactly like the exemplar.

    **Survivorship — named on the Signal axis.** The four pure-plays are *today's* surviving,
    exchange-listed quantum names (the de-SPAC cohort that made it to a quote). The many quantum
    SPACs that delisted or never listed are absent; that bias runs **bullish** (we keep the
    survivors). Using the basket therefore requires ``allow_survivorship=True`` — the desk's opt-in
    — and the caveat (these four were not nameable, as survivors, in 2021) must travel with any
    published number.

    Returns ``(returns_df, bench_df)`` — monthly simple total returns (T x N) and a 2-column frame
    with the SPY and QQQ monthly return series aligned to the same index. The panel is trimmed to
    the window over which **all** kept names have data (the youngest pure-play sets the start, since
    the basket only exists once all four trade); the in-progress final month is dropped.
    """
    if not allow_survivorship:
        raise PermissionError(
            "load_real() returns a SURVIVING-cohort quantum basket: the four pure-plays are the "
            "de-SPAC names that made it to a listing, and the many quantum SPACs that delisted are "
            "absent (a bias that runs bullish). Pass allow_survivorship=True to opt in, and carry "
            "the caveat (these four were not nameable, as survivors, in 2021) into every number."
        )
    if tickers is None:
        tickers = FIELD

    ppath = _panel_cache(cache_dir)
    bpath = _bench_cache(cache_dir)
    if not fetch:
        if not (os.path.exists(ppath) and os.path.exists(bpath)):
            raise FileNotFoundError(
                f"No cached panel at {ppath} / {bpath}. "
                f"Call load_real(fetch=True, allow_survivorship=True) once to populate."
            )
        rets = pd.read_parquet(ppath)
        bench = pd.read_parquet(bpath)
    else:
        prices = _fetch_prices(tickers + [BENCH, TECH], start, cache_dir)
        monthly = prices.resample("ME").last()
        rets = monthly.pct_change().dropna(how="all")
        # Drop the in-progress (partial) final month — a stamped run never holds a partial bar
        # (house rule: as-of is never in the future).
        today = pd.Timestamp.today().normalize()
        rets = rets[rets.index < today.to_period("M").to_timestamp()]
        bench = rets[[BENCH, TECH]].rename(columns={BENCH: "spy", TECH: "qqq"})
        rets = rets[[c for c in tickers if c in rets.columns]]
        os.makedirs(cache_dir, exist_ok=True)
        rets.to_parquet(ppath)
        bench.to_parquet(bpath)

    keep = [c for c in (tickers or list(rets.columns)) if c in rets.columns]
    rets = rets[keep]
    # Trim to the window where every kept name has data (the youngest pure-play sets the start —
    # the basket only exists once all four trade).
    rets = rets.dropna()
    bench = bench.reindex(rets.index)
    aligned = pd.concat([rets, bench], axis=1).dropna()
    cols = [c for c in keep if c in aligned.columns]
    return aligned[cols], aligned[["spy", "qqq"]]


def _fetch_prices(tickers: list[str], start: str, cache_dir: str) -> pd.DataFrame:
    """Daily total-return closes for the field via yfinance (network on call)."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("no price data returned for the quantum field")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px.dropna(how="all")


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the return panel, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(returns.to_numpy(), nan=-9.99).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
