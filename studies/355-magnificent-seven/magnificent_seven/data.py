"""Data layer for Study 355 (Magnificent-Seven).

The "Mag 7" is a *cross-sectional selection* claim dressed as a strategy: from a large field
of US mega-caps, seven names (AAPL, MSFT, GOOGL/GOOG, AMZN, NVDA, META, TSLA) are held
equal-weight and raced against the S&P 500. The catch is that the seven are *named because they
won* — they are the ex-post top-of-the-pile of the 2015-2026 sample. So the data layer must
supply a **panel** (a T x N monthly-return matrix of a mega-cap UNIVERSE, not just the seven)
plus the benchmark, so the harness can ask the decisive question: how much of the Mag-7 spread
is a real, forward-tradable tilt, and how much is *selecting the winners after the fact*?

Two tapes, one shape (a return panel + a benchmark return series):

- ``synthetic_panel`` — a *deterministic, offline* generator. A single knob, ``alpha_spread``,
  plants the one thing a real factor would have: a *persistent* expected-return tilt on a
  fixed, pre-specified set of names. At ``alpha_spread=0`` every name has the same expected
  return, so any basket that "beats" the index is pure luck — and the ex-post "pick the 7
  winners" rule still manufactures a large positive spread from selection alone (the placebo
  the study exists to expose). At ``alpha_spread>0`` a *named-in-advance* basket really does
  out-earn — the forward-tradable case the Mag 7 is *not*. This is the study's null and its
  positive control in one bottle.

- ``load_real`` — a real monthly panel of the mega-cap universe plus the SPY benchmark, via
  yfinance, cache-first (parquet) with a graceful offline fallback. **Survivorship / look-ahead
  is the whole point.** The universe and the Mag-7 membership are *today's* knowledge projected
  back to 2015, so any "edge" is at least partly the winners-known-in-hindsight bias — named on
  the Signal axis and carried into the verdict. ``allow_lookahead_selection=True`` is mandatory
  for the ex-post-selected basket, the desk's opt-in mirroring the survivorship guard.

No look-ahead is baked into the *return accounting*: a basket's weights for month ``t+1`` are
formed from membership known *before* ``t+1`` (one execution lag where a signal is used). The
look-ahead the study measures is the *membership selection* itself, which is the subject.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# The seven, as the term is used in 2024-2026. GOOG/GOOGL are the two Alphabet share classes;
# we keep GOOGL (the term means "Alphabet" once). These are *current* labels — naming them
# requires knowing 2015-2026 returns, which is exactly the look-ahead the study dissects.
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# A larger mega-cap FIELD the seven were drawn from — the set a 2015 investor could plausibly
# have considered "the big liquid US names." It includes winners AND also-rans (INTC, CSCO, IBM,
# DIS, KO, XOM, PFE, T, VZ, GE…). The Mag 7 is the ex-post top of THIS field; the field is what
# makes the selection visible. (Itself current-membership, so still survivorship-tilted upward —
# the dead names of 2015-2026 are absent; named on the axis.)
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",          # the seven
    "JPM", "JNJ", "V", "PG", "XOM", "HD", "KO", "PEP", "MRK",         # steady mega-caps
    "WMT", "CVX", "BAC", "DIS", "CSCO", "INTC", "IBM", "ORCL", "QCOM",
    "PFE", "T", "VZ", "GE", "CMCSA", "ADBE", "CRM", "NFLX", "PYPL",
    "WFC", "MCD", "NKE", "TXN", "AMD", "C",
]

BENCH = "SPY"   # S&P 500 total-return proxy


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_periods: int = 132,            # 11 years monthly, like 2015-2026
    n_assets: int = 40,
    n_basket: int = 7,
    alpha_spread: float = 0.0,
    market_vol: float = 0.040,
    idio_vol: float = 0.060,
    seed: int = 355,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible monthly panel with an optional *named-in-advance* basket premium.

    Each asset's monthly return is::

        r_it = beta_i * mkt_t + alpha_i + eps_it

    where ``mkt_t`` is a common market factor, ``eps_it`` is idiosyncratic, and ``alpha_i`` is
    a constant per-name expected-return tilt that is **non-zero only for the first ``n_basket``
    names** (the *true*, pre-specified factor basket) and equal to ``alpha_spread`` there.

    - ``alpha_spread = 0`` → every name has the same expected return. A *pre-named* basket and
      the index have the same expected return (its spread is luck, mean ~0). But the ex-post
      "pick the ``n_basket`` highest-realised-return names" rule **still** beats the index by a
      large margin — that spread is manufactured by selection. This is the placebo.
    - ``alpha_spread > 0`` → the *pre-named* basket genuinely out-earns (a real, forward-
      tradable factor). The ex-post basket beats it by even more (real tilt + selection noise).

    The benchmark is the **equal-weight** mean of all ``n_assets`` names (a clean 1/N index of
    the same universe), so a basket's spread over it isolates name selection, not weighting.

    Returns ``(returns_df, bench, truth)``: ``returns_df`` is (n_periods x n_assets) monthly
    simple returns, ``bench`` is the (n_periods,) benchmark return series, ``truth`` records the
    planted parameters and the *true* (pre-named) basket columns.
    """
    rng = np.random.default_rng(seed)

    betas = np.clip(rng.normal(1.0, 0.20, n_assets), 0.4, 1.8)
    alpha = np.zeros(n_assets)
    alpha[:n_basket] = alpha_spread                       # only the pre-named basket has a tilt

    mkt = rng.normal(0.006, market_vol, n_periods)        # common factor
    eps = rng.normal(0.0, idio_vol, size=(n_periods, n_assets))
    raw = mkt[:, None] * betas[None, :] + alpha[None, :] + eps

    # Decorative monthly labels via period_range (NEVER date_range(periods=BIG, freq="M") —
    # that path overflows ns-Timestamps for long spans; see CI pitfalls).
    idx = pd.period_range("2015-01", periods=n_periods, freq="M").to_timestamp()
    didx = pd.DatetimeIndex(idx, name="date")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    rdf = pd.DataFrame(raw, index=didx, columns=cols)

    bench = pd.Series(raw.mean(axis=1), index=didx, name="bench")  # equal-weight 1/N index

    truth = {
        "n_periods": n_periods,
        "n_assets": n_assets,
        "n_basket": n_basket,
        "alpha_spread": alpha_spread,
        "market_vol": market_vol,
        "idio_vol": idio_vol,
        "seed": seed,
        "true_basket_cols": cols[:n_basket],
    }
    return rdf, bench, truth


# ---------------------------------------------------------------------------
# Real tape — mega-cap universe + SPY via yfinance, cache-first
# ---------------------------------------------------------------------------
def _panel_cache(cache_dir: str) -> str:
    return os.path.join(cache_dir, "mag7_panel.parquet")


def _bench_cache(cache_dir: str) -> str:
    return os.path.join(cache_dir, "mag7_bench.parquet")


def load_real(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_lookahead_selection: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Real monthly return panel for the mega-cap universe + the SPY benchmark.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError`` (the
    offline contract) so the reproducible core and the notebooks stay deterministic — they fall
    back to the frozen ``R`` numbers exactly like the exemplar.

    **Look-ahead / survivorship — the whole point.** Both the universe and the Mag-7 membership
    are *current* knowledge projected back to ``start``. The seven are named *because* they won
    2015-2026; the field excludes any mega-cap that died on the way. Using the ex-post-selected
    basket therefore requires ``allow_lookahead_selection=True`` — the desk's opt-in mirroring
    the survivorship guard — and the caveat (the seven were not nameable in 2015) must travel
    with any published number. Loading the *panel* itself does not require the flag: forming a
    pre-specified or random basket from it is honest; only the "pick the winners" rule is the
    trap, enforced in :mod:`magnificent_seven.strategy`.

    Returns ``(returns_df, bench)`` — monthly simple total returns (T x N) and the SPY monthly
    return series aligned to the same index.
    """
    if not allow_lookahead_selection:
        raise PermissionError(
            "load_real() returns a CURRENT-membership mega-cap field projected back to 2015. "
            "The Mag-7 basket itself is named with the full sample in view (look-ahead), and "
            "the field excludes names that died. Pass allow_lookahead_selection=True to opt in, "
            "and carry the caveat (these seven were not nameable in 2015) into every number."
        )
    if tickers is None:
        tickers = UNIVERSE

    ppath = _panel_cache(cache_dir)
    bpath = _bench_cache(cache_dir)
    if not fetch:
        if not (os.path.exists(ppath) and os.path.exists(bpath)):
            raise FileNotFoundError(
                f"No cached panel at {ppath} / {bpath}. "
                f"Call load_real(fetch=True, allow_lookahead_selection=True) once to populate."
            )
        rets = pd.read_parquet(ppath)
        bench = pd.read_parquet(bpath)["bench"]
    else:
        prices = _fetch_prices(tickers + [BENCH], start, cache_dir)
        monthly = prices.resample("ME").last()
        rets = monthly.pct_change().dropna(how="all")
        # Drop the in-progress (partial) final month — a stamped run never holds a partial bar
        # (house rule: as-of is never in the future).
        today = pd.Timestamp.today().normalize()
        rets = rets[rets.index < today.to_period("M").to_timestamp()]
        bench = rets[BENCH].rename("bench")
        rets = rets[[c for c in tickers if c in rets.columns]]
        os.makedirs(cache_dir, exist_ok=True)
        rets.to_parquet(ppath)
        bench.to_frame().to_parquet(bpath)

    keep = [c for c in (tickers or list(rets.columns)) if c in rets.columns]
    rets = rets[keep].dropna(how="all")
    bench = bench.reindex(rets.index)
    aligned = pd.concat([rets, bench], axis=1).dropna()
    return aligned[keep], aligned["bench"]


def _fetch_prices(tickers: list[str], start: str, cache_dir: str) -> pd.DataFrame:
    """Daily total-return closes for the universe via the shared loader (network on call)."""
    try:
        from quantlab import data as qld  # shared loader, if importable

        frames = {}
        for t in tickers:
            try:
                bars = qld.load(t, start=start, mode="total_return")
                frames[t] = bars["close"] if "close" in bars else bars.iloc[:, 0]
            except Exception:
                continue
        if frames:
            px = pd.DataFrame(frames)
            if px.index.tz is not None:
                px.index = px.index.tz_localize(None)
            return px.dropna(how="all")
    except Exception:
        pass

    # Fallback: yfinance directly (auto_adjust → total-return-ish adjusted closes).
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("no price data returned for the mega-cap universe")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px.dropna(how="all")


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the return panel, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(returns.to_numpy(), nan=-9.99).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
