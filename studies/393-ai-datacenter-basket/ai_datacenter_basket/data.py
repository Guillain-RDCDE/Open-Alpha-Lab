"""Data layer for Study 393 (AI-Datacenter-Basket).

The "AI build-out basket" is a *cross-sectional selection* claim dressed as a strategy: from a
large field of plausible "datacenter-and-power" candidates (chips, cooling/electrical gear,
networking, servers, the utilities that sell the electricity, the REITs that own the buildings),
eight names — NVDA, VRT, ETN, CEG, VST, SMCI, ANET, DELL — are held equal-weight and raced
against the market (SPY) and the tech tape (QQQ). The catch is that those eight are *named because
they won* — they are the ex-post top-of-the-pile of the 2019-2026 sample. So the data layer must
supply a **panel** (a T x N monthly-return matrix of the candidate FIELD, not just the eight) plus
the benchmarks, so the harness can ask the decisive question: how much of the basket's spread is a
real, forward-tradable theme tilt, and how much is *selecting the winners after the fact*?

Two tapes, one shape (a return panel + benchmark return series):

- ``synthetic_panel`` — a *deterministic, offline* generator. A single knob, ``alpha_spread``,
  plants the one thing a real theme would have: a *persistent* expected-return tilt on a fixed,
  pre-specified set of names. At ``alpha_spread=0`` every name has the same expected return, so any
  basket that "beats" the index is pure luck — and the ex-post "pick the winners" rule still
  manufactures a large positive spread from selection alone (the placebo the study exists to
  expose). At ``alpha_spread>0`` a *named-in-advance* basket really does out-earn — the forward-
  tradable case the headline basket is *not*. This is the study's null and positive control in one.

- ``load_real`` — a real monthly panel of the candidate field plus the SPY / QQQ benchmarks, via
  yfinance, cache-first (parquet) with a graceful offline contract. **Survivorship / look-ahead is
  the whole point.** The field and the eight-name membership are *today's* knowledge projected back
  to 2019, so any "edge" is at least partly winners-known-in-hindsight bias — named on the Signal
  axis and carried into the verdict. ``allow_lookahead_selection=True`` is mandatory, the desk's
  opt-in mirroring the survivorship guard.

No look-ahead is baked into the *return accounting*: a fixed basket's weights are formed from
membership that does not change, and the look-ahead the study measures is the *membership
selection* itself, which is the subject. No network import lives at module top.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# The eight, as the "AI datacenter + power" basket is marketed in 2024-2026: the GPU (NVDA), the
# power/cooling/electrical gear (VRT, ETN), the merchant power utilities that sell the electricity
# (CEG, VST), the AI server box (SMCI, DELL), the datacenter switch fabric (ANET). These are
# *current* labels — naming exactly these eight requires knowing 2019-2026 returns, which is
# precisely the look-ahead the study dissects.
BASKET = ["NVDA", "VRT", "ETN", "CEG", "VST", "SMCI", "ANET", "DELL"]

# A larger candidate FIELD the eight were drawn from — the set a 2019 investor could plausibly have
# called "ways to play the datacenter / power build-out." It spans chips & equipment (NVDA, AMD,
# AVGO, MU, AMAT, LRCX, SMCI, DELL, HPE, ANET, CSCO, JNPR), power/electrical/cooling/industrials
# (VRT, ETN, EMR, PWR, HUBB, JCI, CARR, GEV), utilities that sell datacenter electricity (CEG, VST,
# NRG, NEE, D, SO, DUK, AEP, EXC), and datacenter REITs / infra (EQIX, DLR, AMT). The headline
# eight are the ex-post top of THIS field; the field is what makes the selection visible. (Itself
# current-membership, so still survivorship-tilted upward — names that delisted are absent; named
# on the axis. GEV/CEG/VST are post-spin tickers with short histories — the loader trims to the
# common window and reports the count.)
FIELD = [
    "NVDA", "VRT", "ETN", "CEG", "VST", "SMCI", "ANET", "DELL",        # the headline eight
    "AMD", "AVGO", "MU", "AMAT", "LRCX", "HPE", "CSCO", "JNPR",        # more chips & gear
    "EMR", "PWR", "HUBB", "JCI", "CARR",                              # electrical / cooling / power-build
    "NRG", "NEE", "D", "SO", "DUK", "AEP", "EXC",                     # utilities (sell the electricity)
    "EQIX", "DLR", "AMT",                                             # datacenter REITs / infra
]

BENCH = "SPY"     # the market (S&P 500 total-return proxy)
TECH = "QQQ"      # the tech tape (Nasdaq-100 total-return proxy)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_periods: int = 84,             # 7 years monthly, like 2019-2026
    n_assets: int = 31,
    n_basket: int = 8,
    alpha_spread: float = 0.0,
    market_vol: float = 0.045,
    idio_vol: float = 0.075,
    seed: int = 393,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible monthly panel with an optional *named-in-advance* basket premium.

    Each asset's monthly return is::

        r_it = beta_i * mkt_t + alpha_i + eps_it

    where ``mkt_t`` is a common market factor, ``eps_it`` is idiosyncratic, and ``alpha_i`` is a
    constant per-name expected-return tilt that is **non-zero only for the first ``n_basket`` names**
    (the *true*, pre-specified theme basket) and equal to ``alpha_spread`` there.

    - ``alpha_spread = 0`` → every name has the same expected return. A *pre-named* basket and the
      index have the same expected return (its spread is luck, mean ~0). But the ex-post "pick the
      ``n_basket`` highest-realised-return names" rule **still** beats the index by a large margin —
      that spread is manufactured by selection. This is the placebo.
    - ``alpha_spread > 0`` → the *pre-named* basket genuinely out-earns (a real, forward-tradable
      theme). The ex-post basket beats it by even more (real tilt + selection noise).

    The benchmark is the **equal-weight** mean of all ``n_assets`` names (a clean 1/N index of the
    same field), so a basket's spread over it isolates name selection, not weighting.

    Returns ``(returns_df, bench, truth)``: ``returns_df`` is (n_periods x n_assets) monthly simple
    returns, ``bench`` is the (n_periods,) benchmark return series, ``truth`` records the planted
    parameters and the *true* (pre-named) basket columns.
    """
    rng = np.random.default_rng(seed)

    betas = np.clip(rng.normal(1.05, 0.25, n_assets), 0.4, 2.0)
    alpha = np.zeros(n_assets)
    alpha[:n_basket] = alpha_spread                       # only the pre-named basket has a tilt

    mkt = rng.normal(0.008, market_vol, n_periods)        # common factor
    eps = rng.normal(0.0, idio_vol, size=(n_periods, n_assets))
    raw = mkt[:, None] * betas[None, :] + alpha[None, :] + eps

    # Decorative monthly labels via period_range (NEVER date_range(periods=BIG, freq="M") — that
    # path overflows ns-Timestamps for long spans; see CI pitfalls).
    idx = pd.period_range("2019-01", periods=n_periods, freq="M").to_timestamp()
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
# Real tape — candidate field + SPY/QQQ via yfinance, cache-first
# ---------------------------------------------------------------------------
def _panel_cache(cache_dir: str) -> str:
    return os.path.join(cache_dir, "datacenter_panel.parquet")


def _bench_cache(cache_dir: str) -> str:
    return os.path.join(cache_dir, "datacenter_bench.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_panel_cache(cache_dir)) and os.path.exists(_bench_cache(cache_dir))


def load_real(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_lookahead_selection: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real monthly return panel for the candidate field + the SPY & QQQ benchmarks.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError`` (the
    offline contract) so the reproducible core and notebooks stay deterministic — they fall back to
    the frozen ``R`` numbers exactly like the exemplar.

    **Look-ahead / survivorship — the whole point.** Both the field and the eight-name membership
    are *current* knowledge projected back to ``start``. The eight are named *because* they won
    2019-2026; the field excludes any candidate that delisted. Using the ex-post-selected basket
    therefore requires ``allow_lookahead_selection=True`` — the desk's opt-in mirroring the
    survivorship guard — and the caveat (these eight were not nameable in 2019) must travel with any
    published number.

    Returns ``(returns_df, bench_df)`` — monthly simple total returns (T x N) and a 2-column frame
    with the SPY and QQQ monthly return series aligned to the same index. The panel is trimmed to
    the window over which **all** kept names have data (post-spin tickers like CEG/VST/GEV shorten
    it); names with too little history are dropped and the kept count is reported by the caller.
    """
    if not allow_lookahead_selection:
        raise PermissionError(
            "load_real() returns a CURRENT-membership datacenter/power candidate field projected "
            "back to 2019. The eight-name basket itself is named with the full sample in view "
            "(look-ahead), and the field excludes names that delisted. Pass "
            "allow_lookahead_selection=True to opt in, and carry the caveat (these eight were not "
            "nameable in 2019) into every number."
        )
    if tickers is None:
        tickers = FIELD

    ppath = _panel_cache(cache_dir)
    bpath = _bench_cache(cache_dir)
    if not fetch:
        if not (os.path.exists(ppath) and os.path.exists(bpath)):
            raise FileNotFoundError(
                f"No cached panel at {ppath} / {bpath}. "
                f"Call load_real(fetch=True, allow_lookahead_selection=True) once to populate."
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
    # Trim to the window where every kept name has data; drop names too short to keep that
    # window usable (a name present <50% of the field's span is dropped, not kept as a stub).
    span = len(rets)
    enough = [c for c in keep if rets[c].notna().mean() >= 0.50]
    rets = rets[enough].dropna()
    bench = bench.reindex(rets.index)
    aligned = pd.concat([rets, bench], axis=1).dropna()
    cols = [c for c in enough if c in aligned.columns]
    return aligned[cols], aligned[["spy", "qqq"]]


def _fetch_prices(tickers: list[str], start: str, cache_dir: str) -> pd.DataFrame:
    """Daily total-return closes for the field via the shared loader (network on call)."""
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
        raise RuntimeError("no price data returned for the datacenter/power field")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px.dropna(how="all")


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the return panel, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(returns.to_numpy(), nan=-9.99).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
