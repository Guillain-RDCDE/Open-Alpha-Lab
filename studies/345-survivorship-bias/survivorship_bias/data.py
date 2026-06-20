"""Data layer for Study 345 (Survivorship-Bias).

The whole study is a *measurement of a bias*, so the data layer must be able to hand the
SAME tape in two states:

- **FULL** — every firm that ever existed, including the ones that **died**: their price
  drifts down and then they *delist* (leave the tape on a near-total loss). This is the
  experience an investor actually lived through.
- **SURVIVORS** — the same panel with every dead firm dropped, as if the universe were
  built from *today's* membership and projected backwards. The losers were removed
  *before the backtest ever saw them*. This is the textbook survivorship trap.

A strategy run on SURVIVORS sees an inflated (often manufactured or sign-inverted) edge,
because the worst outcomes were deleted by construction. The delta between the two reads
is the bias — and a synthetic generator with a tunable ``death_rate`` lets us plant it
deterministically (a positive control) and switch it off (a null).

Two tapes, one shape — a (T x N) monthly-return panel plus an ``alive`` mask:

- ``synthetic_panel`` — deterministic, offline, seeded; plants delisting and (optionally)
  a true cross-sectional signal so the harness can show the bias acting *on top of* a real
  effect, a manufactured one, and a sign-flipped one.
- ``load_real``       — a real cross-section through the desk's opt-in survivorship guard,
  cache-first with a graceful offline fallback. ``allow_survivorship_bias=True`` is
  *mandatory* because the convenience universe is current-membership — exactly the bias the
  study exists to expose.

No look-ahead is baked in: the cross-sectional signal at month ``t`` is formed from data
known *at or before* ``t`` and earns the return of ``t+1`` (one ``shift``, in strategy.py).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# A small, liquid US large-cap set — the convenience "universe". These are *current*
# names, so a panel built from them and projected back is survivorship-biased: the
# delistings (Lehman, Bear, WorldCom, Enron, GM '09 …) that an honest universe carries
# are simply absent. That is precisely the bias load_real refuses to run blind to.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "XOM", "HD", "KO", "PEP", "MRK", "ABBV", "WMT", "CVX", "BAC", "DIS",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_periods: int = 240,
    n_assets: int = 200,
    death_rate: float = 0.30,
    signal_strength: float = 0.0,
    market_vol: float = 0.045,
    idio_vol: float = 0.07,
    seed: int = 345,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible monthly panel with planted *delisting* and an optional true signal.

    Construction (everything deterministic in ``seed``):

    - Each asset draws a hidden ``quality`` score ~ N(0,1). A fraction ``death_rate`` of
      the *lowest-quality* names are doomed: at a randomly drawn month they begin a steep
      terminal decline, then **delist** (their returns become NaN — they leave the tape).
      A delisting month books a large loss (a near-total wipe-out), exactly like a
      bankruptcy an index would later scrub from its membership.
    - Each asset's surviving monthly return is
      ``r_it = beta_i * mkt_t + signal_strength * quality_i + eps_it``.
      With ``signal_strength = 0`` there is **no** true cross-sectional edge — any edge a
      strategy "finds" on the survivors-only panel is manufactured by the deletion.
      With ``signal_strength != 0`` there is a real edge, and the bias acts on top of it
      (it can *inflate* a real edge or, for a contrarian rule, *invert* its sign).

    The crucial coupling: **quality drives both the true return AND who dies.** Low-quality
    names tend to drift down *and* to delist — so dropping the dead names is not random,
    it preferentially removes the left tail. That is what makes survivorship bias point
    *upward* for almost any long-biased rule.

    Returns ``(returns_df, alive_df, truth)``:
      - ``returns_df`` (T x N) monthly simple returns; a name's cells are **NaN after it
        delists** (and the delisting month carries its terminal loss).
      - ``alive_df`` (T x N) boolean mask: True while the name trades, False after delist.
      - ``truth`` records the planted parameters and the doomed-name set.
    """
    rng = np.random.default_rng(seed)

    quality = rng.normal(0.0, 1.0, n_assets)
    betas = np.clip(rng.normal(1.0, 0.25, n_assets), 0.3, 2.0)

    mkt = rng.normal(0.006, market_vol, n_periods)
    eps = rng.normal(0.0, idio_vol, size=(n_periods, n_assets))
    drift = signal_strength * quality  # constant-in-time expected-return tilt

    raw = mkt[:, None] * betas[None, :] + drift[None, :] + eps  # (T x N)

    alive = np.ones((n_periods, n_assets), dtype=bool)

    # The doomed: the lowest-quality fraction. They die at a randomly drawn month, after a
    # short terminal slide. Deaths are spread across the sample (not all at the end), so
    # the survivor panel really is missing names throughout.
    n_dead = int(round(death_rate * n_assets))
    if n_dead > 0:
        doomed = np.argsort(quality)[:n_dead]            # worst names
        # Death month uniformly in the back two-thirds of the sample (a name must trade a
        # while before it can blow up).
        first = max(6, n_periods // 3)
        # Strictly before the final bar, so a doomed name is always truly gone from the
        # SURVIVORS view (a death on the very last month would never get masked).
        death_month = rng.integers(first, n_periods - 1, size=n_dead)
        for j, a in enumerate(doomed):
            dm = int(death_month[j])
            # A terminal slide over the ~6 months before the wipe-out.
            slide_start = max(0, dm - 6)
            raw[slide_start:dm, a] -= 0.04  # steady bleed
            raw[dm, a] = -0.80              # the delisting loss (near-total wipe)
            alive[dm + 1:, a] = False       # gone from the tape thereafter
    else:
        doomed = np.array([], dtype=int)

    returns = np.where(alive, raw, np.nan)
    # The delisting month itself is the last LIVE bar (it carries the wipe-out), so re-mark
    # alive at the death month as True (it already is) and only mask strictly after.

    # Decorative monthly labels via period_range (NEVER date_range(periods=BIG, freq="M")
    # — that overflows ns-Timestamps for long spans; see CI pitfalls).
    idx = pd.period_range("1990-01", periods=n_periods, freq="M").to_timestamp()
    cols = [f"A{i:03d}" for i in range(n_assets)]
    didx = pd.DatetimeIndex(idx, name="date")
    rdf = pd.DataFrame(returns, index=didx, columns=cols)
    adf = pd.DataFrame(alive, index=didx, columns=cols)

    truth = {
        "n_periods": n_periods,
        "n_assets": n_assets,
        "death_rate": death_rate,
        "n_dead": int(n_dead),
        "signal_strength": signal_strength,
        "seed": seed,
        "doomed_cols": [cols[a] for a in doomed],
        "quality": quality.tolist(),
    }
    return rdf, adf, truth


def survivors_only(returns: pd.DataFrame, alive: pd.DataFrame) -> pd.DataFrame:
    """The survivorship-biased view: keep ONLY names that are still alive at the end.

    This is exactly what a current-membership universe gives you — every firm that delisted
    before today is gone, so the panel you backtest on never contained the losers. Columns
    that delist (any False in the final row of ``alive``) are dropped entirely, and the
    survivors keep their full history (no NaNs to mask).
    """
    survives = alive.iloc[-1]
    keep = survives[survives].index
    return returns[keep]


# ---------------------------------------------------------------------------
# Real tape — large US cross-section via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "survivorship_panel.parquet")


def load_real(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_survivorship_bias: bool = False,
) -> pd.DataFrame:
    """Real monthly return panel for the convenience large-cap universe.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract) so the reproducible core and tests stay deterministic.

    **Survivorship — the whole point.** ``tickers`` defaults to *current* large-cap members
    projected back, so the panel is itself survivorship-biased (the failures that fell out
    of the index are absent). The study uses it as a *real illustration* of the bias the
    synthetic panel plants; using it therefore requires ``allow_survivorship_bias=True`` —
    the desk's opt-in (``quantlab.universe`` / ``hf_data``) — and the caveat must travel
    with any published number. Here the bias is the subject, not a nuisance.

    Returns a (T x N) DataFrame of monthly simple total returns.
    """
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real() builds a CURRENT-membership cross-section projected backwards, "
            "which IS the survivorship trap this study dissects. Pass "
            "allow_survivorship_bias=True to opt in (and keep the caveat with the numbers)."
        )
    if tickers is None:
        tickers = DEFAULT_UNIVERSE

    rpath = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(rpath):
            raise FileNotFoundError(
                f"No cached panel at {rpath}. "
                f"Call load_real(fetch=True, allow_survivorship_bias=True) once to populate."
            )
        rets = pd.read_parquet(rpath)
    else:
        prices = _fetch_prices(tickers, start, cache_dir)
        monthly = prices.resample("ME").last()
        rets = monthly.pct_change().dropna(how="all")
        # Drop the in-progress (partial) final month — a stamped run never holds a partial
        # bar (house rule: as-of is never in the future).
        today = pd.Timestamp.today().normalize()
        rets = rets[rets.index < today.to_period("M").to_timestamp()]
        os.makedirs(cache_dir, exist_ok=True)
        rets.to_parquet(rpath)

    rets = rets[[c for c in tickers if c in rets.columns]].dropna(how="all")
    return rets


def _fetch_prices(tickers: list[str], start: str, cache_dir: str) -> pd.DataFrame:
    """Daily total-return closes for the universe via the shared loader (network on call)."""
    try:
        from quantlab import data as qld

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

    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("no price data returned for the survivorship universe")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px.dropna(how="all")


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the panel, for the as-of stamp.

    NaNs (delisted cells) are mapped to a sentinel so the hash is stable across reruns.
    """
    arr = np.ascontiguousarray(np.nan_to_num(returns.to_numpy(), nan=-9.99).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
