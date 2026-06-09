"""Data access for the gamma study — the **regime panel** and where it comes from.

One object runs the whole study: a per-**session** frame, one row per trading day, carrying just
enough to test the GEX pitch:

    gex        signed net dealer gamma exposure at the PRIOR close ($bn per 1% move)
    neg_gamma  gex < 0  -> dealers short gamma, the "amplifier" the pitch warns about
    vix        the PRIOR-close VIX                     -> the confound the study turns on
    rv         the day's range-based (Parkinson) vol   -> tests "negative gamma => more vol"
    de         |close-open| / (high-low), in [0,1]      -> trend-vs-chop: the "character" of the day

The GEX sign is computed under the standard SqueezeMetrics / retail-dashboard **dealer
convention**: customers buy index puts (protection) and overwrite calls, so dealers are *long*
call gamma and *short* put gamma —

    GEX = sum_calls(gamma * OI * 100 * spot^2)  -  sum_puts(gamma * OI * 100 * spot^2)

That convention is an *assumption* about an unobservable (real dealer inventory); it is the
study's single load-bearing modelling choice and is flagged as such in the README. Everything
downstream only needs the **sign** and the per-strike walls, not the exact dollar scale.

This module produces the panel two ways, the desk's standing split:

    * :func:`synthetic_panel` — fully **offline**. Toy sessions in which a persistent VIX regime
      drives *both* the GEX sign *and* the realised character (the confound), plus an *independent*
      genuine gamma effect of size ``beta`` layered on top. That ground truth is exactly what the
      tests assert the decomposition recovers — a raw regime gap that mostly survives or mostly
      collapses once VIX is partialled out, depending on ``beta``. No network in CI.
    * :func:`compute_gex` + :func:`fetch_chain` + :func:`build_panel` — reduce a real SPY option
      chain (Alpha Vantage ``HISTORICAL_OPTIONS``) to a signed GEX and walls, cache each chain to
      parquet, and merge with daily SPY/VIX bars into the same panel. **Cache-only** unless
      ``fetch=True``: a missing cache is skipped, never a silent stall or re-download.

``HISTORICAL_OPTIONS`` is a **premium** Alpha Vantage endpoint (the free key is rejected, and the
free options sources we checked — DoltHub's ``post-no-preference/options``, OptionsDX — carry greeks
but **no open interest**, so they cannot weight a GEX). One request is one (symbol, date) chain;
:func:`fetch_chain` caps each run and states the cap out loud (house rule: no silent caps). History
reaches back to 2008-01-01. :func:`snapshot_chain` is the free, key-less alternative — yfinance's
*live* chain — but it is a snapshot (no history) and its open interest is often sparse/unreliable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

# The canonical regime-panel columns every consumer (signals, decompose) expects.
PANEL_COLUMNS = ["spot", "gex", "neg_gamma", "vix", "open", "high", "low", "close", "rv", "de"]

# Alpha Vantage HISTORICAL_OPTIONS contract fields we rely on (values arrive as strings).
_AV_FIELDS = ("type", "strike", "gamma", "open_interest", "expiration")


# --------------------------------------------------------------------------- #
# Realised character — pure functions of one day's OHLC
# --------------------------------------------------------------------------- #

def parkinson_vol(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980) range estimator of one day's volatility: ``ln(H/L) / (2*sqrt(ln 2))``.

    A range-based vol proxy that needs only the day's high and low — abundant, and far more
    efficient than a single close-to-close return. Monotone in the day's true vol, which is all
    the regime test needs.
    """
    hl = np.log(np.asarray(high, float) / np.asarray(low, float))
    return hl / (2.0 * np.sqrt(np.log(2.0)))


def directional_efficiency(open_, high, low, close) -> np.ndarray:
    """``|close - open| / (high - low)`` in [0, 1] — how much of the day's range was *directional*.

    Near 1 the day went one way and stayed (a **trend** day); near 0 it travelled a wide range but
    closed where it opened (a **range / chop** day). This is the price-only stand-in for the GEX
    pitch's central claim: positive gamma => range (low DE), negative gamma => trend (high DE).
    """
    rng = np.asarray(high, float) - np.asarray(low, float)
    body = np.abs(np.asarray(close, float) - np.asarray(open_, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        de = np.where(rng > 0, body / rng, np.nan)
    return np.clip(de, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, with a VIX-driven confound and an independent gamma effect
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the decomposition."""
    beta_vol: float       # genuine extra Parkinson vol on a negative-gamma day, *beyond* the VIX regime
    beta_de: float        # genuine extra directional efficiency (trendiness) on a negative-gamma day
    confounded: bool      # True: VIX drives BOTH the GEX sign and the realised character
    n_sessions: int


def synthetic_panel(
    n_sessions: int = 750,
    beta_vol: float = 0.0020,
    beta_de: float = 0.060,
    vix_mean: float = 18.0,
    vix_persist: float = 0.94,
    vix_sigma: float = 2.2,
    gamma_confound: float = 0.35,
    rv_vix_slope: float = 0.00065,
    de_vix_slope: float = 0.012,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A toy book of sessions with a **known** genuine gamma effect and a VIX-driven confound.

    The data-generating process, in order:

    * a persistent **VIX** regime ``vix_d`` (AR(1), mean ``vix_mean``, persistence ``vix_persist``)
      — *observable*, and the thing the real study controls for;
    * the **GEX sign**: ``P(neg_gamma_d) = logistic(gamma_confound * (vix_d - vix_mean))`` — so a
      high-VIX day is more likely to be negative-gamma (put-heavy). This is the **confound**: VIX
      moves the regime label;
    * the realised **vol** ``rv_d = rv0 + rv_vix_slope*vix_d + beta_vol*neg_gamma_d + noise`` and
      **directional efficiency** ``de_d = clip(de0 + de_vix_slope*vix_d + beta_de*neg_gamma_d +
      noise, 0, 1)``. Both rise with VIX (the confound) *and*, by ``beta_*``, with negative gamma
      **independently** of VIX — the genuine effect the pitch needs.

    So the *raw* gap "negative-gamma days are more vol / more trending" is real by construction, but
    only the ``beta_*`` part survives once VIX is partialled out. Set ``beta_vol = beta_de = 0`` and
    the raw gap is **pure confound** — present, but vanishing under the VIX control. That pair of
    behaviours is exactly what :mod:`decompose` must distinguish, and what the tests assert.

    Returns ``(panel, truth)`` — ``panel`` is the canonical regime panel (one row per day) on a
    business-day index; ``truth`` carries the baked-in betas. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2022-01-03", periods=n_sessions, name="date")

    # 1) Persistent VIX regime (AR(1)), floored to a sane level.
    vix = np.empty(n_sessions)
    vix[0] = vix_mean
    for t in range(1, n_sessions):
        vix[t] = vix_mean + vix_persist * (vix[t - 1] - vix_mean) + rng.normal(0, vix_sigma)
    vix = np.clip(vix, 9.0, 80.0)

    # 2) GEX sign driven by VIX (the confound) + idiosyncratic noise.
    logit = gamma_confound * (vix - vix_mean) + rng.normal(0, 1.0, n_sessions)
    neg_gamma = logit > 0.0                                   # high VIX -> more likely short-gamma
    # A signed GEX magnitude purely for display realism (sign is what matters).
    gex = np.where(neg_gamma, -1.0, 1.0) * np.abs(rng.normal(3.0, 1.2, n_sessions))

    # 3) Realised character: a VIX trend + an INDEPENDENT gamma effect (beta) + noise.
    rv = 0.0090 + rv_vix_slope * vix + beta_vol * neg_gamma + rng.normal(0, 0.0016, n_sessions)
    rv = np.clip(rv, 1e-4, None)
    de = 0.30 + de_vix_slope * (vix - vix_mean) + beta_de * neg_gamma + rng.normal(0, 0.10, n_sessions)
    de = np.clip(de, 0.0, 1.0)

    # A plausible OHLC consistent with (rv, de) so compute-from-OHLC paths have something to chew on.
    spot = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n_sessions)))
    rng_abs = rv * spot * 2.0 * np.sqrt(np.log(2.0))         # invert Parkinson: H/L spread
    high = spot + rng_abs / 2.0
    low = spot - rng_abs / 2.0
    sgn = rng.choice([-1.0, 1.0], n_sessions)
    open_ = spot - sgn * de * rng_abs / 2.0
    close = spot + sgn * de * rng_abs / 2.0

    panel = pd.DataFrame(
        {
            "spot": spot, "gex": gex, "neg_gamma": pd.array(neg_gamma, dtype="boolean"),
            "vix": vix, "open": open_, "high": high, "low": low, "close": close,
            "rv": rv, "de": de,
        },
        index=idx,
    )[PANEL_COLUMNS]

    truth = GroundTruth(beta_vol=beta_vol, beta_de=beta_de, confounded=gamma_confound > 0,
                        n_sessions=n_sessions)
    return panel, truth


# --------------------------------------------------------------------------- #
# Real tape — the GEX reducer, the Alpha Vantage chain fetch, and the panel build
# --------------------------------------------------------------------------- #

def compute_gex(chain: pd.DataFrame, spot: float, multiplier: float = 100.0) -> dict:
    """Reduce one option chain to a signed net GEX and the call/put walls.

    Under the dealer convention (long call gamma, short put gamma), each contract's dollar gamma is
    ``gamma * open_interest * multiplier * spot^2``; calls add, puts subtract. Returns the net GEX
    (scaled to **$bn per 1% move**, i.e. ``* 1e-2 / 1e9``), the per-strike call/put **walls**
    (the strikes with the largest one-sided dollar gamma — the pitch's ceiling/floor), and a crude
    **gamma flip** (the strike where cumulative net dollar gamma crosses zero). Only the *sign* of
    ``gex`` is load-bearing downstream; the walls/flip feed the going-further checks.

    ``chain`` must carry ``type`` (call/put), ``strike``, ``gamma``, ``open_interest``.
    """
    nan_out = {"gex": float("nan"), "call_wall": float("nan"),
               "put_wall": float("nan"), "gamma_flip": float("nan")}
    if chain.empty:
        return nan_out
    df = chain.copy()
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df = df[np.isfinite(df["gamma"]) & np.isfinite(df["open_interest"])]
    if df.empty:
        return nan_out

    is_call = df["type"].astype(str).str.lower().str.startswith("c")
    dollar_gamma = df["gamma"].to_numpy(float) * df["open_interest"].to_numpy(float) \
        * multiplier * (spot ** 2)
    signed = np.where(is_call.to_numpy(), dollar_gamma, -dollar_gamma)
    gex = float(signed.sum()) * 1e-2 / 1e9                  # $bn per 1% move

    calls = df[is_call.to_numpy()]
    puts = df[~is_call.to_numpy()]
    call_dollar = calls["gamma"].to_numpy(float) * calls["open_interest"].to_numpy(float)
    put_dollar = puts["gamma"].to_numpy(float) * puts["open_interest"].to_numpy(float)
    call_wall = float(calls["strike"].iloc[int(np.argmax(call_dollar))]) if len(calls) else float("nan")
    put_wall = float(puts["strike"].iloc[int(np.argmax(put_dollar))]) if len(puts) else float("nan")

    # Gamma flip: scan strikes, find where cumulative net dollar gamma changes sign.
    by_strike = (pd.Series(signed, index=df["strike"].to_numpy(float))
                 .groupby(level=0).sum().sort_index())
    cum = by_strike.cumsum()
    flip = float("nan")
    sign_change = np.where(np.diff(np.sign(cum.to_numpy())) != 0)[0]
    if sign_change.size:
        flip = float(cum.index[sign_change[0]])

    return {"gex": gex, "call_wall": call_wall, "put_wall": put_wall, "gamma_flip": flip}


def bs_gamma(spot, strike, t_years, iv, r: float = 0.0):
    """Black-Scholes gamma, vectorised: ``φ(d1) / (S·σ·√T)``.

    The real (free) data path gets per-contract gamma from each option's *implied vol* and
    days-to-expiry rather than a data vendor's pre-computed greek. ``t_years`` is floored at half a
    day and ``iv`` at a tiny positive so 0DTE / zero-IV rows don't blow up. ``r`` ≈ 0 is fine for
    the *sign* of net GEX, which is all that's load-bearing.
    """
    t = np.maximum(np.asarray(t_years, float), 0.5 / 365.0)
    iv = np.maximum(np.asarray(iv, float), 1e-4)
    spot = float(spot)
    d1 = (np.log(spot / np.asarray(strike, float)) + (r + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)
    return pdf / (spot * iv * np.sqrt(t))


def snapshot_chain(symbol: str = "SPY", max_dte: int = 60, fetch: bool = False):
    """Pull the **live** option chain via yfinance and reduce it to a GEX-ready frame.

    Free and key-less, but a **snapshot** (no history): yfinance serves only the *current* chain.
    Returns ``(chain, spot)`` where ``chain`` has the ``compute_gex`` columns
    (``type``/``strike``/``gamma``/``open_interest``), with **gamma from :func:`bs_gamma`** on each
    contract's implied vol and days-to-expiry, restricted to expiries within ``max_dte`` days and
    to contracts with positive open interest. Returns ``(empty, nan)`` unless ``fetch=True`` — the
    offline core never imports yfinance. Use it to read today's real GEX, or run daily to
    accumulate a forward regime panel (the only free route to real history is to collect it).
    """
    if not fetch:
        return pd.DataFrame(), float("nan")
    import yfinance as yf

    tk = yf.Ticker(symbol)
    spot = float(tk.fast_info.get("lastPrice") or tk.history(period="1d")["Close"].iloc[-1])
    now = pd.Timestamp.now().normalize()
    frames = []
    for exp in tk.options:
        dte = (pd.Timestamp(exp) - now).days
        if dte < 0 or dte > max_dte:
            continue
        oc = tk.option_chain(exp)
        for side, df in (("call", oc.calls), ("put", oc.puts)):
            d = df[["strike", "openInterest", "impliedVolatility"]].copy()
            d["type"] = side
            d["gamma"] = bs_gamma(spot, d["strike"], dte / 365.0, d["impliedVolatility"])
            d = d.rename(columns={"openInterest": "open_interest"})
            frames.append(d[["type", "strike", "gamma", "open_interest"]])
    if not frames:
        return pd.DataFrame(), spot
    chain = pd.concat(frames, ignore_index=True)
    chain = chain[chain["open_interest"].fillna(0) > 0]
    return chain, spot


def synthetic_chain(spot: float = 100.0, call_oi_scale: float = 1.0, put_oi_scale: float = 1.0,
                    n_strikes: int = 41, seed: int = 0) -> pd.DataFrame:
    """A toy SPY-like option chain with a Gaussian gamma profile around ``spot``.

    Each strike gets a bell-shaped per-contract gamma (peaked at the money) and an open interest
    drawn around ``call_oi_scale`` / ``put_oi_scale``. Tuning those scales flips the *sign* of the
    net GEX deterministically (call-heavy => positive, put-heavy => negative), which is exactly
    what :func:`compute_gex` is unit-tested against. Mirrors the Alpha Vantage column names.
    """
    rng = np.random.default_rng(seed)
    strikes = np.linspace(spot * 0.85, spot * 1.15, n_strikes)
    gamma = np.exp(-0.5 * ((strikes - spot) / (spot * 0.05)) ** 2) * 0.05  # peaked ATM
    rows = []
    for k, g in zip(strikes, gamma):
        rows.append(("call", k, g, max(0.0, rng.normal(2000 * call_oi_scale, 300))))
        rows.append(("put", k, g, max(0.0, rng.normal(2000 * put_oi_scale, 300))))
    return pd.DataFrame(rows, columns=["type", "strike", "gamma", "open_interest"])


def _chain_cache_path(symbol: str, date: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"chain_{symbol}_{date}.parquet")


def fetch_chain(symbol: str, date: str, cache_dir: str = DEFAULT_CACHE,
                fetch: bool = False, api_key: str | None = None) -> pd.DataFrame:
    """Return (and cache) the Alpha Vantage ``HISTORICAL_OPTIONS`` chain for ``(symbol, date)``.

    **Cache-only by default**: a cached parquet is returned as-is; otherwise an empty frame is
    returned *unless* ``fetch=True``, in which case the endpoint is hit once and cached. The
    network import (``requests``) is lazy so the offline core never needs it. The API key is read
    from ``api_key`` or the ``ALPHAVANTAGE_API_KEY`` env var — never hard-coded. ``date`` is a
    trading day ``YYYY-MM-DD`` (history from 2008-01-01); the returned frame has the
    :data:`_AV_FIELDS` columns cast to numerics (``gamma``, ``open_interest`` floats; ``type`` str).
    """
    path = _chain_cache_path(symbol, date, cache_dir)
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()

    import requests  # lazy: offline core never imports it

    key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("set ALPHAVANTAGE_API_KEY (HISTORICAL_OPTIONS is a premium endpoint)")
    url = ("https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS"
           f"&symbol={symbol}&date={date}&apikey={key}")
    payload = requests.get(url, timeout=60).json()
    data = payload.get("data")
    if not data:                                            # rate-limited or empty trading day
        return pd.DataFrame()
    raw = pd.DataFrame(data)
    keep = [c for c in _AV_FIELDS if c in raw.columns]
    out = raw[keep].copy()
    for c in ("strike", "gamma", "open_interest"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(path)
    return out


def build_panel(chains: dict[str, pd.DataFrame], spots: pd.Series,
                bars: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Assemble the regime panel from per-date chains, daily SPY bars, and the VIX series.

    For each date ``d`` with both a chain and next-session bars: ``gex`` is :func:`compute_gex` on
    the chain at ``d``'s close (spot ``spots[d]``), attributed to the **next** trading session
    ``d+1`` (the GEX is known *before* that session opens); ``vix`` is ``d``'s close; ``rv`` and
    ``de`` are computed from session ``d+1``'s OHLC. Returns the canonical :data:`PANEL_COLUMNS`
    frame on the *outcome* (``d+1``) date index — one clean row per forecastable session.

    ``bars`` is a daily OHLC frame (columns Open/High/Low/Close) indexed by date; ``vix`` and
    ``spots`` are close series on the same calendar.
    """
    bars = bars.sort_index()
    dates = bars.index
    rows = []
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        key = pd.Timestamp(d).strftime("%Y-%m-%d")
        if key not in chains or chains[key].empty or d not in spots.index or d not in vix.index:
            continue
        g = compute_gex(chains[key], float(spots.loc[d]))
        if not np.isfinite(g["gex"]):
            continue
        o, h, l, c = (float(bars.loc[nxt, col]) for col in ("Open", "High", "Low", "Close"))
        rv = float(parkinson_vol(np.array([h]), np.array([l]))[0])
        de = float(directional_efficiency(np.array([o]), np.array([h]),
                                          np.array([l]), np.array([c]))[0])
        rows.append((nxt, float(spots.loc[d]), g["gex"], g["gex"] < 0,
                     float(vix.loc[d]), o, h, l, c, rv, de))

    out = pd.DataFrame(rows, columns=["date"] + PANEL_COLUMNS).set_index("date")
    out.index = pd.DatetimeIndex(out.index, name="date")
    out["neg_gamma"] = out["neg_gamma"].astype("boolean")
    return out
