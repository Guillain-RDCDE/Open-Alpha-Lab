"""Data layer for Study 372 — Max-Pain Pinning (option-chain snapshot + synthetic episodes).

Two sources, both offline-friendly:

* **Real tape (a snapshot).** A single-date cross-section of liquid US underlyings. For each
  we pull the nearest liquid option expiry via yfinance, compute the **max-pain strike** from
  call/put open interest, and store the spot, the max-pain strike, and the open interest.
  Cached under ``_cache/mp_snapshot.csv`` (one row per underlying). True *expiry-day* outcomes
  are not retained by yfinance, so this is a **snapshot**, not a pinning panel — named as such
  everywhere. ``fetch_snapshot`` (network) is only used once to build the cache and is never
  imported by the notebooks' offline cells.

* **Synthetic.** A deterministic, fixed-seed generator producing many (underlying, expiry)
  **episodes**: a strike grid, an open-interest profile that defines a max-pain strike, a
  spot a few days before expiry, and an expiry close. A tunable ``pin`` knob adds a
  mean-reverting drag toward max-pain over the final days. It is the positive control: with
  ``pin = 0`` the expiry close is a pure random walk (NO pinning, and the inference must NOT
  manufacture significance); with a large ``pin`` the close must land demonstrably closer to
  max-pain than to a random strike.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "mp_snapshot.csv")

# A transparent, fixed basket of liquid US underlyings (indices + large-caps) used for the
# max-pain snapshot. Chosen for deep, dense option chains so the max-pain strike is meaningful.
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "TSLA", "AMD", "JPM", "XOM", "WMT", "KO", "PG", "HD", "BAC", "PFE",
]


# --------------------------------------------------------------------------- #
# Max-pain from an option chain (shared by the real fetch and the engine)
# --------------------------------------------------------------------------- #
def max_pain_strike(strikes: np.ndarray, call_oi: np.ndarray,
                    put_oi: np.ndarray) -> float:
    """The max-pain strike: the settlement price that minimises total option payout.

    For a candidate settlement ``s``, calls with strike ``K < s`` pay ``s - K`` and puts with
    strike ``K > s`` pay ``K - s`` (per unit of open interest). Max-pain is the strike that
    minimises the sum of those payouts (= maximises the value expiring worthless). Inputs are
    aligned arrays over the same strike grid.
    """
    strikes = np.asarray(strikes, dtype=float)
    call_oi = np.asarray(call_oi, dtype=float)
    put_oi = np.asarray(put_oi, dtype=float)
    payout = np.empty(len(strikes))
    for i, s in enumerate(strikes):
        call_pay = np.maximum(s - strikes, 0.0) @ call_oi
        put_pay = np.maximum(strikes - s, 0.0) @ put_oi
        payout[i] = call_pay + put_pay
    return float(strikes[int(np.argmin(payout))])


# --------------------------------------------------------------------------- #
# Real tape (snapshot)
# --------------------------------------------------------------------------- #
def fetch_snapshot(tickers: list[str] | None = None, min_oi: int = 2_000,
                   asof: str = "2026-06-22", path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download one option-chain snapshot per underlying and cache the max-pain table.

    Network-only; used once to build ``_cache/mp_snapshot.csv``. For each ticker we take the
    nearest expiry whose total open interest exceeds ``min_oi`` and whose strike grid brackets
    the spot (a guard against split/raw price mismatches), compute the max-pain strike, and
    record ``ticker, exp, asof, spot, maxpain, oi, nstrk``. Never imported by offline cells.
    """
    import time

    import yfinance as yf

    tickers = tickers or UNIVERSE
    rows = []
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            exps = t.options
            if not exps:
                continue
            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if len(hist) else float("nan")
            for exp in exps[:6]:
                oc = t.option_chain(exp)
                calls, puts = oc.calls, oc.puts
                tot = int(calls["openInterest"].sum() + puts["openInterest"].sum())
                if tot < min_oi:
                    continue
                strikes = np.union1d(calls["strike"].values, puts["strike"].values)
                co = calls.set_index("strike")["openInterest"].reindex(strikes).fillna(0).values
                po = puts.set_index("strike")["openInterest"].reindex(strikes).fillna(0).values
                if not (strikes.min() <= spot <= strikes.max()):
                    continue  # data-quality guard
                mp = max_pain_strike(strikes, co, po)
                rows.append(dict(ticker=tk, exp=exp, asof=asof, spot=round(spot, 2),
                                 maxpain=mp, oi=tot, nstrk=len(strikes)))
                break
            time.sleep(0.25)
        except Exception:  # noqa: BLE001 — best-effort fetch; skip a flaky name
            continue
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached max-pain snapshot and add the spot-vs-max-pain gap (signed %)."""
    df = pd.read_csv(path)
    df["gap_pct"] = (df["spot"] - df["maxpain"]) / df["spot"] * 100.0
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic pinning episodes
# --------------------------------------------------------------------------- #
def synthetic_episodes(n_episodes: int = 400, pin: float = 0.0, seed: int = 372,
                       n_strikes: int = 21, sig_daily: float = 0.012,
                       days_to_expiry: int = 5) -> pd.DataFrame:
    """Deterministic (underlying, expiry) episodes with a tunable pinning ``pin`` knob.

    Each episode:
      * a strike grid of ``n_strikes`` evenly-spaced strikes around 100 (spacing = 1 unit),
      * an open-interest profile peaked at a random interior strike → a known **max-pain**
        strike (computed by :func:`max_pain_strike`, so the engine is self-consistent),
      * a spot ``days_to_expiry`` sessions before expiry, drawn near the grid centre,
      * an expiry **close** evolved as a random walk with daily vol ``sig_daily`` …
      * … plus, when ``pin > 0``, a mean-reverting **drag toward max-pain** each remaining
        day: ``drift_t = pin * (maxpain - price_t) / price_t``. ``pin`` is the daily fraction
        of the gap closed; ``pin = 0`` ⇒ a pure random walk (NO pinning).

    With ``pin = 0`` the close-to-max-pain distance must be statistically indistinguishable
    from the close-to-random-strike distance (no manufactured significance). A large ``pin``
    must drive the close onto max-pain. Returns one row per episode with the columns
    :func:`load_real` produces plus the random-walk-free ``maxpain`` and realised ``close``.
    """
    rng = np.random.default_rng(seed)
    spacing = 1.0
    grid = (np.arange(n_strikes) - n_strikes // 2) * spacing + 100.0
    rows = []
    for _ in range(n_episodes):
        # open-interest hump → a max-pain strike at an interior grid point
        peak = rng.integers(n_strikes // 4, 3 * n_strikes // 4 + 1)
        width = rng.uniform(1.5, 4.0)
        base_oi = np.exp(-0.5 * ((np.arange(n_strikes) - peak) / width) ** 2)
        call_oi = base_oi * rng.uniform(0.6, 1.4, n_strikes) * 1000.0
        put_oi = base_oi * rng.uniform(0.6, 1.4, n_strikes) * 1000.0
        mp = max_pain_strike(grid, call_oi, put_oi)

        # spot a few days out, near the grid centre (not exactly on max-pain)
        spot = 100.0 + rng.normal(0.0, 2.0)
        price = spot
        for _d in range(days_to_expiry):
            shock = rng.normal(0.0, sig_daily)
            drag = pin * (mp - price) / price if pin else 0.0
            price *= np.exp(shock + drag)
        close = price
        rows.append(dict(ticker="SYN", exp="syn", asof="syn",
                         spot=round(spot, 4), maxpain=mp, close=round(close, 4),
                         oi=int(call_oi.sum() + put_oi.sum()), nstrk=n_strikes,
                         lo=float(grid.min()), hi=float(grid.max()), spacing=spacing))
    out = pd.DataFrame(rows)
    out["gap_pct"] = (out["spot"] - out["maxpain"]) / out["spot"] * 100.0
    return out
