"""Data layer for Study 596 (Bond Tent / Rising Equity Glidepath).

Two tapes, one shape (monthly *real* total returns for equities and bonds):

- ``load_shiller`` / ``real_returns`` — the real 150-year Shiller S&P 500
  dataset, **cache-first**: the study's own ``_cache/shiller_sp500.parquet``
  first, then the repo-root staged copy, then a one-time refetch from the
  GitHub raw mirror of the Shiller ``ie_data`` extract. Bond returns are
  approximated from the Shiller 10-year yield (carry minus duration-weighted
  yield change) and **deflated by CPI** so both series are real.
- ``synthetic_world`` — a *deterministic, offline* annual two-asset generator
  with two tunable knobs that mirror the study's decomposition:

  * ``reversion`` — long-horizon mean reversion of the equity price level
    toward trend (bad decades get repaired). ``reversion = 0`` is the i.i.d.
    null; ``reversion > 0`` plants exactly the regime structure a rising
    glidepath is supposed to exploit.
  * the *withdrawal rate* knob lives in ``strategy.simulate`` — with **no
    withdrawals** and i.i.d. returns, a weight schedule and its mirror image
    produce the *same* terminal-wealth distribution (product of independent
    factors is exchangeable), so "rising vs declining, wr = 0, reversion = 0"
    is the exact null where the engine must find nothing.

  Synthetic spans stay well under 250 years (pandas ns-Timestamp CI trap).

No look-ahead lives here: glidepath weights are a deterministic function of
years-since-retirement, fixed at the retirement date (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(STUDY_ROOT, "..", ".."))
STUDY_CACHE = os.path.join(STUDY_ROOT, "_cache")
SHILLER_PATH = os.path.join(STUDY_CACHE, "shiller_sp500.parquet")

# Repo-root staged copies (older studies keep the same parquet here)
_FALLBACKS = [
    os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet"),
    os.path.join(REPO_ROOT, "_cache", "_cache", "shiller_sp500.parquet"),
]

# GitHub raw mirror of the Shiller ie_data extract (datasets/s-and-p-500)
SHILLER_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
)

MOD_DURATION = 7.0  # approximate modified duration of a 10-year Treasury


# ---------------------------------------------------------------------------
# Real tape — Shiller S&P 500 monthly, cache-first
# ---------------------------------------------------------------------------

def load_shiller(cache_path: str = SHILLER_PATH) -> pd.DataFrame:
    """Raw Shiller monthly frame, cache-first (study cache -> repo cache -> web)."""
    if os.path.exists(cache_path):
        raw = pd.read_parquet(cache_path)
    else:
        raw = None
        for fb in _FALLBACKS:
            if os.path.exists(fb):
                raw = pd.read_parquet(fb)
                break
        if raw is None:  # pragma: no cover — one-time refetch
            import urllib.request

            req = urllib.request.Request(
                SHILLER_URL,
                headers={"User-Agent": "OpenAlphaLab research desk "
                                        "guillain@poulpe.us"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = pd.read_csv(io.BytesIO(r.read()))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw.to_parquet(cache_path)
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw.sort_values("Date").set_index("Date")


def real_returns(
    start: str = "1871-02-01",
    end: str | None = None,
    cache_path: str = SHILLER_PATH,
) -> pd.DataFrame:
    """Monthly **real** total returns ``['EQ', 'BD']`` from the Shiller tape.

    Equity (``EQ``): real price return (Shiller ``Real Price``) plus the
    dividend yield implied at month t-1 (``Dividend / SP500 / 12``, shifted
    one month — no look-ahead on the dividend).

    Bond (``BD``): the standard first-order 10-year approximation
    ``y_{t-1}/12 - D * dy_t`` (carry minus duration-weighted yield change,
    D = 7), then **deflated by realised CPI inflation** so the series is real
    — unlike sibling study 172, which left the bond leg nominal.

    Rows require positive SP500, Real Price, Long Interest Rate, Dividend and
    CPI (the parquet tail carries zero-filled placeholder months). The full
    valid window is 1871-02 .. 2023-06.
    """
    raw = load_shiller(cache_path)
    ok = (
        (raw["SP500"] > 0)
        & (raw["Real Price"] > 0)
        & (raw["Long Interest Rate"] > 0)
        & (raw["Dividend"] > 0)
        & (raw["Consumer Price Index"] > 0)
    )
    raw = raw[ok]

    div_yield_monthly = raw["Dividend"] / raw["SP500"] / 12.0
    eq = raw["Real Price"].pct_change() + div_yield_monthly.shift(1)

    y = raw["Long Interest Rate"] / 100.0
    bd_nominal = y.shift(1) / 12.0 - MOD_DURATION * y.diff()
    infl = raw["Consumer Price Index"].pct_change()
    bd = (1.0 + bd_nominal) / (1.0 + infl) - 1.0

    df = pd.DataFrame({"EQ": eq, "BD": bd}).dropna()
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    df.index.name = "Date"
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a return frame, for the as-of stamp."""
    h = hashlib.sha1()
    for c in sorted(df.columns):
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(df[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline generator (annual returns)
# ---------------------------------------------------------------------------

def synthetic_world(
    n_years: int = 200,
    eq_mu: float = 0.065,        # annual real equity return (log-mean approx)
    eq_vol: float = 0.17,
    bd_mu: float = 0.020,        # annual real bond return
    bd_vol: float = 0.07,
    corr: float = 0.10,
    reversion: float = 0.0,      # kappa: pull of the equity level toward trend
    seed: int = 596,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible **annual** two-asset real-return world.

    Equity log return: ``x_t = mu - reversion * V_{t-1} + sig * e_t`` with
    ``V_t = V_{t-1} + (x_t - mu)`` — the cumulative deviation of the log price
    level from trend. ``reversion = 0`` is a random walk (i.i.d. returns);
    ``reversion ~ 0.05-0.15`` plants slow, decade-scale mean reversion: bad
    stretches are systematically repaired later, exactly the structure a
    rising-equity glidepath is supposed to harvest.

    Returns ``(frame, truth)``; ``frame`` has columns ``['EQ', 'BD']`` of
    annual simple real returns indexed by a decorative ``period_range`` of
    years (span <= 250 years by construction).
    """
    if n_years > 250:
        raise ValueError("keep synthetic spans <= 250 years (ns-Timestamp trap)")
    rng = np.random.default_rng(seed)
    mu_e = np.log1p(eq_mu) - 0.5 * eq_vol**2
    mu_b = np.log1p(bd_mu) - 0.5 * bd_vol**2

    z = rng.standard_normal((n_years, 2))
    z_b = corr * z[:, 0] + np.sqrt(1.0 - corr**2) * z[:, 1]

    x_e = np.empty(n_years)
    v = 0.0
    for t in range(n_years):
        x_e[t] = mu_e - reversion * v + eq_vol * z[t, 0]
        v += x_e[t] - mu_e
    x_b = mu_b + bd_vol * z_b

    idx = pd.period_range("1900", periods=n_years, freq="Y")
    frame = pd.DataFrame(
        {"EQ": np.expm1(x_e), "BD": np.expm1(x_b)}, index=idx
    )
    truth = dict(
        n_years=n_years, eq_mu=eq_mu, eq_vol=eq_vol, bd_mu=bd_mu,
        bd_vol=bd_vol, corr=corr, reversion=reversion, seed=seed,
    )
    return frame, truth
