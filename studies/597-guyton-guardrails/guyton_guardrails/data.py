"""Data layer for Study 597 (Guyton-Klinger Guardrails).

Two tapes, one shape (monthly *nominal* total returns for equities and bonds
plus monthly CPI inflation — the guardrail rules are defined on nominal
withdrawal amounts and nominal portfolio values, so the simulation runs in
nominal space and every income/wealth outcome is deflated back to real):

- ``load_shiller`` / ``nominal_returns`` — the real 150-year Shiller S&P 500
  dataset, **cache-first**: the study's own ``_cache/shiller_sp500.parquet``
  first, then repo-root / sibling staged copies, then a one-time refetch from
  the GitHub raw mirror of the Shiller ``ie_data`` extract. Bond returns are
  the standard 10-year first-order approximation (carry minus
  duration-weighted yield change), left **nominal**; CPI inflation is carried
  alongside so the simulator can deflate.
- ``synthetic_world`` — a *deterministic, offline* annual two-asset generator
  with **constant inflation** and two tunable knobs that mirror the study's
  mechanics:

  * ``eq_vol`` / ``eq_mu`` — ruin risk. Crank volatility up (or the premium
    down) and fixed-rate retirements start failing — exactly the effect the
    guardrails are supposed to rescue. The **null** is a calm world
    (low vol, adequate premium, moderate withdrawal) where the fixed rule
    already succeeds in *every* cohort: there is nothing to rescue, so the
    ruin detector Δsuccess(guardrails − fixed, same rate) must read exactly 0.
  * the *withdrawal rate* knob lives in ``strategy.simulate``.

  Synthetic spans stay well under 250 years (pandas ns-Timestamp CI trap);
  the index is a decorative ``period_range`` of years.

No look-ahead lives here: every rule input the simulator uses (last year's
inflation, last year's portfolio return, current portfolio value) is known at
the start-of-year withdrawal date (see ``strategy.py``).
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

# Repo-root / sibling staged copies (other studies keep the same parquet)
_FALLBACKS = [
    os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet"),
    os.path.join(REPO_ROOT, "_cache", "_cache", "shiller_sp500.parquet"),
    os.path.join(REPO_ROOT, "studies", "596-bond-tent-glidepath",
                 "_cache", "shiller_sp500.parquet"),
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


def nominal_returns(
    start: str = "1871-02-01",
    end: str | None = None,
    cache_path: str = SHILLER_PATH,
) -> pd.DataFrame:
    """Monthly **nominal** total returns + CPI inflation ``['EQ', 'BD', 'INFL']``.

    Equity (``EQ``): nominal price return (Shiller ``SP500``) plus the
    dividend yield implied at month t-1 (``Dividend / SP500 / 12``, shifted
    one month — no look-ahead on the dividend).

    Bond (``BD``): the standard first-order 10-year approximation
    ``y_{t-1}/12 - D * dy_t`` (carry minus duration-weighted yield change,
    D = 7), left **nominal** — the guardrail rules live in nominal space.

    Inflation (``INFL``): realised monthly CPI change, used by the simulator
    both for the rules' inflation adjustment and to deflate income/wealth
    back to real. Simulating in nominal space and deflating is algebraically
    identical to sibling study 596's real-space simulation for the fixed
    (Bengen) rule — a stated cross-check, not an accident.

    Rows require positive SP500, Long Interest Rate, Dividend and CPI (the
    parquet tail carries zero-filled placeholder months). The full valid
    window is 1871-02 .. 2023-06.
    """
    raw = load_shiller(cache_path)
    ok = (
        (raw["SP500"] > 0)
        & (raw["Long Interest Rate"] > 0)
        & (raw["Dividend"] > 0)
        & (raw["Consumer Price Index"] > 0)
    )
    raw = raw[ok]

    div_yield_monthly = raw["Dividend"] / raw["SP500"] / 12.0
    eq = raw["SP500"].pct_change() + div_yield_monthly.shift(1)

    y = raw["Long Interest Rate"] / 100.0
    bd = y.shift(1) / 12.0 - MOD_DURATION * y.diff()
    infl = raw["Consumer Price Index"].pct_change()

    df = pd.DataFrame({"EQ": eq, "BD": bd, "INFL": infl}).dropna()
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
    eq_mu: float = 0.065,        # annual REAL equity return
    eq_vol: float = 0.17,
    bd_mu: float = 0.020,        # annual REAL bond return
    bd_vol: float = 0.07,
    corr: float = 0.10,
    infl: float = 0.030,         # constant annual inflation
    seed: int = 597,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible **annual** two-asset world in nominal terms.

    Real log returns are i.i.d. normal (lognormal levels, means matched to
    ``eq_mu``/``bd_mu`` simple real returns); a constant inflation rate turns
    them nominal: ``(1 + real) * (1 + infl) - 1``. Columns ``['EQ', 'BD',
    'INFL']`` of annual simple nominal returns + inflation, indexed by a
    decorative ``period_range`` of years (span <= 250 by construction).

    The tunable planted effect is **ruin risk**: raise ``eq_vol`` (or cut
    ``eq_mu``) until fixed-rate retirements start failing — the guardrails'
    rescue is what the detector must then light up on. The null is the calm
    world where the fixed rule already succeeds everywhere.
    """
    if n_years > 250:
        raise ValueError("keep synthetic spans <= 250 years (ns-Timestamp trap)")
    rng = np.random.default_rng(seed)
    mu_e = np.log1p(eq_mu) - 0.5 * eq_vol**2
    mu_b = np.log1p(bd_mu) - 0.5 * bd_vol**2

    z = rng.standard_normal((n_years, 2))
    z_b = corr * z[:, 0] + np.sqrt(1.0 - corr**2) * z[:, 1]

    real_eq = np.expm1(mu_e + eq_vol * z[:, 0])
    real_bd = np.expm1(mu_b + bd_vol * z_b)

    idx = pd.period_range("1900", periods=n_years, freq="Y")
    frame = pd.DataFrame(
        {
            "EQ": (1.0 + real_eq) * (1.0 + infl) - 1.0,
            "BD": (1.0 + real_bd) * (1.0 + infl) - 1.0,
            "INFL": np.full(n_years, infl),
        },
        index=idx,
    )
    truth = dict(
        n_years=n_years, eq_mu=eq_mu, eq_vol=eq_vol, bd_mu=bd_mu,
        bd_vol=bd_vol, corr=corr, infl=infl, seed=seed,
    )
    return frame, truth
