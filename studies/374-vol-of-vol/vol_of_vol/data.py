"""Data layer for Study 374 — Vol-of-Vol (VVIX vs VIX as an equity-risk timer).

Two sources, both offline-friendly:

* **Real tape.** Daily closes for ``^VVIX`` (the volatility of the VIX), ``^VIX`` (the
  volatility of the S&P 500), and ``SPY`` (total-return-ish, auto-adjusted), via yfinance
  (no key), cached under ``_cache/vol_tape.csv``. VVIX is the CBOE index that applies the
  VIX construction to VIX *options* — the "vol of vol." It begins in 2007.

* **Synthetic.** A deterministic, fixed-seed generator that produces VVIX/VIX/return paths
  with a **planted edge knob**: ``edge`` controls how much a high-VVIX state actually
  predicts the next forward-return drawdown. With ``edge = 0`` the positive control must NOT
  manufacture significance (high VVIX is uninformative about returns beyond what VIX already
  says); a large planted edge MUST light up. This is the machinery proof — never market
  evidence.

Pure numpy + pandas + stdlib for the offline path. ``fetch_tape`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "vol_tape.csv")

TICKERS = ["^VVIX", "^VIX", "SPY"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "2007-01-01", end: str | None = None,
               path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download ^VVIX, ^VIX, SPY via yfinance and cache a wide close CSV.

    Network-only; used once to build ``_cache/vol_tape.csv``. Never imported by the offline
    notebook cells. Columns: ``vvix``, ``vix``, ``spy`` (auto-adjusted SPY close).
    """
    import yfinance as yf

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    out = pd.DataFrame({
        "vvix": raw["^VVIX"],
        "vix": raw["^VIX"],
        "spy": raw["SPY"],
    }).dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached frame (index = date, columns = ``vvix``, ``vix``, ``spy``).

    The CSV cached by ``fetch_tape`` already has lower-case columns; if a raw wide CBOE
    file is dropped in instead, the loader normalises the three columns it needs.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want, aliases in (("vvix", ("vvix", "^vvix")),
                          ("vix", ("vix", "^vix")),
                          ("spy", ("spy", "^gspc", "close"))):
        for a in aliases:
            if a in cols:
                rename[cols[a]] = want
                break
    df = df.rename(columns=rename)
    return df[["vvix", "vix", "spy"]].dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_tape(n_days: int = 4_800, edge: float = 0.0, seed: int = 374,
                   mu_daily: float = 0.0003, base_vix: float = 18.0,
                   base_vvix: float = 90.0) -> pd.DataFrame:
    """Deterministic VVIX/VIX/SPY paths with a PLANTED vol-of-vol edge.

    Construction:
      * a latent fear factor ``f`` (AR(1)) drives VIX; VVIX is VIX-driven **plus an
        independent vol-of-vol component** ``g`` (also AR(1)) so VVIX carries information
        VIX does not;
      * daily SPY log-returns have drift ``mu_daily`` and a volatility that scales with
        VIX (so high VIX already precedes turbulence — the confound VVIX must beat);
      * if ``edge != 0`` an **extra** negative forward drift proportional to ``edge`` is
        injected on days when the *independent* vol-of-vol component ``g`` is high — i.e.
        the planted edge is **incremental to VIX**: it lives only in the part of VVIX that
        VIX cannot see. ``edge = 0`` ⇒ high VVIX says nothing beyond VIX.

    With ``edge = 0`` the inference must NOT find that VVIX beats VIX (no false positive);
    a large ``edge`` (e.g. 0.04) must light up the incremental test. The control proves the
    engine isolates *incremental* vol-of-vol signal, not VIX leaking through.
    """
    rng = np.random.default_rng(seed)

    # latent fear factor f (drives both VIX and SPY vol) and an independent vol-of-vol
    # component g (drives only the residual of VVIX) — both mean-reverting.
    f = np.empty(n_days); g = np.empty(n_days)
    f[0] = 0.0; g[0] = 0.0
    for t in range(1, n_days):
        f[t] = 0.97 * f[t - 1] + rng.normal(0, 1.0)
        g[t] = 0.95 * g[t - 1] + rng.normal(0, 1.0)

    fz = (f - f.mean()) / f.std()
    gz = (g - g.mean()) / g.std()

    vix = np.clip(base_vix + 7.0 * fz, 9.0, 80.0)
    # VVIX = VIX level component + an INDEPENDENT vol-of-vol component (gz)
    vvix = np.clip(base_vvix + 5.0 * fz + 14.0 * gz, 60.0, 200.0)

    # daily vol scales with VIX (annualised VIX -> daily sigma)
    sig_daily = vix / 100.0 / np.sqrt(252.0)
    ret = rng.normal(mu_daily, 1.0, size=n_days) * sig_daily

    # planted INCREMENTAL edge: extra negative forward drift when gz (the part of VVIX
    # orthogonal to VIX) is high. Spread over the next ~21 trading days.
    if edge != 0.0:
        h = 21
        hot = gz > 1.0
        for t in np.where(hot)[0]:
            lo, hi = t + 1, min(t + 1 + h, n_days)
            ret[lo:hi] -= edge / h

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    return pd.DataFrame({"vvix": vvix, "vix": vix, "spy": price}, index=idx)
