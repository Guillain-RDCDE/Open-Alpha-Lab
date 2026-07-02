"""Data layer for Study 558 (Failures-To-Deliver) — the settlement-fail "squeeze predictor".

**Failures-to-deliver (FTD)** are shares that a seller was contractually obliged to deliver at
settlement but did not. The SEC publishes a fails-to-deliver file (aggregate open fails per CUSIP,
twice a month) precisely because persistent fails can signal naked shorting or settlement stress.
Meme-stock-era microstructure folklore turned this into a *trading signal*: a **spike in FTD is a
short-squeeze predictor** — if delivery is failing, the theory goes, shorts are trapped and a
squeeze is imminent, so forward returns after an FTD spike should be large and positive.

This study tests that claim. But note the data reality up front:

- The SEC FTD file is a **bulk, semi-monthly, T+~5-settlement-day-lagged** flat file keyed by
  CUSIP, with *no* price/volume alongside it. It is not on the retail no-key stack (``yfinance``
  has no FTD endpoint), and stitching CUSIP→ticker→adjusted-price across the meme-stock cohort
  with survivorship-clean delisting is a data-engineering project in its own right. So the free,
  reproducible tape this desk can actually build **does not exist** for FTD. This study is
  therefore **synthetic-only** — and a synthetic-only study can never earn a ``REAL`` signal (that
  needs a robust ``t >= 2`` on a real tape). The data-availability limitation is stated openly on
  the SIGNAL axis; the honest ceiling here is ``WEAK``/``NONE``.

What we CAN do rigorously is build a faithful synthetic FTD *event panel* with a single planted
knob, prove the event-study engine detects a planted squeeze and stays flat at the null, and show
what the folklore would require of the real data to clear the bar.

The generator:

- ``synthetic_panel`` — a deterministic, offline daily panel of ``n_names`` tickers over
  ``n_days`` days. Each name has a fails-to-deliver series (a persistent, right-skew, occasionally
  spiking process) and a daily return series. A single knob, ``squeeze_alpha``, plants the effect:
  on the days *following* an FTD spike, the name earns an extra ``squeeze_alpha`` of forward drift.
  ``squeeze_alpha > 0`` reproduces the folklore (spikes foreshadow a pop); ``= 0`` is the null
  (FTD spikes carry no forward information). The study's positive control and null in one bottle.
  Tests never touch the network.
- ``fetch_panel`` — a documented stub. There is no free FTD tape to fetch; it returns empty and
  names the limitation, so the reproducible core never needs (and never has) a network path.

No look-ahead: a spike is identified from the FTD level *at* day t (public, if lagged, in the real
world); the position is entered at the day-t close and the *forward* return window (t+1 .. t+h) is
what we measure. The signal never sees a future return. The one-bar entry lag (enter at t's close,
measure t+1 onward) is the single documented execution convention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 558
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic FTD panel."""

    squeeze_alpha: float  # extra forward drift on post-spike days (>0 = the folklore)

    @property
    def has_effect(self) -> bool:
        return self.squeeze_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 60,
    n_days: int = 756,
    squeeze_alpha: float = 0.0,
    spike_z: float = 2.5,
    base_vol: float = 0.028,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily FTD + return panel with a tunable squeeze effect.

    For each of ``n_names`` tickers we generate:

    - a **fails-to-deliver** series ``ftd``: a persistent AR(1) log-level process (fails cluster —
      a name that fails today tends to fail tomorrow) with an occasional multiplicative jump, so
      the series is right-skewed and *spikes* now and then, like the real SEC file.
    - a daily **return** series ``ret``: a modest positive drift plus idiosyncratic noise. When
      ``squeeze_alpha != 0``, every day *following* an FTD **spike** (an FTD z-score, computed from
      that name's own trailing history, exceeding ``spike_z``) receives an extra
      ``squeeze_alpha`` of drift spread over the next few days — the planted "post-spike pop".

    A *spike* is defined per name from its trailing FTD z-score, using only data up to and
    including day t (no look-ahead). The planted pop lands on days t+1..t+h, so the signal (the
    spike at t) never peeks at the return it predicts.

    Returns a long-format frame with columns ``[name, day, ftd, ret]`` and the ``WorldTruth``.
    """
    rng = np.random.default_rng(seed)
    horizon = 5  # days over which a planted pop is spread
    phi = 0.90

    # --- FTD level: AR(1) in logs with occasional multiplicative jumps, vectorised over names ---
    # Shape (n_days, n_names); loop only over days (AR(1) is sequential in time).
    log_ftd = np.empty((n_days, n_names))
    log_ftd[0] = rng.normal(9.0, 1.0, size=n_names)
    innov = rng.normal(0.0, 0.25, size=(n_days, n_names))
    jump_hit = rng.random((n_days, n_names)) < 0.03  # ~3% of name-days get a fail jump
    jump_mag = rng.exponential(1.4, size=(n_days, n_names))
    jumps = np.where(jump_hit, jump_mag, 0.0)
    for t in range(1, n_days):
        log_ftd[t] = 8.0 * (1 - phi) + phi * log_ftd[t - 1] + innov[t] + jumps[t]
    ftd = np.exp(log_ftd)  # (n_days, n_names)

    # --- trailing z-score per name (own past only) -> spike flags ---
    spike = _trailing_z_mat(ftd, lookback=60) >= spike_z  # (n_days, n_names) bool

    # --- returns: drift + noise, plus a planted post-spike pop if alpha != 0 ---
    ret = rng.normal(0.0004, base_vol, size=(n_days, n_names))
    if squeeze_alpha != 0.0:
        add = np.zeros((n_days, n_names))
        # a spike at day d contributes squeeze_alpha/horizon to days d+1 .. d+horizon
        for lag in range(1, horizon + 1):
            add[lag:] += spike[:-lag] * (squeeze_alpha / horizon)
        ret = ret + add

    names = np.array([f"N{j:03d}" for j in range(n_names)])
    panel = pd.DataFrame(
        {
            "name": np.repeat(names, n_days),
            "day": np.tile(np.arange(n_days), n_names),
            "ftd": ftd.T.reshape(-1),
            "ret": ret.T.reshape(-1),
        }
    )
    return panel, WorldTruth(squeeze_alpha=squeeze_alpha)


def _trailing_z_mat(x: np.ndarray, lookback: int = 60) -> np.ndarray:
    """Column-wise trailing z-score of a (days, names) matrix (past-only, no look-ahead)."""
    df = pd.DataFrame(x)
    past = df.shift(1)
    m = past.rolling(window=lookback, min_periods=20).mean()
    sd = past.rolling(window=lookback, min_periods=20).std(ddof=1)
    z = (df - m) / sd
    z = z.where(sd > 0)
    return z.to_numpy()


def _trailing_z(x: np.ndarray, lookback: int = 60) -> np.ndarray:
    """Trailing z-score of ``x`` using a rolling *past-only* window (data up to t-1).

    z[t] = (x[t] - mean(x[t-lookback..t-1])) / std(x[t-lookback..t-1]). Uses only the past, so it
    is causal (no look-ahead). Points with fewer than 20 prior observations are NaN, treated as
    no-spike. Vectorised with a shifted rolling window (min 20, max ``lookback`` obs).
    """
    s = pd.Series(np.asarray(x, dtype=float))
    past = s.shift(1)  # window ends at t-1 (strictly past) -> no look-ahead
    m = past.rolling(window=lookback, min_periods=20).mean()
    sd = past.rolling(window=lookback, min_periods=20).std(ddof=1)
    z = (s - m) / sd
    z = z.where(sd > 0)
    return z.to_numpy()


def spike_flags(ftd: np.ndarray, spike_z: float = 2.5, lookback: int = 60) -> np.ndarray:
    """Boolean per-day FTD-spike flags from the trailing z-score (causal, no look-ahead)."""
    z = _trailing_z(np.asarray(ftd, dtype=float), lookback=lookback)
    return np.nan_to_num(z, nan=-np.inf) >= spike_z


# ---------------------------------------------------------------------------
# "Real" tape — a documented stub: there is no free FTD tape to fetch
# ---------------------------------------------------------------------------
def fetch_panel(fetch: bool = False) -> pd.DataFrame:
    """No-op real-data loader — the free FTD tape does not exist on the retail stack.

    The SEC fails-to-deliver file is a bulk semi-monthly CUSIP-keyed flat file with no prices
    attached and no ``yfinance`` endpoint; a survivorship-clean CUSIP→ticker→adjusted-price join
    across the meme-stock cohort is out of scope for a no-key reproducible desk. This function
    therefore always returns an empty frame (``fetch`` is accepted for signature symmetry with
    other studies but there is nothing to fetch). The limitation is stated on the SIGNAL axis:
    with no real tape, this study is synthetic-only and cannot earn a ``REAL`` stamp.
    """
    return pd.DataFrame()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
