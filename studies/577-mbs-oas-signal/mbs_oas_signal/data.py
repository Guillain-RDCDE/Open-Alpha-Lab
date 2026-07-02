"""Data layer for Study 577 (MBS-OAS-Signal) — the mortgage-spread risk-off lead.

The claim needs two time series at a shared (weekly) frequency:

- **MBS OAS** — the option-adjusted spread on agency mortgage-backed securities (bps over
  Treasuries, after removing the prepayment-option value). A mean-reverting level that spikes in
  stress (2008, 2011, 2020, 2022).
- **A risk asset return** — the forward return of an equity/credit book (the thing OAS is supposed
  to lead). We carry a broad equity total-return proxy; the credit leg is the same machinery with a
  higher OAS loading.

Why synthetic-only. A clean, long, **free** daily/weekly agency-MBS OAS series does not exist for a
no-key retail stack: the canonical OAS series are vendor index products (ICE BofA US Mortgage-Backed
Securities OAS, Bloomberg MBS OAS) behind licences, and FRED's mortgage series (e.g. MORTGAGE30US,
the 30y primary rate) are *yields*, not option-adjusted spreads. So this study plants and measures
the effect in a deterministic synthetic world and states the wall openly on the SIGNAL axis — a
synthetic-only study can never be `REAL`.

Two functions, one schema:

- ``synthetic_series`` — a deterministic, offline generator. A single knob, ``lead_beta``, plants
  the risk-off *lead*: a widening in OAS this week lowers next week's risk-asset return
  (``lead_beta < 0`` reproduces the folklore; ``= 0`` is the null). The study's positive control and
  null in one bottle. Nothing here touches the network.
- ``fetch_series`` — the real-tape hook. There is **no free OAS tape**, so this is a documented
  cache-first stub: ``fetch=False`` returns an empty frame (offline default); ``fetch=True`` raises
  a clear ``NotImplementedError`` naming the licensed sources. The study therefore runs, and can
  only be graded, on the synthetic world.

No look-ahead. The signal at week *t* (the OAS level / its recent change) uses only information
known by the close of week *t*; the return it is tested against is the **forward** return over week
*t+1* (one explicit execution lag). The generator itself never leaks future data into the signal.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

WEEKS_PER_YEAR = 52
SEED = 577


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic series."""

    lead_beta: float  # forward-return loading per unit of standardised OAS change (neg = the lead)

    @property
    def has_lead(self) -> bool:
        return self.lead_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_weeks: int = 780,          # ~15 years of weekly data
    lead_beta: float = -0.9,     # forward-return %pt per +1 sd weekly OAS change (neg = the lead)
    oas_kappa: float = 0.06,     # OAS mean-reversion speed (weekly)
    oas_mean: float = 55.0,      # long-run OAS level (bps)
    oas_shock: float = 6.0,      # weekly OAS innovation sd (bps)
    ret_vol: float = 2.2,        # idiosyncratic weekly risk-asset return sd (%)
    drift: float = 0.15,         # weekly risk-asset drift (%)
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible weekly (OAS, forward risk-asset return) panel with a tunable *lead*.

    The OAS is a mean-reverting (Ornstein-Uhlenbeck-style) level with fat, clustered stress spikes
    (a few Poisson jumps standing in for 2008/2020/2022). From its **standardised weekly change** we
    plant the lead into *next* week's return::

        oas[t]      = oas[t-1] + kappa*(mean - oas[t-1]) + shock*eps[t] + jumps[t]
        doas_z[t]   = standardised (oas[t] - oas[t-1])
        ret[t+1]    = drift + lead_beta * doas_z[t] + ret_vol * u[t+1]

    With ``lead_beta < 0`` a widening in OAS this week lowers *next* week's return (the risk-off
    lead); ``= 0`` is the null (returns are unrelated to OAS moves). The returned frame is indexed by
    a weekly ``DatetimeIndex`` and carries, per row *t*:

    - ``oas``       — the OAS level at week *t* (bps),
    - ``d_oas``     — the raw weekly change (bps),
    - ``d_oas_z``   — the standardised weekly change (the signal, known at week *t*),
    - ``fwd_ret``   — the risk-asset return over week *t+1* (the thing the signal must lead),
    - ``ret``       — the contemporaneous week-*t* return (for reference).

    The last row's ``fwd_ret`` is dropped (no future to observe). Returns ``(frame, truth)``.
    """
    rng = np.random.default_rng(seed)

    # --- OAS path: OU mean-reversion + clustered stress jumps ---
    oas = np.empty(n_weeks)
    oas[0] = oas_mean
    eps = rng.standard_normal(n_weeks)
    # Poisson-ish stress jumps: rare, positive (spreads widen in stress), decaying tail.
    jump_hit = rng.random(n_weeks) < 0.012
    jump_size = np.where(jump_hit, rng.gamma(shape=2.0, scale=22.0, size=n_weeks), 0.0)
    for t in range(1, n_weeks):
        oas[t] = oas[t - 1] + oas_kappa * (oas_mean - oas[t - 1]) + oas_shock * eps[t] + jump_size[t]
    oas = np.clip(oas, 8.0, None)  # OAS cannot be deeply negative

    d_oas = np.diff(oas, prepend=oas[0])            # weekly change (first = 0)
    sd = d_oas.std(ddof=1)
    d_oas_z = (d_oas - d_oas.mean()) / sd if sd > 0 else d_oas * 0.0

    # --- risk-asset return: forward return led by THIS week's standardised OAS change ---
    u = rng.standard_normal(n_weeks)
    ret = np.empty(n_weeks)
    ret[:] = np.nan
    # ret[t+1] depends on d_oas_z[t]
    for t in range(0, n_weeks - 1):
        ret[t + 1] = drift + lead_beta * d_oas_z[t] + ret_vol * u[t + 1]
    ret[0] = drift + ret_vol * u[0]                 # first week has no prior signal

    idx = pd.date_range("2011-01-07", periods=n_weeks, freq="W-FRI")
    frame = pd.DataFrame(
        {
            "oas": oas,
            "d_oas": d_oas,
            "d_oas_z": d_oas_z,
            "ret": ret,
            "fwd_ret": np.append(ret[1:], np.nan),  # fwd_ret[t] = ret[t+1]
        },
        index=idx,
    )
    frame.index.name = "week"
    frame = frame.iloc[:-1].copy()                  # drop last row (no forward return)
    return frame, WorldTruth(lead_beta=lead_beta)


# ---------------------------------------------------------------------------
# Real tape — no free OAS series exists; documented stub
# ---------------------------------------------------------------------------
def _cache_path() -> str:
    return os.path.join(DEFAULT_CACHE, "mbs_oas.parquet")


def fetch_series(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real MBS-OAS tape hook — cache-first, but **no free source exists**.

    ``fetch=False`` (default): returns the cached parquet if a licensed user ever dropped one in
    ``_cache/``, else an **empty frame** — so the reproducible core stays fully offline.

    ``fetch=True``: raises ``NotImplementedError``. A clean, long agency-MBS *option-adjusted spread*
    series is a licensed vendor product (ICE BofA US MBS OAS, Bloomberg), not on any no-key retail
    feed; FRED exposes mortgage *yields* (MORTGAGE30US), not OAS. The wall is the point — this study
    is graded on the synthetic world and is capped below ``REAL`` on the SIGNAL axis.
    """
    path = _cache_path()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()
    raise NotImplementedError(
        "No free agency-MBS OAS series exists for a no-key stack. The option-adjusted spread is a "
        "licensed vendor index (ICE BofA US MBS OAS / Bloomberg); FRED's mortgage series are yields, "
        "not OAS. Drop a licensed weekly OAS parquet into _cache/mbs_oas.parquet to enable a real "
        "tape; this study is otherwise synthetic-only and capped below REAL."
    )


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
