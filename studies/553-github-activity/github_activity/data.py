"""Data layer for Study 553 (GitHub-Activity) — engineering telemetry as an alt-data nowcast.

The claim under test: a listed tech firm's **public open-source velocity** — the run-rate of
commits, merged pull-requests and new stars across its GitHub org — is a live *innovation-intensity*
signal. When a company's engineering output *accelerates*, the story goes, real product shipping is
accelerating too, and the market hasn't priced it yet — so a **surge in commit/star velocity should
foreshadow forward stock returns**. It is the alt-data dream: read the factory floor of software
before the earnings call does.

Why this study is **synthetic-only** (and honest about it):

- There is **no free, point-in-time, survivorship-clean mapping** from GitHub orgs to tickers.
  Firms rename, fork, archive, and *move code private*; the GitHub REST/GraphQL feeds are
  rate-limited and are a *current* snapshot (deleted repos and rewritten histories vanish), so a
  retail stack cannot reconstruct the commit/star velocity a trader would actually have seen at each
  past date. A backtest built on today's snapshot is look-ahead-and-survivorship-contaminated by
  construction.
- So we build a **deterministic synthetic panel** with a single knob (``ic``) that plants the
  cross-sectional predictive relation, and state the data-availability limitation openly on the
  SIGNAL axis. A synthetic-only study can **never** be `REAL` (that needs a robust *t* ≥ 2 on a
  real tape) — the ceiling here is `WEAK`.

One schema, one generator:

- ``synthetic_panel`` — a reproducible cross-section (one row per firm per rebalance period) of an
  *observable* GitHub velocity z-score and a realised **forward return**. The knob ``ic`` is the
  planted rank information-coefficient between velocity and next-period return:
  ``ic = 0`` is the null (velocity tells you nothing); ``ic > 0`` plants the believers' effect.
  The positive control and the null in one bottle. Fully offline; tests never touch the network.

No look-ahead: within each period the velocity z-score is measured over a *trailing* window and the
return it is scored against is the *forward* (next-period) return — one documented execution lag.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 553

# A stylised field of listed tech firms with visible public GitHub orgs (labels only; the panel is
# synthetic — these names anchor the *story*, not any real fetched data).
UNIVERSE = [
    "MSFT", "GOOGL", "META", "NVDA", "AMZN", "AAPL", "ORCL", "CRM", "ADBE", "IBM",
    "INTC", "AMD", "SNOW", "DDOG", "NET", "MDB", "TEAM", "SHOP", "PLTR", "TWLO",
    "OKTA", "ZS", "CRWD", "PANW", "NOW", "WDAY", "HUBS", "DOCN", "GTLB", "ESTC",
]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    ic: float  # planted rank information-coefficient (velocity -> forward return); 0 = null

    @property
    def has_signal(self) -> bool:
        return self.ic != 0.0


def synthetic_panel(
    n_firms: int = 30,
    n_periods: int = 40,
    ic: float = 0.08,
    ann_vol: float = 0.55,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible firm x period panel with a tunable velocity->return information-coefficient.

    Each period each firm has:

    - ``gh_vel`` — a standardised **GitHub velocity** score (commit + PR + star run-rate, blended
      and z-scored cross-sectionally each period; higher = the org is shipping faster than peers).
      Modelled as a persistent latent (firms have durable engineering cultures) plus period noise.
    - ``fwd_ret`` — the firm's realised **forward return** over the next period.

    The forward return is built so that the *cross-sectional* (Pearson/Spearman) correlation
    between ``gh_vel`` and ``fwd_ret`` is approximately ``ic``:

        fwd_ret = beta*mkt + sigma * ( ic * z(gh_vel) + sqrt(1-ic**2) * idio )

    (Signal and idio combined so the idiosyncratic sleeve keeps unit variance as ``ic`` varies, so
    ``ic`` reads directly as the planted per-period information-coefficient.) With ``ic = 0``
    velocity is uninformative (the null); a *small* ``ic`` (0.02-0.05) is the realistic alt-data
    magnitude the believers hope for.

    Returns a long/tidy frame with columns ``period, firm, gh_vel, fwd_ret`` and the ``WorldTruth``.
    """
    rng = np.random.default_rng(seed)
    n_firms = min(n_firms, len(UNIVERSE))
    firms = UNIVERSE[:n_firms]

    # Durable engineering-culture latent per firm (persists across periods).
    culture = rng.standard_normal(n_firms)

    ic = float(np.clip(ic, 0.0, 0.95))
    a = ic
    b = np.sqrt(max(1.0 - ic * ic, 0.0))

    per_period_vol = ann_vol / np.sqrt(4.0)  # ~quarterly rebalance
    betas = 0.8 + 0.5 * rng.standard_normal(n_firms)  # market betas, tech-heavy => >0

    rows = []
    for t in range(n_periods):
        # Observable velocity: persistent culture + period shock, z-scored across the field.
        raw_vel = 0.7 * culture + 0.7 * rng.standard_normal(n_firms)
        vel = (raw_vel - raw_vel.mean()) / (raw_vel.std(ddof=1) or 1.0)

        mkt = 0.02 + 0.06 * rng.standard_normal()  # period market factor
        idio = rng.standard_normal(n_firms)
        idio = (idio - idio.mean()) / (idio.std(ddof=1) or 1.0)

        fwd = betas * mkt + per_period_vol * (a * vel + b * idio)

        for j, fm in enumerate(firms):
            rows.append((t, fm, float(vel[j]), float(fwd[j])))

    panel = pd.DataFrame(rows, columns=["period", "firm", "gh_vel", "fwd_ret"])
    return panel, WorldTruth(ic=ic)


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes("number")
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
