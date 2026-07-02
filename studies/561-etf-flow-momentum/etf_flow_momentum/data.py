"""Data layer for Study 561 (ETF-Flow-Momentum) — the flow-chasing sort.

The claim (flow-based momentum): sector/asset ETFs that are pulling in the most *new money*
— net creation-unit inflows, i.e. shares outstanding rising as authorised participants create
units to meet demand — go on to keep outperforming their peers. The reversal view says the
opposite: heavy inflows mark crowded, over-extended sectors, so chasing flows is a trap and the
high-inflow ETFs subsequently *lag*.

**Data availability — synthetic only.** There is no free, no-key, point-in-time feed of ETF
*creation-unit flows*. Reconstructing net flows honestly needs a clean daily panel of
*shares-outstanding* per ETF joined to daily NAV — the shares-outstanding series that
data vendors sell, not something yfinance exposes reliably (its ``sharesOutstanding`` for ETFs
is a single stale scalar, not a daily history). A ``price × Δshares`` proxy built from what a
retail stack can reach would smuggle in exactly the look-ahead and staleness the study is trying
to avoid. So this study is **synthetic-only**: the free real data does not exist. Per the desk's
rubric a synthetic-only study can never earn ``REAL`` (that needs a robust ``t >= 2`` on a real
tape) — it is capped at ``WEAK`` / ``NONE``, and the data-availability limit is stated openly on
the SIGNAL axis (like the lego-returns, whisky-cask and sneaker-resale studies).

The generator below is deterministic and offline. A single knob, ``flow_alpha``, plants the
effect:

- ``flow_alpha > 0`` — flow *momentum*: the ETFs with the largest recent inflows earn the
  *highest* forward returns (the claim, at full strength).
- ``flow_alpha < 0`` — flow *reversal*: heavy-inflow ETFs subsequently *lag* (the trap).
- ``flow_alpha = 0`` — the null: forward return is unrelated to flow.

One panel schema, two roles: the positive control (knob on) and the null (knob off) in one
bottle. Tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

# A plausible sector/asset-ETF universe (labels only — the panel is synthetic, no prices are
# fetched). Roughly the SPDR sector suite plus a few asset/thematic sleeves, so the reader has a
# concrete mental picture of "which ETF pulled in money".
UNIVERSE = [
    "XLK",  # tech
    "XLF",  # financials
    "XLE",  # energy
    "XLV",  # health care
    "XLI",  # industrials
    "XLY",  # consumer discretionary
    "XLP",  # consumer staples
    "XLU",  # utilities
    "XLB",  # materials
    "XLRE", # real estate
    "XLC",  # communication services
    "GLD",  # gold
    "TLT",  # long treasuries
    "HYG",  # high yield
    "EEM",  # emerging markets
    "IWM",  # small caps
]

MONTHS_PER_YEAR = 12


class WorldTruth:
    """The planted effect for the synthetic panel."""

    def __init__(self, flow_alpha: float):
        self.flow_alpha = float(flow_alpha)

    @property
    def has_effect(self) -> bool:
        return self.flow_alpha != 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"WorldTruth(flow_alpha={self.flow_alpha!r})"


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_etfs: int = 16,
    n_months: int = 120,
    flow_alpha: float = 0.005,
    flow_persist: float = 0.55,
    idio_vol: float = 0.05,
    seed: int = 561,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible ETF-flow panel with a tunable flow→return effect.

    Each of ``n_etfs`` funds has a latent *popularity* state that drifts month to month
    (``flow_persist`` controls how sticky flows are — real ETF flows are strongly
    autocorrelated). From that state we derive the **observable** monthly *net flow* (as a
    fraction of prior-month assets: positive = creations/inflows, negative = redemptions), and
    a forward one-month return:

        ret[t+1] = market[t+1] + flow_alpha * z(flow[t]) + idio

    where ``z(flow[t])`` is the cross-sectional z-score of this month's flow. With
    ``flow_alpha > 0`` the high-inflow ETFs earn the *highest* next-month return (flow momentum);
    ``< 0`` is the reversal/trap; ``= 0`` is the null. A common ``market[t+1]`` term is shared
    across funds each month (so the panel has a real cross-sectional common factor the flow signal
    must beat, not just idiosyncratic noise).

    Returns a long panel with columns ``month``, ``etf``, ``flow`` (this month's net flow,
    the signal), ``fwd_ret`` (next month's return, the outcome), plus the ``true_pop`` latent
    for diagnostics — one row per (etf, month). This is exactly the schema the strategy consumes.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    tickers = list(UNIVERSE)[:n_etfs] if n_etfs <= len(UNIVERSE) else [
        f"E{j:02d}" for j in range(n_etfs)
    ]
    n_etfs = len(tickers)

    # Latent popularity state per ETF, an AR(1) so flows are sticky (autocorrelated).
    pop = rng.standard_normal(n_etfs) * 0.5
    rows = []
    for t in range(n_months):
        shock = rng.standard_normal(n_etfs)
        pop = flow_persist * pop + np.sqrt(max(1e-9, 1.0 - flow_persist**2)) * shock
        # Observable net flow this month: driven by popularity + measurement noise, in
        # fraction-of-assets units (roughly +/- a few % of AUM/month, the real ETF scale).
        flow = 0.015 * pop + 0.006 * rng.standard_normal(n_etfs)
        # Cross-sectional z-score of flow (the signal the sort actually uses).
        fmu, fsd = flow.mean(), flow.std(ddof=1)
        zflow = (flow - fmu) / fsd if fsd > 0 else flow * 0.0
        # Forward one-month return: common market factor + planted flow effect + idio.
        market = 0.007 + 0.04 * rng.standard_normal()  # shared across funds this month
        idio = idio_vol * rng.standard_normal(n_etfs)
        fwd = market + flow_alpha * zflow + idio
        for j, tk in enumerate(tickers):
            rows.append(
                {
                    "month": t,
                    "etf": tk,
                    "flow": float(flow[j]),
                    "fwd_ret": float(fwd[j]),
                    "true_pop": float(pop[j]),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, WorldTruth(flow_alpha=flow_alpha)


# ---------------------------------------------------------------------------
# "Real" fetch stub — kept for schema/parity; the free real data does not exist.
# ---------------------------------------------------------------------------
def fetch_panel(fetch: bool = False) -> pd.DataFrame:
    """Real-tape loader — intentionally empty (see the module docstring).

    There is no free, no-key, point-in-time daily *shares-outstanding* feed to reconstruct
    honest ETF creation-unit flows, so this study is synthetic-only. The loader exists for
    schema parity with the rest of the desk and always returns an empty frame (``fetch``
    is accepted but ignored — there is nothing to fetch).
    """
    return pd.DataFrame(columns=["month", "etf", "flow", "fwd_ret"])


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
