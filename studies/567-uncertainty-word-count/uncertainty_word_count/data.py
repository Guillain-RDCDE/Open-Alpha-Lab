"""Data layer for Study 567 (Uncertainty-Word-Count) — the LM-uncertainty firm-event panel.

The unit of observation is a **firm disclosure event**: a firm files (or holds a call), and the text
carries an *uncertainty density* — the share of tokens that fall in the Loughran-McDonald (2011)
**Uncertainty** word list ("uncertain", "maybe", "could", "risk", "approximately", "depend", ...).
The primary outcome is the firm's **forward realised volatility** — the annualised return volatility
over the window *after* the filing — which the hedging density is claimed to forecast; the secondary
outcome is the **forward return** over the same window (the softer "lower returns" half of the claim).

Two tapes, one schema (columns: ``ticker``, ``period``, ``uncert``, ``trail_vol``, ``fwd_vol``,
``fwd_ret``):

- ``synthetic_panel`` — a deterministic, offline generator (fixed seed = 567). A single knob,
  ``uncert_beta``, plants the uncertainty→forward-vol link: forward vol loads on the *residual*
  uncertainty density (the part orthogonal to trailing vol), so ``uncert_beta = 0`` is the null
  (hedging words tell you nothing beyond how jumpy the firm already was) and ``uncert_beta > 0`` is
  the planted anomaly (hedgier text → higher forward vol). A second knob, ``ret_beta`` (≤ 0), plants
  the secondary return drag. This is the study's positive control and null in one bottle; tests never
  touch the network.

- ``fetch_panel`` — the **real-tape stub.** Parsed EDGAR filings scored against a licensed LM
  uncertainty lexicon and joined to event-time forward vol are a paid-vendor / hand-parsed academic
  product, not something a no-key retail stack can reach. So this returns the cached panel if a reader
  has dropped one into ``_cache/`` in the study schema, and otherwise an **empty frame** — never a
  fabricated tape. Because there is no free real feed, the Signal axis is capped at ``WEAK`` (a
  ``REAL`` stamp needs a robust *t* ≥ 2 on a real tape) and the limitation is named openly, like the
  desk's lego-returns / whisky-cask / earnings-call-tone synthetic-only studies.

Honesty baked into the generator:

- **Uncertainty is entangled with recent volatility.** Firms that are already jumpy write hedgier
  disclosures, so the naive uncertainty→forward-vol slope double-counts the *persistence* of vol
  (realised vol is strongly autocorrelated). The generator makes ``uncert`` correlate with
  ``trail_vol`` on purpose, so the strategy must *control for trailing vol* to isolate the textual
  edge — the study's central honesty check.
- **Execution lag.** Forward vol / return are measured strictly *after* the filing period; the signal
  (uncertainty density) is known at the filing, the position is entered the next session, no future
  data enters the score.
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

# A fixed illustrative universe of large-cap names for the synthetic panel's ticker labels only
# (the numbers are generated, not fetched — the labels just make the panel readable).
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "ORCL",
    "JPM", "BAC", "WFC", "GS", "XOM", "CVX", "JNJ", "PFE", "MRK", "ABBV",
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "DIS", "CAT", "GE",
    "BA", "HON", "T", "VZ", "CMCSA", "NFLX", "CRM", "ADBE", "INTC", "AMD",
    "QCOM", "TXN",
]

# A tiny illustrative slice of the Loughran-McDonald Uncertainty word list (for the notebooks'
# plain-language layer only; the synthetic panel never parses text — it generates the density).
LM_UNCERTAINTY_SAMPLE = [
    "uncertain", "maybe", "could", "risk", "approximately", "depend", "possibly",
    "perhaps", "unclear", "fluctuate", "presumably", "roughly", "seldom", "sometimes",
    "believe", "may", "might", "likelihood", "indefinite", "unpredictable",
]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    uncert_beta: float  # forward-vol loading on *residual* uncertainty (0 = null; >0 = the anomaly)
    ret_beta: float     # forward-return loading on residual uncertainty (<=0 = the return drag)

    @property
    def has_anomaly(self) -> bool:
        return self.uncert_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic firm-event panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 40,
    n_periods: int = 12,
    uncert_beta: float = 0.020,
    ret_beta: float = -0.020,
    vol_persist: float = 0.55,
    base_vol: float = 0.28,
    fwd_vol_idio: float = 0.09,
    fwd_ret_idio: float = 0.055,
    seed: int = 567,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible firm-event panel with a tunable *uncertainty-word* anomaly.

    For each firm-period we draw a trailing realised vol ``trail_vol`` (already-jumpy firms), and an
    ``uncert`` density that is *entangled* with it — jumpy firms write hedgier text — plus an
    idiosyncratic uncertainty component (the residual hedging a textual edge would live in)::

        uncert     = base + load * z(trail_vol) + uncert_resid
        fwd_vol     = vol_persist * trail_vol                # vol is autocorrelated (the confound)
                      + uncert_beta * uncert_resid           # the TEXTUAL edge (the claim)
                      + vol_idio
        fwd_ret     = market_drift
                      + ret_beta   * uncert_resid            # the softer return-drag half
                      + ret_idio

    Because forward vol loads on the *residual* uncertainty (the part orthogonal to trailing vol),
    the honest test must control for ``trail_vol``: the naive uncertainty→forward-vol slope is
    inflated by vol persistence, and only the trail-vol-controlled slope isolates the words.
    ``uncert_beta = 0`` is the null (hedging adds nothing beyond how jumpy the firm already was);
    ``uncert_beta > 0`` plants the anomaly. One row per firm-period with columns ``ticker``,
    ``period``, ``uncert``, ``trail_vol``, ``fwd_vol``, ``fwd_ret`` — the schema ``fetch_panel``
    would produce from real filings.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    tickers = (UNIVERSE * ((n_firms // len(UNIVERSE)) + 1))[:n_firms]

    rows = []
    for pd_ix in range(n_periods):
        # Already-jumpy firms: trailing realised vol, lognormal-ish, positive.
        trail_vol = np.clip(base_vol + 0.12 * rng.standard_normal(n_firms), 0.06, 1.2)
        tv_z = (trail_vol - trail_vol.mean()) / (trail_vol.std(ddof=1) or 1.0)

        uncert_resid = rng.standard_normal(n_firms)                  # residual hedging (edge lives here)
        # Uncertainty density is entangled with trailing vol (jumpy firms hedge) + residual.
        # Trailing vol is the DOMINANT driver of the density (the confound), residual is secondary:
        uncert = 0.010 + 0.0035 * tv_z + 0.0018 * uncert_resid       # density ~1% of tokens

        vol_idio = fwd_vol_idio * rng.standard_normal(n_firms)
        fwd_vol = np.clip(
            vol_persist * trail_vol + uncert_beta * uncert_resid + vol_idio, 0.03, 2.0
        )

        market_drift = 0.02
        ret_idio = fwd_ret_idio * rng.standard_normal(n_firms)
        fwd_ret = market_drift + ret_beta * uncert_resid + ret_idio

        for j in range(n_firms):
            rows.append(
                {
                    "ticker": tickers[j],
                    "period": f"P{pd_ix + 1:02d}",
                    "uncert": float(uncert[j]),
                    "trail_vol": float(trail_vol[j]),
                    "fwd_vol": float(fwd_vol[j]),
                    "fwd_ret": float(fwd_ret[j]),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, WorldTruth(uncert_beta=uncert_beta, ret_beta=ret_beta)


# ---------------------------------------------------------------------------
# Real tape — a cache-first STUB (no free feed exists; empty by design)
# ---------------------------------------------------------------------------
def _panel_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "uncertainty_panel.parquet")


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real uncertainty-density → forward-vol/return panel, cache-first — **stub**.

    Parsed EDGAR filings scored against a licensed LM uncertainty lexicon and joined to event-time
    forward volatility are a paid-vendor / hand-parsed product; there is no free, no-key feed to
    download. So this returns:

    - the cached panel if a reader has placed one at ``_cache/uncertainty_panel.parquet`` in the
      study schema (``ticker``, ``period``, ``uncert``, ``trail_vol``, ``fwd_vol``, ``fwd_ret``), or
    - an **empty frame** otherwise (``fetch=True`` cannot manufacture data that does not exist).

    Never fabricates a tape. The absence of a free real feed is why the Signal axis is capped at
    ``WEAK`` and named on the SIGNAL axis.
    """
    path = _panel_cache()
    if os.path.exists(path):
        df = pd.read_parquet(path)
        needed = {"ticker", "period", "uncert", "trail_vol", "fwd_vol", "fwd_ret"}
        if needed.issubset(df.columns):
            return df
    # No free feed to fetch; return empty regardless of `fetch` (documented stub).
    _ = fetch
    return pd.DataFrame(
        columns=["ticker", "period", "uncert", "trail_vol", "fwd_vol", "fwd_ret"]
    )


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
