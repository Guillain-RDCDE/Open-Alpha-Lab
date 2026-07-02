"""Data layer for Study 565 (Filing-Readability) — the Loughran-McDonald text anomaly.

Loughran & McDonald (2014) argue that raw readability formulas (Gunning **fog**) are noisy in a
financial setting and that plain **file size** of the 10-K is a cleaner, more robust proxy for how
hard a filing is to process. The anomaly: firms with **less readable** (higher fog, longer, bigger
file) 10-Ks earn **lower** subsequent returns and show **higher** post-filing return volatility —
consistent with managers obfuscating bad news and the market underreacting.

This study works with what a no-key retail stack can actually reach — which for point-in-time 10-K
*text* is **nothing clean**:

- The 10-K full text lives on SEC EDGAR, but a survivorship-free, point-in-time panel that lines up
  each filing's readability with the *right firm's forward return* (via CRSP/Compustat identifiers)
  is not a free retail artifact. yfinance has no filing text at all.

So Study 565 is **synthetic-only**. Two entry points share one schema:

- ``synthetic_panel`` — a deterministic, offline generator. A single knob, ``obf_alpha``, plants
  the anomaly: the *less readable* a firm's filing (higher latent obfuscation), the **lower** its
  forward return (``obf_alpha < 0`` reproduces the LM readability anomaly; ``= 0`` is the null).
  The positive control and the null in one bottle. Tests never touch the network.
- ``fetch_panel`` — the real-tape **stub**. It is cache-first like every study, but the free stack
  cannot populate it, so it returns an empty frame by construction (``fetch=False`` default). The
  hook exists so a reader with an EDGAR-text + point-in-time-returns pipeline can drop a cached
  panel in and re-run; without it, ``HAVE_REAL`` is ``False`` everywhere.

Survivorship & point-in-time: a real replication needs (a) the *filing-date* readability, lined up
with the return measured *after* the filing is public (a documented ~1-day+ execution lag), and (b)
the delisted/bankrupt firms — the murkiest filers are disproportionately the ones that later blow
up, so a survivor panel biases *against* the anomaly. Both are named on the SIGNAL axis. Because
there is no real tape, this study is **capped at WEAK/NONE** and can never be `REAL`.
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

SEED = 565

# The three observable readability legs the LM literature names, in the schema the strategy reads:
#   fog          — Gunning fog index of the 10-K (higher = harder to read)
#   length_kw    — document length in thousands of words (longer = harder to process)
#   filesize_mb  — the plain gross file size of the 10-K in MB (LM's preferred robust proxy)
READABILITY_LEGS = ("fog", "length_kw", "filesize_mb")


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic readability panel."""

    obf_alpha: float  # forward-return alpha per unit of latent obfuscation (negative = the anomaly)

    @property
    def has_anomaly(self) -> bool:
        return self.obf_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 400,
    obf_alpha: float = -0.10,
    idio_vol: float = 0.14,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section with a tunable *readability anomaly*.

    Each firm gets a latent **obfuscation** level ``g`` (how hard its 10-K is to read) drawn from a
    log-normal-ish right-skewed distribution (most filings readable, a long murky tail). From ``g``
    we derive the *observable* legs the strategy sees — a high fog index, a long document, a big
    file — each a noisy monotone transform of ``g`` (so no single leg is a perfect proxy, matching
    LM's point that fog is noisy and file size is cleaner). Forward return is:

        forward_ret = market_drift + obf_alpha * (g - g_bar) + idio

    and the post-filing idiosyncratic volatility *rises* with ``g`` (murkier filings, more
    uncertainty), reproducing the LM "higher risk" leg.

    With ``obf_alpha < 0`` the least-readable firms earn the *lowest* return (the anomaly);
    ``obf_alpha = 0`` is the null. The returned frame has one row per firm with the columns the
    strategy consumes — ``fog``, ``length_kw``, ``filesize_mb``, ``forward_ret``, ``post_vol`` —
    exactly the schema ``fetch_panel`` would produce from a real EDGAR panel.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Latent obfuscation: standard-normal factor pushed through exp for a right-skewed murky tail.
    z = rng.standard_normal(n_stocks)
    g = np.exp(0.45 * z)                       # >0, right-skewed; median ~1
    g_bar = g.mean()

    # Observable legs — noisy monotone transforms of g (fog noisiest, file size cleanest, per LM):
    fog = np.clip(16.0 + 4.5 * (g - g_bar) + 2.2 * rng.standard_normal(n_stocks), 8.0, 30.0)
    length_kw = np.clip(45.0 + 22.0 * (g - g_bar) + 9.0 * rng.standard_normal(n_stocks), 8.0, 260.0)
    filesize_mb = np.clip(2.4 + 1.6 * (g - g_bar) + 0.5 * rng.standard_normal(n_stocks), 0.3, 18.0)

    drift = 0.09
    idio = idio_vol * rng.standard_normal(n_stocks)
    forward_ret = drift + obf_alpha * (g - g_bar) + idio

    # The "higher risk" leg: post-filing idiosyncratic vol rises with obfuscation.
    post_vol = np.clip(0.22 + 0.16 * (g - g_bar) + 0.03 * rng.standard_normal(n_stocks), 0.05, 1.5)

    tickers = [f"F{j:04d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "fog": fog,
            "length_kw": length_kw,
            "filesize_mb": filesize_mb,
            "forward_ret": forward_ret,
            "post_vol": post_vol,
            "true_g": g,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(obf_alpha=obf_alpha)


# ---------------------------------------------------------------------------
# Real tape — the EDGAR-text stub (empty by construction on a free stack)
# ---------------------------------------------------------------------------
def _panel_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "readability_panel.parquet")


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real-tape stub — cache-first, empty by construction on a free retail stack.

    A real readability panel needs 10-K full text (SEC EDGAR) parsed into fog / length / file size,
    lined up *by filing date* with each firm's *forward* return from a survivorship-free,
    point-in-time return tape (CRSP/Compustat identifiers). None of that is a free yfinance-style
    artifact, so there is no ``fetch=True`` path that a no-key stack can honour: this returns the
    cached parquet if a reader has dropped one in, else an **empty frame**.

    The columns a dropped-in cache must carry to work with the engine:
    ``fog``, ``length_kw``, ``filesize_mb``, ``forward_ret`` (and optionally ``post_vol``).
    """
    path = _panel_cache()
    if os.path.exists(path):
        panel = pd.read_parquet(path)
        return panel
    # No free fetch path exists; the stub stays empty whether or not fetch is requested.
    return pd.DataFrame()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        cols = [c for c in obj.columns if obj[c].dtype != object]
        arr = np.ascontiguousarray(obj[cols].fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
