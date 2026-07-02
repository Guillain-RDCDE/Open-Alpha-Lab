"""Data layer for Study 562 (Block-Trade-Signal) — the informed-block event panel.

The folklore: when a **giant block crosses the tape** — a single print far larger than the
prevailing trade size — it is a fingerprint of an *informed institution* who knows something, and
the stock will *drift* in the direction of that informed flow afterward. The tradable expression
is a **coattail** trade: infer the block's likely sign (buyer- vs seller-initiated), take the same
side, and hold for the drift. The skeptical view (the microstructure literature's own finding):
block prints are largely *liquidity* events negotiated upstairs at a temporary price concession
that **reverts**, not informed events that persist — so the "drift" is either absent or the wrong
sign after costs.

**Data availability — synthetic only.** There is no free, no-key, point-in-time feed of
*classified* block trades. A real block-informedness study needs (a) the full trade-and-quote
(TAQ) tape to flag prints above a size threshold, (b) a Lee-Ready-style quote-rule classification
of each block as buyer- or seller-initiated (which needs the *contemporaneous* national best
bid/offer at the microsecond of the print), and (c) survivorship-free forward returns. yfinance and
every free retail feed expose only daily OHLCV — no intraday prints, no quotes, no trade-direction
sign. A ``daily-volume-spike`` proxy for "a block happened" would smuggle in exactly the
misclassification and look-ahead the study is trying to avoid (a volume spike is not a block, and
its direction is unobservable from daily bars). So this study is **synthetic-only**: the free real
data does not exist. Per the desk's rubric a synthetic-only study can never earn ``REAL`` (that
needs a robust ``t >= 2`` on a real tape) — it is capped at ``WEAK`` / ``NONE``, and the
data-availability limit is stated openly on the SIGNAL axis (like the desk's lego-returns,
whisky-cask and sneaker-resale studies).

The generator below is deterministic and offline. A single knob, ``drift_alpha``, plants the
effect:

- ``drift_alpha > 0`` — informed blocks: the coattail (same-side) trade *earns* forward drift
  (the folklore, at full strength).
- ``drift_alpha < 0`` — liquidity blocks: the price *reverts* after the block, so riding the
  coattail *loses* (the microstructure skeptic's world).
- ``drift_alpha = 0`` — the null: forward return is unrelated to the block's inferred side.

One panel schema, three roles: the positive control (knob > 0), the reversal control (knob < 0)
and the null (knob = 0) in one bottle. Tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

# A plausible large-cap universe (labels only — the panel is synthetic, no prices are fetched).
# The reader gets a concrete mental picture of "which tape a block crossed".
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "PG", "KO",
    "HD", "MCD", "DIS", "WMT", "CAT", "GE", "BA", "T",
]

TRADING_DAYS = 252


class WorldTruth:
    """The planted effect for the synthetic event panel."""

    def __init__(self, drift_alpha: float):
        self.drift_alpha = float(drift_alpha)

    @property
    def has_effect(self) -> bool:
        return self.drift_alpha != 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"WorldTruth(drift_alpha={self.drift_alpha!r})"


# ---------------------------------------------------------------------------
# Synthetic event panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 24,
    n_events: int = 1200,
    drift_alpha: float = 0.010,
    informed_share: float = 0.5,
    ret_vol: float = 0.030,
    seed: int = 562,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible block-print event panel with a tunable coattail-drift effect.

    Each row is one **block-print event**: a giant trade crosses the tape in some stock on some
    day. Every event carries the *observable* characteristics a tape-watcher actually sees —

    - ``block_z``    : the print's size relative to the stock's normal trade size (how *giant*
                       the block is; higher = more eye-catching),
    - ``sign``       : the *inferred* side, +1 buyer-initiated / -1 seller-initiated, from a
                       Lee-Ready-style quote rule (this is what a coattail trader would copy),
    - ``signed_z``   : ``sign * block_z`` — the signed informed-flow signal the strategy trades,

    and the outcome the folklore is about —

    - ``fwd_ret``    : the stock's forward return over the event window *in the block's direction*
                       already folded in via ``sign`` (see below).

    The data-generating process. A latent fraction ``informed_share`` of blocks are *informed*
    (they carry private information); the rest are *liquidity* blocks (an institution rebalancing,
    no information). The forward return in the block's inferred direction is::

        fwd_ret = drift_alpha * signed_z_std * informed_flag + market + idio

    where ``signed_z_std`` is the cross-event-standardised signed size and ``informed_flag`` is 1
    for informed blocks, 0 for liquidity blocks. With ``drift_alpha > 0`` the *informed* blocks
    drift the right way (coattail pays); ``< 0`` makes the price *revert* after the block (the
    liquidity-event skeptic's world, where the coattail loses); ``= 0`` is the null. The
    ``sign`` itself is measured with noise (real quote-rule classification is imperfect), and a
    common ``market`` term is shared across events on the same (stock, day) so the panel has a real
    common factor the signal must beat, not just idiosyncratic noise.

    Returns a long panel with one row per event and columns ``stock``, ``day``, ``block_z``,
    ``sign``, ``signed_z``, ``fwd_ret``, plus the ``informed`` latent for diagnostics — exactly the
    schema the strategy consumes.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    tickers = list(UNIVERSE)[:n_stocks] if n_stocks <= len(UNIVERSE) else [
        f"S{j:02d}" for j in range(n_stocks)
    ]
    n_stocks = len(tickers)

    stock_idx = rng.integers(0, n_stocks, size=n_events)
    day = rng.integers(0, 504, size=n_events)  # ~2 years of trading days

    # Block size relative to normal trade size: a right-skewed magnitude (most blocks are "big",
    # a tail are "giant"). Exponential gives the fat tail; +1 so the smallest block is still a block.
    block_z = 1.0 + rng.exponential(1.0, size=n_events)

    # Inferred side from a noisy Lee-Ready-style quote rule: a latent true side, corrupted by
    # classification noise (real quote-rule direction is imperfect near the midpoint).
    true_side = rng.choice([-1.0, 1.0], size=n_events)
    misclassified = rng.random(n_events) < 0.15  # ~15% of blocks mis-signed
    sign = np.where(misclassified, -true_side, true_side)

    # Which blocks are informed vs pure liquidity.
    informed = (rng.random(n_events) < informed_share).astype(float)

    signed_z = sign * block_z

    idio = ret_vol * rng.standard_normal(n_events)
    # Shared (stock, day) market shock so the cross-section has a real common factor. It hits the
    # coattail P&L in the *inferred* side's direction (``sign``), so it is a genuine common factor
    # the informed-flow signal must beat, not just idiosyncratic noise.
    key = stock_idx * 10_000 + day
    _, inv = np.unique(key, return_inverse=True)
    common = 0.012 * rng.standard_normal(len(_))
    market = sign * common[inv]

    # The planted informed drift. It is expressed in the *coattail's* direction: an informed block
    # that is correctly classified (``sign == true_side``) drifts the coattail's way; a
    # *mis*-classified informed block drifts against it (you copied the wrong side). Liquidity
    # blocks (informed_flag = 0) contribute no drift. Magnitude scales with the standardised block
    # size (giant blocks move more). This is the P&L of copying the block, before costs.
    correct = sign * true_side  # +1 if the inferred side matches the true side, else -1
    fwd_ret = drift_alpha * block_z * informed * correct + market + idio

    panel = pd.DataFrame(
        {
            "stock": [tickers[i] for i in stock_idx],
            "day": day.astype(int),
            "block_z": block_z,
            "sign": sign,
            "signed_z": signed_z,
            "fwd_ret": fwd_ret,
            "informed": informed,
        }
    )
    return panel, WorldTruth(drift_alpha=drift_alpha)


# ---------------------------------------------------------------------------
# "Real" fetch stub — kept for schema/parity; the free real data does not exist.
# ---------------------------------------------------------------------------
def fetch_panel(fetch: bool = False) -> pd.DataFrame:
    """Real-tape loader — intentionally empty (see the module docstring).

    There is no free, no-key feed of *classified* block trades (the intraday TAQ tape plus a
    quote-rule buy/sell classification plus survivorship-free forward returns), so this study is
    synthetic-only. The loader exists for schema parity with the rest of the desk and always
    returns an empty frame (``fetch`` is accepted but ignored — there is nothing to fetch).
    """
    return pd.DataFrame(columns=["stock", "day", "block_z", "sign", "signed_z", "fwd_ret"])


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
