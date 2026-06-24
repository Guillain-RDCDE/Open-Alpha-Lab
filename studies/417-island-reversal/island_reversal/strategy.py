"""Detector + timing rule + honest arbiters — Study 417 (Island Reversal).

The island reversal is the textbook gap-cluster-gap **reversal** figure (Edwards & Magee,
*Technical Analysis of Stock Trends*, 1948; Bulkowski, *Encyclopedia of Chart Patterns*). The
folk recipe:

* **Island top (bearish).** After a rally, an **exhaustion gap up** opens the price clear above
  the prior bar's range. The price then trades for one to a few sessions stranded at that high —
  the *island* — and is sealed off by a **breakaway gap down** that opens *below* the island's
  low, leaving a visible "island" of bars marooned in empty chart space above. Two gaps, opposite
  directions, bracketing the cluster: the rally is exhausted, you **short** the second gap.
* **Island bottom (bullish).** The mirror image: an exhaustion gap **down**, a cluster of bars
  stranded at the low, then a breakaway gap **up** that re-crosses above the island's high. The
  downtrend is done, you **buy** the second gap.

Chart figures are partly in the eye of the beholder — *"a few bars marooned by two gaps"* leaves
real choices (how big a gap counts? how many island bars? must the gaps clear the same far edge?).
So we test the closest **mechanical** definition we can write down and we say so. Our detector, on
daily OHLC:

* An **island top** is: (1) a gap-up bar — its open is at least ``gap_pct`` above the *prior*
  bar's high; (2) an **island** of 1..``max_island`` bars whose collective **low stays above** the
  prior bar's high (the bars never fill the first gap); (3) a **sealing gap-down** bar whose open
  is at least ``gap_pct`` below the island's **low** *and* below the bar that preceded the first
  gap (the island is truly isolated). The sealing gap-down bar is the **signal bar** (the figure
  is only known at its open/close).
* An **island bottom** is the exact mirror with the directions flipped.

Timing: the figure is confirmed at the **sealing gap bar's close**; we **enter the next close**
(one documented lag) and hold ``H`` trading days. The honest question: do forward returns after a
*confirmed* island reversal beat the name's own base rate, in the figure's intended direction?
Arbiters: a one-sample / HAC *t* of the post-island excess over the base rate, a **same-tape
random-date placebo** (the honest control for the tape's own drift), one-way costs, a
detector-strictness robustness sweep (the gap threshold), and the synthetic positive control. A
natural myth-check rides along: the **island bottom** (the bullish twin) — does the figure really
reverse symmetrically on both sides?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)        # trading-day forward horizons


# --------------------------------------------------------------------------- #
# Island detector
# --------------------------------------------------------------------------- #
def detect_islands(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   gap_pct: float = 0.02, max_island: int = 3,
                   side: str = "top") -> list[dict]:
    """Detect confirmed island-top (or island-bottom) reversals on a name's OHLC arrays.

    ``side="top"`` (bearish): an exhaustion gap **up**, a cluster stranded at the high, then a
    sealing gap **down** — the signal is a **short**. ``side="bottom"`` (bullish): the mirror.

    Island top rule (positions are integer bar indices):

    * Bar ``g1`` gaps up: ``open[g1] >= high[g1-1] * (1 + gap_pct)``.
    * The island = bars ``g1 .. g2-1`` (1..``max_island`` of them) whose **low never falls back to
      or below** ``high[g1-1]`` (the first gap stays open across the whole island).
    * Bar ``g2`` (the next bar after the island) gaps down and seals it:
      ``open[g2] <= island_low * (1 - gap_pct)`` **and** ``open[g2] <= high[g1-1]`` (it opens back
      below the pre-gap range, so the island is isolated on both sides).

    Island bottom is the exact mirror (gap down, cluster at the low, sealing gap up). Non-
    overlapping: once an island is taken the scan resumes after its sealing bar.

    Returns a list of dicts, one per confirmed island, with: ``g1`` (first gap bar), ``island``
    (list of island bar indices), ``signal_idx`` = ``g2`` (the sealing gap bar = the entry
    signal), ``island_extreme`` (the island high for a top / low for a bottom), ``side``.
    """
    if side not in ("top", "bottom"):
        raise ValueError(f"side must be 'top' or 'bottom', got {side!r}")
    n = len(open_)
    out: list[dict] = []
    i = 1
    while i < n - 1:
        if side == "top":
            gapped = open_[i] >= high[i - 1] * (1.0 + gap_pct)
        else:
            gapped = open_[i] <= low[i - 1] * (1.0 - gap_pct)
        if not gapped:
            i += 1
            continue
        ref = high[i - 1] if side == "top" else low[i - 1]   # the far edge of the first gap
        # grow the island while the first gap stays open and we are within max_island
        found = None
        for isl_len in range(1, max_island + 1):
            g2 = i + isl_len
            if g2 >= n:
                break
            island = list(range(i, g2))
            # the island must never fill the first gap
            if side == "top":
                if np.min(low[island]) <= ref:
                    break                                    # gap filled -> not an island
                island_low = float(np.min(low[island]))
                seal = (open_[g2] <= island_low * (1.0 - gap_pct)) and (open_[g2] <= ref)
                extreme = float(np.max(high[island]))
            else:
                if np.max(high[island]) >= ref:
                    break
                island_high = float(np.max(high[island]))
                seal = (open_[g2] >= island_high * (1.0 + gap_pct)) and (open_[g2] >= ref)
                extreme = float(np.min(low[island]))
            if seal:
                found = {"g1": int(i), "island": [int(x) for x in island],
                         "signal_idx": int(g2), "island_extreme": extreme,
                         "side": side}
                break
        if found is not None:
            out.append(found)
            i = found["signal_idx"] + 1
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Forward returns after the sealing gap
# --------------------------------------------------------------------------- #
def forward_returns(close: np.ndarray, signal_idx: list[int], horizon: int,
                    lag: int = 1, direction: int = -1) -> np.ndarray:
    """Forward ``horizon``-day return entered ``lag`` bars after each signal close.

    Signal (the sealing gap) known at ``signal_idx`` close; enter at ``signal_idx + lag`` close
    (one documented lag); exit ``horizon`` bars later. ``direction`` is −1 for an island-top
    short and +1 for an island-bottom long, so the returned numbers are the **trade's** P&L in
    the believer's intended direction. Drops signals whose window overruns the tape.
    """
    out = []
    n = len(close)
    for si in signal_idx:
        entry = si + lag
        exit_ = entry + horizon
        if entry < 0 or exit_ >= n:
            continue
        out.append(direction * (close[exit_] / close[entry] - 1.0))
    return np.asarray(out, dtype=float)


def base_rate(close: np.ndarray, horizon: int, direction: int = -1) -> float:
    """The name's own unconditional mean forward ``horizon``-day return (the base rate).

    ``direction`` flips the sign for a short book so the excess is apples-to-apples (a short's
    base rate is the negative of buy-and-hold).
    """
    if len(close) <= horizon + 1:
        return float("nan")
    r = close[horizon:] / close[:-horizon] - 1.0
    return float(direction * np.mean(r))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def one_sample_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``sample`` mean against ``mu0``. NaN if fewer than 2 points."""
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu0) / se)


def hac_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """Newey-West (HAC) t of the mean against ``mu0`` — autocorrelation-robust.

    Events are roughly ordered in time; we use a small Bartlett lag so clustered islands don't
    inflate the t. No quantlab dependency.
    """
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return one_sample_t(r, mu0)
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wgt = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wgt * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    if se == 0:
        return float("nan")
    return float((mu - mu0) / se)


# --------------------------------------------------------------------------- #
# Panel-level orchestration
# --------------------------------------------------------------------------- #
def collect_islands(panel: dict, names: list[str] | None = None,
                    side: str = "top", **detect_kw) -> dict[str, list[dict]]:
    """Run :func:`detect_islands` on every name in the panel; return ticker -> island list."""
    o, h, lo = panel["open"], panel["high"], panel["low"]
    if names is None:
        names = list(panel["close"].columns)
    out = {}
    for tk in names:
        sub = pd.DataFrame({"o": o[tk], "h": h[tk], "lo": lo[tk]}).dropna()
        out[tk] = detect_islands(sub["o"].to_numpy(float), sub["h"].to_numpy(float),
                                 sub["lo"].to_numpy(float), side=side, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 417, side: str = "top",
                   **detect_kw) -> dict:
    """Pool every confirmed island across the panel and arbitrate the forward edge.

    For each name: detect island tops (``side="top"``, a short) or island bottoms
    (``side="bottom"``, a long), take the forward-``horizon`` return after each sealing gap in the
    trade's intended direction (1-day lag), and subtract the name's own (direction-matched) base
    rate (``excess=True``) so the test is "does the figure beat buy-and-hold *for that name*", not
    "is the market up/down". Pools the per-event excess across names; reports n, mean, win-rate,
    one-sample t, HAC t, a same-tape placebo p, and the net-of-cost mean (one round trip =
    2 * cost_bps one-way).

    The same-tape placebo draws random entry dates (same count per name, same base-rate
    subtraction, same direction) and asks how often a random set beats the observed mean — the
    honest control for the tape's own drift. The detector reads OHLC (open/high/low) but the
    forward returns and base rate are measured on the **close** column to match the entry/exit
    convention.
    """
    o, h, lo, closes = panel["open"], panel["high"], panel["low"], panel["close"]
    if names is None:
        names = list(closes.columns)
    direction = -1 if side == "top" else 1
    pooled, raw_pooled = [], []
    n_island = 0
    placebo_means = []
    rng = np.random.default_rng(seed)
    for tk in names:
        sub = pd.DataFrame({"o": o[tk], "h": h[tk], "lo": lo[tk], "c": closes[tk]}).dropna()
        if len(sub) < horizon + 5:
            continue
        oo = sub["o"].to_numpy(float)
        hh = sub["h"].to_numpy(float)
        ll = sub["lo"].to_numpy(float)
        cc = sub["c"].to_numpy(float)
        isls = detect_islands(oo, hh, ll, side=side, **detect_kw)
        n_island += len(isls)
        sig = [d["signal_idx"] for d in isls]
        fr = forward_returns(cc, sig, horizon, lag=lag, direction=direction)
        if len(fr) == 0:
            continue
        br = base_rate(cc, horizon, direction=direction)
        raw_pooled.extend(fr.tolist())
        pooled.extend((fr - (br if excess else 0.0)).tolist())
        n = len(cc)
        hi_idx = n - horizon - lag - 1
        if hi_idx > 1:
            for _ in range(n_draws):
                si = rng.integers(1, hi_idx, size=len(fr))
                entry = si + lag
                exit_ = entry + horizon
                r = direction * (cc[exit_] / cc[entry] - 1.0)
                placebo_means.append(r.mean() - (br if excess else 0.0))
    pooled = np.asarray(pooled, dtype=float)
    raw_pooled = np.asarray(raw_pooled, dtype=float)
    if len(pooled) == 0:
        return {"horizon": horizon, "n_islands": int(n_island), "n_events": 0,
                "mean": float("nan"), "raw_mean": float("nan"), "win": float("nan"),
                "t": float("nan"), "hac_t": float("nan"), "p_placebo": float("nan"),
                "net": float("nan")}
    obs = float(pooled.mean())
    cost = 2.0 * cost_bps / 1e4
    p = (float(np.mean(np.asarray(placebo_means) >= obs))
         if placebo_means else float("nan"))
    return {
        "horizon": horizon, "n_islands": int(n_island), "n_events": int(len(pooled)),
        "mean": obs, "raw_mean": float(raw_pooled.mean()),
        "win": float((pooled > 0).mean()),
        "t": one_sample_t(pooled), "hac_t": hac_t(pooled),
        "p_placebo": p, "net": float(obs - cost),
    }
