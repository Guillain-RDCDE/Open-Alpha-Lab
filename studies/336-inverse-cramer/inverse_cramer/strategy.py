"""The Inverse-Cramer engine and its honest controls — Study 336.

The meme: a pundit (Jim Cramer) is *so* often wrong that simply doing the OPPOSITE of his
on-air calls is an edge. The Inverse Cramer ETF (SJIM, 2023-2024) was a real, if short-lived,
product built on exactly this premise. The testable hypothesis: over a forward window after a
call, the **fade** (trade opposite the stated direction) earns a positive return that beats a
random-direction control on the *same* calls.

We implement it as a per-call forward-return ledger:

- Each call has a stated ``direction`` (+1 bull / -1 bear). The **fade position** is
  ``-direction``. We hold it for a fixed ``fwd`` trading days.
- ``fwd_ret`` is the call's underlying forward return; the fade's gross return is
  ``-direction * fwd_ret`` (minus one round-trip cost).
- The control arm trades a **random** direction on the *same* calls — if the fade knows
  something, it beats the coin.

Execution lag: a call is known *after the show airs*, so the fade is entered at the **next
session's open-equivalent** (we lag the entry by one bar — ``entry_lag=1`` — and measure the
forward return from that bar). One shift, applied once, documented here.

The third axis is **selection bias**: the call list is a *curated* sample of memorable
moments. We never claim the curated edge is real; the synthetic ``selection_bias`` control
shows how a coin-flip pundit produces a "fade edge" once you publish only his worst calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Forward returns from real price tapes
# ---------------------------------------------------------------------------
def forward_return(
    close: pd.Series,
    call_date: pd.Timestamp,
    fwd: int = 21,
    entry_lag: int = 1,
) -> float | None:
    """Forward return over ``fwd`` sessions starting ``entry_lag`` bars after ``call_date``.

    The call is known after the show airs, so we enter at the bar ``entry_lag`` sessions
    after the on-air date (one shift, applied once) and measure close-to-close over the next
    ``fwd`` sessions. Returns ``None`` if the tape does not cover the window.
    """
    idx = close.index
    pos = idx.searchsorted(pd.Timestamp(call_date), side="left")
    entry = pos + entry_lag
    exit_ = entry + fwd
    if entry < 0 or exit_ >= len(idx):
        return None
    p0 = float(close.iloc[entry])
    p1 = float(close.iloc[exit_])
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def build_real_ledger(
    calls: pd.DataFrame,
    tapes: dict[str, pd.Series],
    fwd: int = 21,
    entry_lag: int = 1,
) -> pd.DataFrame:
    """Per-call forward returns from real tapes; rows without coverage are dropped.

    ``calls`` must carry ``ticker``, ``date``, ``direction``. Returns a ledger with
    ``ticker, date, direction, fwd_ret`` for every call whose tape covers the window.
    """
    rows = []
    for _, c in calls.iterrows():
        t = c["ticker"]
        if t not in tapes:
            continue
        fr = forward_return(tapes[t], c["date"], fwd=fwd, entry_lag=entry_lag)
        if fr is None or not np.isfinite(fr):
            continue
        rows.append(
            {"ticker": t, "date": c["date"], "direction": int(c["direction"]), "fwd_ret": fr}
        )
    return pd.DataFrame(rows, columns=["ticker", "date", "direction", "fwd_ret"])


# ---------------------------------------------------------------------------
# The fade engine
# ---------------------------------------------------------------------------
def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


def fade_returns(
    ledger: pd.DataFrame,
    cost_bps: float = 0.0,
    dirs: np.ndarray | None = None,
) -> np.ndarray:
    """Per-call net returns of the position.

    By default the position is the **fade** (``-direction``). Pass ``dirs`` (a ±1 vector,
    the random control) to override the fade direction call-by-call. ``cost_bps`` is a
    one-way cost charged on each leg of the round trip (entry + exit = 2 legs), against NAV.
    """
    d = ledger["direction"].to_numpy(dtype=float)
    fwd = ledger["fwd_ret"].to_numpy(dtype=float)
    pos = -d if dirs is None else np.asarray(dirs, dtype=float)
    gross = pos * fwd
    # One-way cost x NAV, both legs (entry + exit).
    return gross - 2.0 * cost_bps * 1e-4


def synthetic_ledger(calls: pd.DataFrame) -> pd.DataFrame:
    """Adapt a synthetic ``calls`` frame to the ledger shape used by the engine."""
    out = calls[["direction", "fwd_ret"]].copy()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _hac_t(r: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean (no hard quantlab dependency)."""
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    r: np.ndarray,
    block: int = 5,
    n_boot: int = 2000,
    seed: int = 336,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean per-call return."""
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[b] = r[idx.ravel()[:n]].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def summarize(r: np.ndarray) -> dict:
    """Headline statistics for one per-call return vector.

    Returns count, win-rate, mean (bps/call), per-call Sharpe, skew, HAC t-stat, and a
    block-bootstrap 95% CI on the mean. The HAC t is the inference-bar number that decides
    whether the fade is distinguishable from zero.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return dict(n=0, win_rate=float("nan"), mean_bps=float("nan"),
                    sharpe=float("nan"), skew=float("nan"), tstat=float("nan"),
                    ci_lo=float("nan"), ci_hi=float("nan"))
    lo, hi = block_bootstrap_ci(r)
    return dict(
        n=int(n),
        win_rate=float((r > 0).mean()),
        mean_bps=float(r.mean() * 1e4),
        sharpe=float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        skew=float(pd.Series(r).skew()) if n > 2 else float("nan"),
        tstat=_hac_t(r),
        ci_lo=float(lo * 1e4),
        ci_hi=float(hi * 1e4),
    )


def excess_over_coin(ledger: pd.DataFrame, seed: int = 336, cost_bps: float = 0.0) -> dict:
    """Compare the fade to a random-direction control on the *same* calls.

    Returns the fade summary, the coin summary, and the excess mean (bps/call). A fade with
    real anti-signal beats the coin; a fade riding only beta/selection does not.
    """
    n = len(ledger)
    fade = summarize(fade_returns(ledger, cost_bps=cost_bps))
    coin_r = fade_returns(ledger, cost_bps=cost_bps, dirs=random_directions(n, seed=seed))
    coin = summarize(coin_r)
    return {
        "fade": fade,
        "coin": coin,
        "excess_bps": fade["mean_bps"] - coin["mean_bps"],
    }
