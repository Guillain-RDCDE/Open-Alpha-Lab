"""Signal + inference for Study 370 — has the 0DTE boom pinned the SPX tape?

The believers' claim: since same-day options went daily (2022), the SPX/SPY tape is more
**mean-reverting / more pinned** intraday. A daily feed cannot see true intraday
microstructure, so we test the claim on **labelled daily proxies**:

  * **Mean-reversion proxy** — the lag-1 autocorrelation of the daily *intraday*
    (open→close) leg. A *more* mean-reverting tape ⇒ a *more negative* lag-1 autocorr.
    We compare pre-2022 vs post-2022 and ask whether the change is real.
  * **Pinning proxy** — the same-day realised range log(high/low) and the close-to-open
    overnight share. A pinned tape should show *lower* intraday range relative to overnight.

Inference, in the desk's house spirit:
  * a **Welch t** on the difference of the two regimes' lag-1 autocorrelations
    (block-bootstrap standard error, so overlapping/serial structure is respected);
  * a **placebo / permutation null** — shuffle the regime label across many random break
    dates and ask how often a random split produces a |Δ autocorr| as large as 2022's;
  * a **win-rate** of a naive intraday mean-reversion trade vs its own base rate;
  * **one-day execution lag** + **one-way costs × turnover** on the mean-reversion trade.

The honest verdict is set by what daily data *can* establish: a small, not-significant
cross-2022 change on the proxy, on a series that by construction can't see the intraday
phenomenon — real-as-story, weak-as-measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Signal — mean-reversion (lag-1 autocorrelation) and range proxies
# --------------------------------------------------------------------------- #
def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D array (NaN if < 3 points or zero variance)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    xc = x - x.mean()
    denom = (xc * xc).sum()
    if denom == 0:
        return float("nan")
    return float((xc[:-1] * xc[1:]).sum() / denom)


def split_regimes(frame: pd.DataFrame, break_date: pd.Timestamp,
                  col: str = "intraday") -> tuple[np.ndarray, np.ndarray]:
    """Return (pre, post) arrays of ``col`` split at ``break_date``."""
    pre = frame.loc[frame.index < break_date, col].values
    post = frame.loc[frame.index >= break_date, col].values
    return np.asarray(pre, float), np.asarray(post, float)


def regime_table(frame: pd.DataFrame, break_date: pd.Timestamp,
                 col: str = "intraday") -> dict:
    """Pre/post-break summary of the mean-reversion proxy on ``col``.

    Reports each regime's lag-1 autocorr, mean |return|, annualised vol, and the Δ autocorr
    (post − pre): negative Δ means a *more mean-reverting / more pinned* tape post-2022.
    """
    pre, post = split_regimes(frame, break_date, col)
    a_pre, a_post = lag1_autocorr(pre), lag1_autocorr(post)
    return {
        "col": col,
        "n_pre": int(len(pre)), "n_post": int(len(post)),
        "ac_pre": a_pre, "ac_post": a_post, "d_ac": a_post - a_pre,
        "vol_pre": float(pre.std(ddof=1) * np.sqrt(252)),
        "vol_post": float(post.std(ddof=1) * np.sqrt(252)),
        "absret_pre": float(np.abs(pre).mean()),
        "absret_post": float(np.abs(post).mean()),
    }


# --------------------------------------------------------------------------- #
# Inference — block bootstrap Welch t + placebo permutation null
# --------------------------------------------------------------------------- #
def _block_boot_autocorr(x: np.ndarray, n_boot: int, block: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the lag-1 autocorr of ``x`` using overlapping blocks."""
    n = len(x)
    if n < block + 2:
        return np.array([lag1_autocorr(x)] * n_boot)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        out[b] = lag1_autocorr(sample)
    return out


def welch_t_autocorr(frame: pd.DataFrame, break_date: pd.Timestamp,
                     col: str = "intraday", n_boot: int = 2_000,
                     block: int = 20, seed: int = 370) -> dict:
    """Welch-style t of (post − pre) lag-1 autocorr, with block-bootstrap standard errors.

    The bootstrap SE respects the serial structure (overlapping blocks). Returns the two
    autocorrs, their difference, the bootstrap SE of each, and the t-stat of the difference.
    """
    pre, post = split_regimes(frame, break_date, col)
    rng = np.random.default_rng(seed)
    bpre = _block_boot_autocorr(pre, n_boot, block, rng)
    bpost = _block_boot_autocorr(post, n_boot, block, rng)
    a_pre, a_post = lag1_autocorr(pre), lag1_autocorr(post)
    se_pre = float(np.nanstd(bpre, ddof=1))
    se_post = float(np.nanstd(bpost, ddof=1))
    se = np.sqrt(se_pre ** 2 + se_post ** 2)
    t = float((a_post - a_pre) / se) if se > 0 else float("nan")
    return {"ac_pre": a_pre, "ac_post": a_post, "d_ac": a_post - a_pre,
            "se_pre": se_pre, "se_post": se_post, "t": t}


def placebo_break_pvalue(frame: pd.DataFrame, col: str = "intraday",
                         n_draws: int = 5_000, min_frac: float = 0.25,
                         seed: int = 370,
                         break_date: pd.Timestamp | None = None) -> dict:
    """Permutation null for the cross-break change in lag-1 autocorr.

    Pick ``n_draws`` random break dates (each leaving ≥ ``min_frac`` of the sample on both
    sides), compute |Δ autocorr| for each, and report the share whose |Δ| ≥ the |Δ| at the
    real 2022 break. A high p-value means 2022's change is unremarkable among random splits.
    """
    x = frame[col].values
    n = len(x)
    lo = int(n * min_frac)
    hi = int(n * (1 - min_frac))
    if break_date is None:
        from .data import BREAK_DATE
        break_date = BREAK_DATE
    pre, post = split_regimes(frame, break_date, col)
    obs = abs(lag1_autocorr(post) - lag1_autocorr(pre))
    rng = np.random.default_rng(seed)
    cuts = rng.integers(lo, hi, size=n_draws)
    deltas = np.empty(n_draws)
    for i, cut in enumerate(cuts):
        deltas[i] = abs(lag1_autocorr(x[cut:]) - lag1_autocorr(x[:cut]))
    p = float((deltas >= obs).mean())
    return {"obs_d_ac": obs, "placebo_mean_d": float(deltas.mean()),
            "p_value": p, "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# A naive 'trade the pin' intraday mean-reversion rule (tradability)
# --------------------------------------------------------------------------- #
def mean_reversion_trade(frame: pd.DataFrame, break_date: pd.Timestamp | None = None,
                         col: str = "intraday", cost_bps: float = 1.0,
                         lag: int = 1) -> dict:
    """Naive contrarian: position = −sign(yesterday's intraday leg), held one day.

    If the tape is mean-reverting, fading the prior intraday move should pay. We apply a
    **one-day execution lag** (signal from t−1, return from t) and a **one-way cost** of
    ``cost_bps`` per side × turnover. Reported gross/net mean daily PnL and win-rate, both
    overall and split at ``break_date`` (does the pin trade work *better* post-2022?).
    """
    r = frame[col].values
    sig = -np.sign(np.roll(r, lag))         # fade prior leg, entered with `lag` delay
    sig[:lag] = 0.0
    pnl_gross = sig * r
    turnover = np.abs(np.diff(np.concatenate([[0.0], sig])))
    cost = turnover * (cost_bps / 1e4)
    pnl_net = pnl_gross - cost

    def _stats(mask):
        g = pnl_gross[mask]; nz = pnl_net[mask]
        traded = g[sig[mask] != 0]
        return {
            "n": int(mask.sum()),
            "gross_mean": float(g.mean()) if len(g) else float("nan"),
            "net_mean": float(nz.mean()) if len(nz) else float("nan"),
            "win": float((traded > 0).mean()) if len(traded) else float("nan"),
            "sharpe_net": float(nz.mean() / nz.std(ddof=1) * np.sqrt(252))
            if nz.std(ddof=1) > 0 else float("nan"),
        }

    idx = frame.index
    out = {"all": _stats(np.ones(len(r), bool))}
    if break_date is not None:
        out["pre"] = _stats(np.asarray(idx < break_date))
        out["post"] = _stats(np.asarray(idx >= break_date))
    out["cost_bps"] = cost_bps
    return out


# --------------------------------------------------------------------------- #
# Option-chain snapshot summary (pinning context, current snapshot only)
# --------------------------------------------------------------------------- #
def chain_concentration(chain: pd.DataFrame, spot: float,
                        band: float = 0.01) -> dict:
    """How concentrated is the 0DTE chain's volume near spot? (current-snapshot proxy)

    Share of total contract *volume* whose strike sits within ``band`` (±1%) of spot — a
    crude read on how much same-day flow is clustered at-the-money (the 'pinning fuel').
    """
    v = chain["volume"].fillna(0.0)
    total = float(v.sum())
    near = chain["strike"].sub(spot).abs() <= band * spot
    near_v = float(v[near].sum())
    return {"total_volume": total, "near_volume": near_v,
            "share_near": (near_v / total) if total > 0 else float("nan"),
            "band_pct": band * 100, "spot": spot}
