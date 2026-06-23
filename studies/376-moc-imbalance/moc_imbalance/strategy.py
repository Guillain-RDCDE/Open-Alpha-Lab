"""Strategy + inference for Study 376 — MOC-Imbalance (closing pressure & overnight reversal).

The believers' claim, operationalised on a transparent proxy:

    Let ``disp`` = the signed intraday displacement of the close inside the day's range
    (+1 = closed at the high after opening at the low = maximum upward push into the close).
    A closing-auction imbalance is said to **push the last print** and then **reverse
    overnight** — so a *large positive* ``disp`` should be followed by a *negative* overnight
    gap (close→open), and vice versa.

We test it three ways, all on the overnight gap that follows the push:

  * a **reversal regression** ``overnight ~ a + b·disp`` — a robustly *negative* ``b`` is the
    reversal the lore predicts; we report ``b`` with a **HAC (Newey-West) t-stat**;
  * a **placebo / randomization null** — shuffle the displacement-to-gap pairing many times
    and ask how often a random pairing produces a slope at least as negative (the honest
    small-effect test);
  * a **fade strategy** — take a position −sign(disp) into the close (fade the push), earn the
    overnight gap; report the per-trade mean, the win-rate vs the 50% base rate, a **1-day
    execution lag** baked into the panel (entry is *today's* close, the gap is *next* open),
    and **one-way costs** on the round-trip.

We run the whole thing on the full tape **and** on the **index-rebalance subset** (third
Friday of quarter-end months), where real MOC imbalances are largest — the believers'
strongest case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Core regression + HAC inference
# --------------------------------------------------------------------------- #
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Univariate OLS y = a + b·x. Returns (a, b, residuals)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(beta[0]), float(beta[1]), resid


def newey_west_t(x: np.ndarray, y: np.ndarray, lags: int = 5) -> dict:
    """Slope ``b`` of ``y ~ a + b·x`` with a Newey-West (HAC) standard error.

    HAC is the desk's Signal-axis bar: it is robust to the serial correlation and
    heteroskedasticity that overnight gaps carry. Returns the slope, its HAC SE, the HAC
    t-stat, the intercept and n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5:
        return {"b": float("nan"), "se": float("nan"), "t": float("nan"),
                "a": float("nan"), "n": int(n)}
    a, b, resid = _ols(x, y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    # HAC meat: S = sum_j w_j (Gamma_j + Gamma_j'), Bartlett weights
    u = resid[:, None] * X                      # n x 2 score
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = float(np.sqrt(cov[1, 1]))
    t = b / se_b if se_b > 0 else float("nan")
    return {"b": b, "se": se_b, "t": float(t), "a": a, "n": int(n)}


def placebo_pvalue(disp: np.ndarray, overnight: np.ndarray, n_draws: int = 20_000,
                   seed: int = 376) -> dict:
    """Randomization null for the reversal slope.

    Shuffle the pairing between today's displacement and the *next* overnight gap many times;
    a true reversal makes the observed slope **more negative** than chance. We report
    ``p = P[random-pairing slope <= observed slope]`` — small p ⇒ a reversal a coin could not
    have produced.
    """
    disp = np.asarray(disp, dtype=float)
    overnight = np.asarray(overnight, dtype=float)
    n = len(disp)
    if n < 5:
        return {"obs_b": float("nan"), "placebo_b": float("nan"), "p_value": float("nan")}
    _, obs_b, _ = _ols(disp, overnight)
    rng = np.random.default_rng(seed)
    # centre x once; slope of shuffled y on fixed x is (x_c·y_perm)/sum(x_c^2)
    xc = disp - disp.mean()
    denom = float((xc * xc).sum())
    slopes = np.empty(n_draws)
    for i in range(n_draws):
        yp = rng.permutation(overnight)
        slopes[i] = float((xc * yp).sum()) / denom
    p = float((slopes <= obs_b).mean())
    return {"obs_b": float(obs_b), "placebo_b": float(slopes.mean()), "p_value": p}


# --------------------------------------------------------------------------- #
# Fade strategy (overnight reversal) + costs
# --------------------------------------------------------------------------- #
def fade_returns(panel: pd.DataFrame) -> np.ndarray:
    """Per-day return of fading the close push: position −sign(disp) into the close, earn the
    next overnight gap. The 1-day execution lag is structural — entry is today's close, the
    gap is realised at the *next* open.
    """
    disp = panel["disp"].to_numpy()
    overnight = panel["overnight"].to_numpy()
    pos = -np.sign(disp)
    return pos * overnight


def summarize(panel: pd.DataFrame, lags: int = 5) -> dict:
    """Headline stats: HAC reversal slope, placebo p, fade-trade mean/win-rate vs 50%."""
    disp = panel["disp"].to_numpy()
    overnight = panel["overnight"].to_numpy()
    hac = newey_west_t(disp, overnight, lags=lags)
    pl = placebo_pvalue(disp, overnight)
    fr = fade_returns(panel)
    nonzero = fr[np.sign(disp) != 0]
    return {
        "n": int(len(panel)),
        "b": hac["b"],
        "t": hac["t"],
        "se": hac["se"],
        "p_placebo": pl["p_value"],
        "fade_mean": float(fr.mean()) if len(fr) else float("nan"),
        "fade_win": float((nonzero > 0).mean()) if len(nonzero) else float("nan"),
        "corr": float(np.corrcoef(disp, overnight)[0, 1]) if len(disp) > 2 else float("nan"),
    }


def net_of_costs(panel: pd.DataFrame, cost_bps: float = 2.0) -> dict:
    """Fade strategy net of a one-way overnight round-trip cost (close in, open out).

    ``cost_bps`` is charged per round-trip (a fade is one entry at the close and one exit at
    the open). Returns gross/net mean per trade and the implied annualised contribution if
    the fade were run on every day (purely illustrative; capacity is the real constraint).
    """
    fr = fade_returns(panel)
    c = cost_bps / 1e4
    net = fr - c
    ann = (1.0 + net.mean()) ** 252 - 1.0 if len(net) else float("nan")
    return {
        "n_trades": int(len(fr)),
        "gross_mean": float(fr.mean()) if len(fr) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "net_ann": float(ann),
        "cost_bps": cost_bps,
    }


def subset(panel: pd.DataFrame, rebal_only: bool = False) -> pd.DataFrame:
    """Optionally restrict to index-rebalance proxy days (largest real MOC imbalances)."""
    if rebal_only:
        return panel[panel["rebal"]].reset_index(drop=True)
    return panel
