"""Strategy + inference for Study 351 — BTC 5-minute Polymarket momentum.

The viral pitch: with ~2 minutes left in a Polymarket BTC Up/Down 5-minute window, if BTC
has already moved >= $D, buy the favoured side (quoted 0.80-0.99) "because it's basically
decided." Turn $300 into $14,000.

The maths of a binary share bought at price ``p`` that wins with probability ``w``:

    EV per share        = w*(1-p) - (1-w)*p = w - p
    EV per $1 staked     = (w - p) / p           (you stake p to win 1)

So the *only* question is whether the continuation win-rate ``w`` beats the price ``p`` you
pay. ``w`` we measure from the tape (:func:`continuation`). ``p`` is the market's own estimate
of ``w`` — on an efficient book ``p ~= w`` and the edge is ~0; the favourite-longshot tax and
spread push it negative. This module measures ``w``, the theoretical fair price, and runs the
bankroll Monte-Carlo that turns the "$300 -> $14,000" story into a probability.
"""

from __future__ import annotations

import math

import numpy as np

# snapshot column indices in a windows array [open, px@2left, px@1left, close]
SNAP_2MIN, SNAP_1MIN = 1, 2


def continuation(windows: np.ndarray, threshold: float = 70.0,
                 snap: int = SNAP_2MIN) -> dict:
    """Continuation stats for windows that have already moved >= ``threshold`` at ``snap``.

    Returns ``w`` (P[close on the move's side]), ``n`` (signals), ``frac`` (share of windows),
    and the binomial ``z`` of ``w`` against a coin (0.5) — the Signal-axis test.
    """
    o, snap_px, cf = windows[:, 0], windows[:, snap], windows[:, 3]
    move = snap_px - o
    final = cf - o
    fire = np.abs(move) >= threshold
    decided = final != 0
    use = fire & decided
    n = int(use.sum())
    if n == 0:
        return {"w": float("nan"), "n": 0, "frac": 0.0, "z": float("nan")}
    win = np.sign(final[use]) == np.sign(move[use])
    w = float(win.mean())
    z = (w - 0.5) / math.sqrt(0.25 / n)       # binomial test vs a coin flip
    return {"w": w, "n": n, "frac": float(fire.mean()), "z": float(z)}


def sweep(windows: np.ndarray, thresholds, snap: int = SNAP_2MIN) -> list[dict]:
    """Continuation across a grid of $ thresholds (each dict gains ``threshold``)."""
    out = []
    for d in thresholds:
        r = continuation(windows, threshold=d, snap=snap)
        r["threshold"] = d
        out.append(r)
    return out


def ev_per_dollar(w: float, p: float) -> float:
    """EV per $1 staked buying a binary share at price ``p`` that wins with prob ``w``."""
    return (w - p) / p


def _phi(z):
    """Standard-normal CDF (vectorised, stdlib erf — no scipy)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


def theoretical_continuation(threshold: float, sigma_per_min: float,
                             minutes_left: float = 2.0) -> float:
    """Continuation prob *at the boundary* for a driftless arithmetic-Brownian price:
    sitting exactly ``threshold`` ahead, P[stay on that side] = Phi(threshold/(sigma*sqrt(k)))."""
    return float(_phi(threshold / (sigma_per_min * math.sqrt(minutes_left))))


def expected_continuation(threshold: float, sigma_per_min: float,
                          elapsed: float = 3.0, left: float = 2.0,
                          grid: int = 40_001, span: float = 9.0) -> float:
    """Exact continuation a faithful engine should measure on the synthetic tape.

    The snapshot move ``X ~ N(0, sigma*sqrt(elapsed))``; conditional on ``X = x`` the side
    holds with prob ``Phi(|x|/(sigma*sqrt(left)))``. We measure on windows with ``|X| >=
    threshold``, so the right reference is ``E[Phi(|X|/(sigma*sqrt(left))) | |X| >= threshold]``
    — the conditional expectation, not the boundary value. Deterministic quadrature."""
    s_snap = sigma_per_min * math.sqrt(elapsed)
    s_left = sigma_per_min * math.sqrt(left)
    x = np.linspace(-span * s_snap, span * s_snap, grid)
    pdf = np.exp(-0.5 * (x / s_snap) ** 2)
    cont = _phi(np.abs(x) / s_left)
    mask = np.abs(x) >= threshold
    return float(np.sum(pdf[mask] * cont[mask]) / np.sum(pdf[mask]))


def monte_carlo(w: float, p: float, frac: float = 0.50, start: float = 300.0,
                target: float = 14_000.0, n_paths: int = 200_000,
                max_trades: int = 400, seed: int = 351) -> dict:
    """Bankroll simulation of the viral story.

    Each round stake ``frac`` of the bankroll on a favourite that wins with prob ``w`` at
    price ``p`` (payout ``(1-p)/p`` per $1 on a win, -1 on a loss). Returns the probability
    of reaching ``target``, of ruin (< $1), and the median/mean terminal bankroll.
    """
    rng = np.random.default_rng(seed)
    bank = np.full(n_paths, start)
    hit = np.zeros(n_paths, dtype=bool)
    win_mult = (1.0 - p) / p
    for _ in range(max_trades):
        act = (bank > 0) & ~hit
        if not act.any():
            break
        stake = bank[act] * frac
        wins = rng.random(act.sum()) < w
        bank[act] += np.where(wins, stake * win_mult, -stake)
        bank[bank < 1.0] = 0.0
        hit |= bank >= target
    return {
        "p_target": float(hit.mean()),
        "p_ruin": float((bank == 0).mean()),
        "median": float(np.median(bank)),
        "mean": float(bank.mean()),
        "edge": w - p,
    }


def live_buckets(rows: list[dict]) -> dict:
    """Summarise the live capture: realised win-rate ``w``, avg ask ``p`` and edge per bucket."""
    def stat(sel):
        s = [r for r in rows if sel(r)]
        if not s:
            return {"n": 0}
        p = np.array([r["favored_ask"] for r in s])
        win = np.array([r["favored_win"] for r in s], dtype=float)
        return {"n": len(s), "w": float(win.mean()), "p": float(p.mean()),
                "edge": float(win.mean() - p.mean()),
                "ev": float(np.mean((win - p) / p))}
    return {
        "all": stat(lambda r: True),
        "signal": stat(lambda r: r["signal70"]),
        "strong": stat(lambda r: r["abs_move"] >= 100),
    }
