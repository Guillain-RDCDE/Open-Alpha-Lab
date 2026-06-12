"""The article's machine, reimplemented faithfully — *their* code, run on *our* data.

This is a deliberate, line-for-line port of the five-step pipeline in the viral post
("How To Use Markov Chains To Win Every Single Trade"): discretise the price into states,
count transitions, Monte-Carlo the resolution, calibrate against a favorite-longshot table,
size with fractional Kelly. We keep the author's choices verbatim — including the ones we
think are mistakes — so the teardown can never be accused of straw-manning. Where we changed
anything it is **only** for speed (the Monte-Carlo is vectorised but samples the exact same
categorical transitions), or to expose a knob for the ablation (``ablate_markov``), never to
alter the method's output.

The one thing worth flagging up front, because it drives the whole result: the author's
``monte_carlo`` calls a path "YES" when the *simulated price* ends in the **upper half of the
state grid** (``state >= n_states // 2``), not when the *event* resolves YES. Price-above-50¢
and event-resolves-YES are different objects; conflating them is the seed of the artefact the
teardown measures. We preserve it exactly.
"""

from __future__ import annotations

import numpy as np

# The article's calibration table — naive price level -> claimed real-world resolution rate,
# attributed to "Becker's 72.1 M-trade analysis". Reproduced verbatim; its provenance is
# discussed in docs/references.md. The favorite-longshot shape (cheap < fair, dear > fair) is
# the only economic content in the whole pipeline.
CALIBRATION = {
    0.01: 0.0043, 0.05: 0.0418, 0.10: 0.087, 0.20: 0.181,
    0.30: 0.285, 0.50: 0.500, 0.70: 0.715, 0.80: 0.819,
    0.90: 0.913, 0.95: 0.958,
}


def build_transition_matrix(prices, n_states: int = 10) -> np.ndarray:
    """Discretise a price history into ``n_states`` buckets and count normalised transitions.

    A faithful port: bucket each price into ``[0, n_states)``, tally moves from state i to
    state j, normalise each row to sum to 1 (empty rows left as zeros, guarded at sample
    time). With ~60 daily prices this fills a 10x10 (100-cell) matrix from <60 transitions —
    the small-sample problem the article warns about and then ignores.
    """
    states = np.clip((np.asarray(prices) * n_states).astype(int), 0, n_states - 1)
    T = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        T[states[i], states[i + 1]] += 1.0
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def monte_carlo(T: np.ndarray, start_state: int, days: int = 30,
                n_sims: int = 10_000, rng: np.random.Generator | None = None) -> float:
    """Random-walk ``n_sims`` paths through ``T`` for ``days`` steps; return the YES fraction.

    ``n_sims`` defaults to 10,000 — the article's own path count, kept so the port stays
    verbatim. Vectorised but semantically identical to the article's per-path loop: at each step every
    live path samples its next state from its current row of ``T`` (inverse-CDF sampling). A
    path counts as YES if it ends in the **upper half of the grid** — the author's definition,
    preserved. Empty rows (states never observed) are made absorbing, the natural limit of the
    author's row that "sums to zero".
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(T)
    cdf = T.cumsum(axis=1)
    # An all-zero row (state never visited) has cdf all-zero -> make it absorbing.
    dead = cdf[:, -1] == 0
    cdf = cdf.copy()
    cdf[dead] = 1.0  # forces searchsorted to return the state itself (stays put)
    diag_fix = np.where(dead)[0]
    state = np.full(n_sims, start_state, dtype=int)
    for _ in range(days):
        u = rng.random(n_sims)
        nxt = (u[:, None] < cdf[state]).argmax(axis=1)
        # absorbing dead states: keep the same state
        stuck = np.isin(state, diag_fix)
        nxt[stuck] = state[stuck]
        state = nxt
    return float((state >= n // 2).mean())


def calibrate(raw_prob: float) -> float:
    """Linear-interpolate ``raw_prob`` through the article's favorite-longshot table."""
    keys = sorted(CALIBRATION)
    if raw_prob <= keys[0]:
        return CALIBRATION[keys[0]]
    if raw_prob >= keys[-1]:
        return CALIBRATION[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= raw_prob <= hi:
            frac = (raw_prob - lo) / (hi - lo)
            return CALIBRATION[lo] + frac * (CALIBRATION[hi] - CALIBRATION[lo])
    return raw_prob


def kelly(p_win: float, price_cents: float, mult: float = 0.25) -> float:
    """Fractional-Kelly stake for a binary contract at ``price_cents`` with win prob ``p_win``."""
    cost = price_cents / 100.0
    if cost >= 1.0 or cost <= 0.0:
        return 0.0
    b = (1.0 - cost) / cost
    f = (b * p_win - (1.0 - p_win)) / b
    return max(0.0, f) * mult


class MarkovMintSystem:
    """The full five-step pipeline as a single object, with one extra switch for the ablation.

    ``analyze`` returns the decision dict the article's ``analyze`` returns, plus the raw and
    calibrated probabilities and the raw/system edges, so the robustness layer can dissect
    where each number comes from. Set ``ablate_markov=True`` to *replace the entire Markov +
    Monte-Carlo stage with the price itself* (``raw_prob := current_price``): if the
    downstream decisions barely move, the Markov apparatus was inert.
    """

    def __init__(self, n_states: int = 10, n_sims: int = 10_000, edge_threshold: float = 0.03,
                 kelly_mult: float = 0.25):
        self.n_states = n_states
        self.n_sims = n_sims
        self.edge_threshold = edge_threshold
        self.kelly_mult = kelly_mult

    def analyze(self, prices, current_price: float, days_to_expiry: int,
                rng: np.random.Generator | None = None, ablate_markov: bool = False) -> dict:
        if ablate_markov:
            raw_prob = float(current_price)
        else:
            T = build_transition_matrix(prices, self.n_states)
            start = min(int(current_price * self.n_states), self.n_states - 1)
            raw_prob = monte_carlo(T, start, days=days_to_expiry, n_sims=self.n_sims, rng=rng)

        cal_prob = calibrate(raw_prob)
        edge = cal_prob - current_price          # the "system edge" the article trades
        raw_edge = raw_prob - current_price       # what the Monte-Carlo alone produced

        if abs(edge) < self.edge_threshold:
            direction = "PASS"
            f = 0.0
        else:
            direction = "BUY YES" if edge > 0 else "BUY NO"
            p_dir = cal_prob if edge > 0 else 1.0 - cal_prob
            f = kelly(p_dir, current_price * 100.0, self.kelly_mult)

        return {
            "raw_prob": raw_prob,
            "cal_prob": cal_prob,
            "market_price": float(current_price),
            "raw_edge": raw_edge,
            "system_edge": edge,
            "direction": direction,
            "kelly_frac": f,
        }
