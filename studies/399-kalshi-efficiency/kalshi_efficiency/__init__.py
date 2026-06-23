"""Study 399 — Kalshi-Efficiency (are event-contract prices well-calibrated?).

Kalshi is a CFTC-regulated event-contract exchange: each market is a binary that pays $1
if an event resolves YES and $0 otherwise, so its price *is* a probability in cents. The
believer's pitch: if the crowd is systematically over- or under-pricing some bucket of
contracts, there is a free edge in the mispricing — buy the cheap longshots (or the rich
favourites) and harvest the gap between price and frequency.

Kalshi's full resolved-contract history is **not free**, so the offline core of this study is
a **transparent, clearly-labelled illustrative event-contract dataset** — a *methods demo*,
not a real backtest. We build a synthetic book of resolved binaries whose prices are
well-calibrated *except* for a controllable **favourite–longshot bias** (the classic racetrack
distortion: longshots over-bet, favourites under-bet). The machinery — reliability/calibration
curve, Brier-score decomposition, a longshot-vs-favourite return spread with a t-stat, a
randomization null, one execution lag and one-way fees — is exactly the desk's shared engine.

The lesson is statistical, not directional: a calibration curve drawn from a finite book of
binaries *will* wiggle off the diagonal by luck, and even a real favourite–longshot tilt is a
few cents that an exchange's fee and the bid/ask eat before you can harvest it. (Same
"high-confidence number, no harvestable edge" shape as Study 351 Polymarket and the
research-method demos 343–350.)

See :mod:`kalshi_efficiency.data` (illustrative real-tape loader + deterministic synthetic
control with a planted favourite–longshot edge knob) and :mod:`kalshi_efficiency.strategy`
(calibration curve, Brier decomposition, longshot/favourite return spread, t / randomization
null, fees)."""

from . import data, strategy

__all__ = ["data", "strategy"]
