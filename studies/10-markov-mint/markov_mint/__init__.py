"""Markov-Mint (Study 10) — "use Markov chains to win every single trade", tested honestly.

The claim comes from a viral prediction-market post (Alex / @de1lymoon, *How To Use Markov
Chains To Win Every Single Trade*): discretise a Polymarket contract's price into states,
build a transition matrix, Monte-Carlo the resolution, calibrate against a favorite-longshot
table, size with Kelly, execute as a maker — and harvest a repeatable edge. This study ports
the author's five-step pipeline verbatim and runs it on markets whose ground truth we control.

The punchline, established offline: on a market whose price is the fair posterior (a
martingale — the textbook efficient market), the Monte-Carlo "edge" is pure finite-sample
noise that shrinks as history grows, and the calibration step makes the whole Markov stage
*inert* — deleting it barely moves a single trade. What remains is a one-line bet on the
favorite-longshot bias, an effect that lives in the tails and dies to a normal bid/ask.

Modules:
  * :mod:`markov_mint.data`       — synthetic binary markets: an efficient (martingale) null
                                     and a *planted favorite-longshot wedge*, with realized
                                     outcomes so every trade is scored against ground truth.
  * :mod:`markov_mint.markov`     — the article's machine reimplemented faithfully (transition
                                     matrix, Monte-Carlo, calibration table, Kelly), with one
                                     switch to ablate the Markov stage.
  * :mod:`markov_mint.robustness` — the falsification battery: the headline HAC test, the
                                     noise-vs-history scaling, the inertness check, the costed
                                     P&L, and the planted-edge recovery.
"""

from . import data, markov, robustness

__all__ = ["data", "markov", "robustness"]
