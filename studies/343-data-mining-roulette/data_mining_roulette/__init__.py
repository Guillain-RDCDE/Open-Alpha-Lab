"""Study 343 — Data-Mining-Roulette: the p-hacking machine.

Generate a large family of *random* trading rules on a tape, look at the best one,
and ask the only question that matters: **how good does the luckiest rule look when
there is, by construction, nothing to find?**

The headline is a methodology verdict, not a tradable edge. On a pure-noise tape, the
single best of N=1,000 random rules routinely posts a backtest Sharpe and a naive
*t*-stat that any blog would call "real" — until you correct for the fact that you ran
a thousand of them. The fix (White's Reality Check / Bonferroni on the *maximum*
statistic, not each rule's own) collapses the mirage. We confirm it both ways: a
positive control (a planted rule the harness *must* find) and a null (where the only
thing it finds is luck dressed as skill).

Distinct from Study 350 (Dartboard-Portfolio), which randomises *which stocks* you hold;
here we randomise the *rule* — the entry/exit logic itself — and study the distribution
of the best backtest under the null. The roulette wheel is the strategy space.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
