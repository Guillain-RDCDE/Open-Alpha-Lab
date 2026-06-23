"""Study 401 — Signal-Stacking: does combining weak signals manufacture an edge?

The viral pitch: take K signals "none of which would pass a significance test alone, most
hovering around 52%", normalise them into one composite z-score "weighted by historical
Sharpe", and the combination "produces something retail has never seen." This package
builds that machine and tests the claim against its own controls.

The honest answer has a real kernel and a fatal catch. Stacking weak signals *does* raise
the information coefficient — but only like **√K**, and **only when the signals carry
genuine, decorrelated edge**. On a pure-noise stack (the null), the composite times nothing:
its Sharpe sits squarely in the permutation null. With redundant signals, the √K boost
collapses to a low ceiling — fifty copies of one idea are worth barely more than one. And
the gorgeous "ζ-field vs buy-and-hold" equity curve the threads show is manufactured by
*selecting* the signals and their Sharpe-weights in-sample; it decays the moment it leaves
the data it was mined from.

A positive control (signals with a planted, decorrelated edge the stacker *must* compound)
proves the harness banks a real blend; the null and the snooped split show that, absent real
decorrelated information, the "composite field" is luck dressed in Greek letters.

Sibling of the desk's research-method demos (Studies 343–350): where Study 343 randomises
the *rule*, this study takes the *combination* claim head-on — the maths of why fifty weak
signals are powerful only when they are both real and different.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
