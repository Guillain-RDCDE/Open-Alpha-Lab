"""Study 341 — MLP-Pipelines: the AMLP *fat-yield-vs-energy-beta* teardown.

Pipeline MLPs (and their wrapper ETF **AMLP**, the Alerian MLP ETF) advertise a 7–8%
distribution yield as a bond-like "income" sleeve. This study takes that apart on two axes:

1. **The income illusion** — distribution yield decomposed against NAV erosion (how much of the
   "yield" is *return of capital* handed back, vs genuine yield on a flat-or-rising NAV) and the
   total-return race against the broad market (SPY) and a pure-energy benchmark (XLE).
2. **The energy beta** — AMLP is sold as a yield play, but it is really a leveraged bet on the
   oil/energy complex. We measure its beta to crude/energy with a HAC *t*, and show it crashed
   with oil in 2014–16 and 2020.

Distinct from [Study 337 — Covered-Call-ETF](../../337-covered-call-etf/) (an *option-overlay*
income illusion) and [Study 57 — Yield-Trap](../../57-yield-trap/) (high-dividend *stocks*): this
is the *midstream/MLP asset-class* identity test, with **energy beta** as the third axis.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
