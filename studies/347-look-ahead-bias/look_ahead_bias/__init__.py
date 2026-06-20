"""Study 347 — Look-Ahead-Bias: how much free alpha you earn by accidentally
peeking one bar into the future.

A research-method demonstration, not a tradable signal. We take ONE ordinary
signal (a z-score mean-reversion rule on daily closes) and run it under three
execution conventions on the *same* tape:

- **peek** — same-bar execution: the signal computed *at* the close of ``t`` is
  used to earn the return *of* ``t`` (the bar that produced the signal). This is
  the canonical look-ahead bug: you trade on information you only had after the
  bar finished.
- **lag1** — the desk's one-execution-lag rule: signal known at the close of
  ``t`` earns the return of ``t+1`` (one ``shift``, applied once).
- **lag0_close** — a subtler variant: signal from the close, executed at the
  *same* close. Identical to ``peek`` for a close-to-close rule, kept so the
  decomposition is explicit.

The point: on a tape with **no real edge** (a random walk, our null) the lagged
strategy earns ~0 Sharpe while the peeking strategy manufactures a large,
HAC-significant Sharpe out of thin air — pure bias. On a tape with a real,
*lagged-tradable* edge (our positive control) the peek still inflates the
already-real number. The inflation is the look-ahead tax, quantified.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
