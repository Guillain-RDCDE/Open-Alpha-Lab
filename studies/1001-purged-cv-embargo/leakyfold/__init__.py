"""Study 1001 — The Leaky Fold.

Cross-validation is the default way to estimate how well a model will do on data it
has not seen. On time series it does not work, and the reason is not subtle: shuffling the folds
puts tomorrow in the training set and today in the test set. Add the standard practice of
labelling each observation with a *forward* return — which makes neighbouring labels overlap —
and the test set is largely contained in the training set. This study measures the size of the
resulting illusion, implements the fix (López de Prado's purging and embargoing), and checks
that the fix does not simply destroy everything including real signal.

- :mod:`leakyfold.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`leakyfold.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
