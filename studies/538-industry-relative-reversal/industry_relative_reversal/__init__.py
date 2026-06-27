"""Study 538 -- Industry-Relative-Reversal.

Hameed-Mian (2015) / Da-Liu-Schaumburg (2014): the one-month return reversal is
much stronger when the formation return is measured **industry-relative** (each
stock's own one-month return minus its industry's mean). The *within-industry*
component reverses; the *across-industry* component (the industry's own move) does
not. This study builds an industry-adjusted one-month reversal and contrasts it
head-to-head with the raw one-month reversal of
[Study 329](../../329-one-month-reversal/).

Public surface:

- ``data``     -- the fixed survivor basket, its GICS sector map, the cache-first
                  monthly panel loader, and a deterministic synthetic panel with a
                  tunable *within-industry* reversal knob.
- ``strategy`` -- raw and industry-relative formation signals, the long-short
                  quintile engine, HAC inference, a placebo (industry-label
                  shuffle) null, turnover/cost accounting, and the synthetic
                  positive control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
