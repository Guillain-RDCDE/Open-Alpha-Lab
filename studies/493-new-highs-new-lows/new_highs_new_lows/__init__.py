"""Study 493 — New-Highs-New-Lows (NH-NL breadth line).

A mechanical, falsifiable encoding of the classic new-highs/new-lows breadth indicator:
count how many basket members print a fresh 52-week high, subtract those at a fresh 52-week
low, smooth the *net* fraction, and fire a long on a breadth thrust (up-cross of a positive
threshold). The folklore says **breadth leads price** — the NH-NL line forecasts the index.
We test that as a forward-return study against random-entry and shuffled-membership placebos,
with costs. Breadth is proxied by a small basket of liquid ETFs (a coarse proxy that caps the
test — see docs).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
