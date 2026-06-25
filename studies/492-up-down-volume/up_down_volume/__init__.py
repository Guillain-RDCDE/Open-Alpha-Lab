"""Study 492 — Up-Down-Volume (market-breadth up/down volume ratio).

A mechanical, falsifiable encoding of the up/down-volume-ratio folklore (the volume side of
the Arms index / TRIN; the "selling climax" of classic tape-reading). We aggregate daily
up-volume vs down-volume across a basket of liquid sector ETFs (a proxy for exchange-wide
advance/decline volume), fire a long on a down-volume *selling climax* (the ratio at its rolling
lower tail), enter at the next close, and measure forward SPY returns against a drift-matched
random-entry baseline and a shuffled-volume placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
