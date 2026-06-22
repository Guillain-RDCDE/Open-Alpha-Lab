"""Study 353 -- Smart-Money-Concepts (ICT order blocks & fair-value gaps).

Tests the viral "ICT / Smart Money Concepts" YouTube meme on real SPY daily data: do
fair-value gaps (3-bar voids) fill more / faster than random same-size zones, and do forward
returns from "order block" zones beat returns from random levels? The decisive control is a
matched random walk -- which **also** produces gaps and bounces -- so the Signal axis is the
*excess* over that null, never the raw fill-rate.

See :mod:`smart_money_concepts.data` (real SPY tape loader + deterministic random-walk control)
and :mod:`smart_money_concepts.strategy` (FVG/OB detection, fill-rate & forward-return
inference vs the null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
