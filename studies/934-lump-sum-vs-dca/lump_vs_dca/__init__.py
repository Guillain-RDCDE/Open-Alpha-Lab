"""Study 934 — Lump Sum vs DCA.

Race a $1 lump sum against $1 dripped in over twelve monthly tranches, with the
uninvested balance parked in **BIL** (real T-bill total return), over every start
month of the sample. See ``data`` for the tapes and ``strategy`` for the experiment
and its inference.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
