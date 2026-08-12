"""Study 885 — Ultra-Short Credit Pickup.

Ultra-short investment-grade credit ETFs (JPST / ICSH / MINT) pay a spread over
plain T-bill vehicles (BIL / SHV) for taking a *tiny* sliver of credit and
duration risk. The question: is the ultra-short credit sleeve a genuine, near-
riskless STRUCTURAL pickup — a higher excess-of-bills Sharpe than bills, with only
marginally more drawdown — or does the 2020/2022 stress (and real costs) eat it?

Public API re-exported for the notebooks / examples / tests.
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
