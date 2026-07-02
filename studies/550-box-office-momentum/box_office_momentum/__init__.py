"""Study 550 — Box-Office-Momentum.

Weekend box-office receipts as a consumer-sentiment *leading indicator* for media/studio
stocks and the broad tape. Synthetic-only (the free retail stack cannot reach a clean,
survivorship-honest historical box-office panel), so the study is capped at WEAK/NONE.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
