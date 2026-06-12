"""Study 62 — Premium-Seller: does a covered-call "income" ETF beat just owning the index?

Covered-call funds (QYLD) sell calls on their holdings to harvest option premium — sold as superior,
income-rich equity exposure. They aren't: QYLD trailed its OWN underlying (QQQ) by ~11%/yr at a lower
Sharpe, because writing calls caps the upside far more than the premium cushions the downside. The
"income" is an accounting illusion paid for with your gains.
"""
from . import data, strategy  # noqa: F401
