"""Study 49 — Black-Gold: does the price of oil predict the stock market?

Driesprong, Jacobsen & Maat (2008) found that oil-price changes forecast next-month equity returns —
oil up, stocks down, with a delay. We test it on every calendar month of tradable oil-futures data
(2000–2026, a verified hole-free monthly grid built from daily closes) and find nothing: the predictive
slope is exactly zero (t = −0.03), and a timing rule built on it — T-bill credited in cash — still
trails buy-and-hold. A cross-asset predictability claim that doesn't survive into the era you'd trade it.
"""

from . import data, strategy  # noqa: F401
