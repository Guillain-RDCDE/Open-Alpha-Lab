"""Study 49 — Black-Gold: does the price of oil predict the stock market?

Driesprong, Jacobsen & Maat (2008) found that oil-price changes forecast next-month equity returns —
oil up, stocks down, with a delay. We test it on every month of tradable oil data (2000–2026) and find
nothing: the relationship is insignificant, the sign is *wrong*, and a timing rule built on it badly
trails buy-and-hold. A cross-asset predictability claim that doesn't survive into the era you'd trade it.
"""

from . import data, strategy  # noqa: F401
