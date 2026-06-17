"""Study 227 — Natgas-Winter: is the winter natural-gas spike a tradable season?

UNG (United States Natural Gas Fund) exhibits a well-known winter-demand pattern: heating demand peaks
November-March, and traders have long tried to ride the seasonal spike. But UNG is a futures-based ETF
and suffers severe contango roll-yield drag — the 'widow-maker' of commodity ETFs. We test a seasonal
long (buy UNG in October, sell in March) on every complete season since UNG's 2007 inception and find
that roll drag erases any price-level seasonality: the seasonal strategy earns a negative CAGR, and the
t-stat on the winter premium is nowhere near 2. None/Mirage.
"""

from . import data, strategy  # noqa: F401
