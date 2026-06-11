"""Study 43 — Free-Lunch: betting against beta, and the leverage bill that isn't free.

Low-beta assets earn more per unit of risk than high-beta ones (Frazzini & Pedersen 2014). The BAB
factor exploits it — long low-beta (levered up to beta 1), short high-beta (levered down). The catch:
making the low-beta leg beta-1 needs ~3× leverage, and once you pay to borrow it, the famous Sharpe
collapses. We measure the gross edge and the financing bill that eats it.
"""

from . import data, strategy  # noqa: F401
