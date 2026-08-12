"""Study 906 — EM Local Bonds, FX-Hedged (a UUP-overlay proxy).

Test whether stripping the FX from EM local-currency bonds (EMLC/LEMB/EBND) via a
long-UUP dollar-index overlay leaves a real local-rate carry vs USD-EM debt (EMB) and cash.
"""

from . import data, strategy  # noqa: F401
