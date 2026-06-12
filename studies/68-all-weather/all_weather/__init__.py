"""Study 68 — All-Weather: risk parity (inverse-vol allocation) vs the usual mixes.

Risk parity genuinely diversifies — unlevered, on SPY/IEF/GLD/DBC it earned a higher Sharpe (~0.91) and
a far smaller drawdown (~−17%) than 60/40 (0.80 / −31%) or equities (0.64 / −55%). But it returned only
about half of plain SPY: the "all-weather portfolio that beats everything" is overstated — its edge is
risk-adjusted and leans on a 40-year bond bull and on leverage to turn low vol into competitive return.
"""
from . import data, strategy  # noqa: F401
