"""Study 360 -- NAAIM-Exposure (the pros' positioning as a contrarian tell).

The NAAIM Exposure Index is a weekly survey of NAAIM member firms -- active money
managers -- reporting their *actual* current equity exposure on a 0-200% scale
(0 = fully in cash, 100 = fully invested, 200 = 2x leveraged long; the series can
go slightly negative when members are net short). NAAIM publishes the full weekly
history free since July 2006. Folklore treats it as a *contrarian* sentiment gauge:
when the pros are all-in (extreme high exposure) you should sell; when they have
bailed to cash (extreme low exposure) you should buy.

This study asks whether NAAIM extremes actually time forward SPY returns better
than the unconditional weekly drift. It is the **professional-positioning** twin of
the individual-investor survey ([Study 257 -- AAII](../257-aaii-sentiment/)), the
options crowd ([Study 261 -- put/call](../261-put-call-ratio/)) and the leverage
gauge ([Study 260 -- margin-debt](../260-margin-debt/)) -- same contrarian template,
different crowd (managers who *act*, not retail who *opine*).

See :mod:`naaim_exposure.data` (real NAAIM weekly + SPY tape, a deterministic
synthetic positive control, and a small hardcoded public fallback) and
:mod:`naaim_exposure.strategy` (regime sort, predictive HAC regression, contrarian
timing overlay vs buy-and-hold, all with one-week execution lag and one-way costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
