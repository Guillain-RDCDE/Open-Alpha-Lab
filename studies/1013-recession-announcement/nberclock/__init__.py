"""Study 1013 — It's Official.

The National Bureau of Economic Research does not *forecast* recessions; it *dates* them, and it
dates them late. The committee announced that the Great Recession had begun on 1 December 2008 —
a year after the December 2007 peak and ten weeks after Lehman Brothers. It announced that the
recession had ended on 20 September 2010, fifteen months after the June 2009 trough. The market's
own turning points came long before either press release.

That lag is the raw material of a popular piece of folk wisdom: *by the time it is official, the
market has already done its falling — the announcement is a buy signal; and by the time the
recovery is declared, the cheap prices are long gone.* This study measures each half of that
sentence separately, on a century of total-return data for the lead/lag and on every official
announcement since the committee began issuing them in 1980.

- :mod:`nberclock.data` — the hard-coded NBER chronology and announcement dates, the two bundled
  real tapes (Fama-French monthly total return, skfolio daily S&P 500 price index) and the
  deterministic synthetic world with a ``signal_strength`` knob.
- :mod:`nberclock.strategy` — the lead/lag measurement, the announcement event study, the
  rotation (circular-shift) permutation test, the power curve and the timing rules.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
