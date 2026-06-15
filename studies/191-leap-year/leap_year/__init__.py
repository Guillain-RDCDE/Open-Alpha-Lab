"""Study 191 — Leap-Year.

Folk claim: leap years (which coincide with US presidential elections and the Summer
Olympics every four years) produce systematically better — or worse — stock returns
than other years.  We test calendar-year returns of the S&P 500 in leap vs non-leap
years, control for the four-year presidential-cycle confound (year-of-term), and apply
a multiple-comparisons correction for the three hypotheses a researcher could have
tested (leap-year premium, election-year premium, Olympiad premium).

Effective n for leap years: ~25 observations from the modern GSPC era (1928-2025).
"""

from leap_year import data, strategy  # noqa: F401
