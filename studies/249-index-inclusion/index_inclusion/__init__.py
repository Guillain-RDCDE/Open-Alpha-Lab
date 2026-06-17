"""Study 249 — Index-Inclusion: is there still a free pop when a stock joins the S&P 500?

Tests the S&P 500 index-inclusion effect: a hardcoded table of notable additions
(announce + effective date) drives two windows:
  (a) pop: announce-to-effective abnormal return (the classic 'inclusion pop')
  (b) give-back: post-effective reversal as index buying pressure fades.

The classic pop faded as the trade became crowded; the give-back also attenuated.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
