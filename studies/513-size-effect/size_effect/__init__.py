"""Study 513 -- Size-Effect (Banz 1981 SMB): small-caps vs large-caps.

A cross-sectional sort of a broad survivor basket by market capitalisation. Long the
small-cap half, short the large-cap half, beta- and dollar-neutral variants, with one
documented execution lag, real one-way costs plus short borrow, a placebo/label-shuffle
null, seed-robust synthetic positive control, and the two slices the original anomaly is
famous for: a **January** concentration and a **post-1980 decay**.

The basket is the set of names still trading in 2026 -- survivorship is named explicitly
on the Signal axis and all positive numbers are read as upper bounds.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
