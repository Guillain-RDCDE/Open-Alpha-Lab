"""Study 226 — Crude-Seasonality: does crude oil rally every spring into the driving season?

Monthly seasonality of crude oil (CL=F) and energy equities (XLE). The familiar story: refiners
ramp up in spring, driving demand rises, so April–June should be a reliably bullish window and
August–November a bearish one. We test it on 2000–2026 monthly data (309 months, CL=F; XLE from
1999): the spring-versus-autumn *spread* is real (t = 2.60) but individual months rarely clear the
bar after multiple-testing adjustment, the XLE timer badly trails buy-and-hold, and the crude timer's
apparent edge is concentrated in post-2012 data dominated by shale supply shocks. Weak signal at best;
Mirage as a calendar trading rule.
"""

from . import data, strategy  # noqa: F401
