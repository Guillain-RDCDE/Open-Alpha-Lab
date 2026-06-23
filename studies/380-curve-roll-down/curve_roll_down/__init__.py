"""Study 380 — Curve-Roll-Down (riding the yield curve).

The believers' rule: buy a bond on a **steep** curve, hold a year; if the curve is
unchanged the bond ages down to a lower yield, so you collect its carry PLUS a roll-down
price gain — supposedly a reliable beat over cash. Roll-down is a *static-curve* concept.

True constant-maturity Treasury yields ARE free on yfinance (``^IRX``/``^FVX``/``^TNX``/
``^TYX``), so we read the real curve, compute the textbook expected roll+carry of a 5y
duration sleeve, and compare it to the **realized** excess-over-cash once the curve
actually moves. The decisive finding is that the realized excess is real-but-fragile: it
is the duration term premium (carry), it survives a HAC t only weakly, and a single
hiking cycle erases the promised roll-down — a free lunch the curve takes back.

See :mod:`curve_roll_down.data` (real CMT-yield loader + deterministic synthetic control
with a planted slope->edge knob) and :mod:`curve_roll_down.strategy` (curve mechanics,
realized excess, Newey-West HAC t, slope-timing placebo, costs, regime split)."""

from . import data, strategy

__all__ = ["data", "strategy"]
