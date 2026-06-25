"""Study 474 — Bill Williams' Accelerator Oscillator (AC).

A mechanical, falsifiable encoding of Bill Williams' *Accelerator Oscillator*: the
Awesome Oscillator (AO = SMA5(median) - SMA34(median)) minus its own 5-bar SMA, i.e. the
second derivative of price ("acceleration"). The folklore says AC *leads* AO and price —
it changes colour (turns up/down) before momentum does, so two consecutive rising green
bars above zero is a high-probability buy. We test that as a forward-return study against a
drift-matched random-entry baseline and a shuffled-window placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
