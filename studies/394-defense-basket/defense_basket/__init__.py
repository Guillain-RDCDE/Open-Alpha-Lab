"""Study 394 — Defense-Basket (do defense stocks rally on war headlines?).

The folklore: when war, invasion or rearmament hits the tape, defense contractors (Lockheed,
RTX, Northrop, General Dynamics) "always" rally — a reliable, tradable reflex. We test it as
an **event study**: across a fixed, hardcoded set of geopolitical-shock dates we measure the
equal-weight defense basket's cumulative return *in excess of SPY* over the trading month
after each shock, entering one day after the headline.

The decisive finding is statistical, not directional: with ~20 lifetime shocks, the average
event-window excess is statistically indistinguishable from drawing the same number of random
dates — a high "win-rate" that mostly reflects defense's mild beta-driven drift, not a
shock-specific edge. Same thematic-basket / sample-size pathology that haunts the desk's other
"buy the headline" studies.

See :mod:`defense_basket.data` (real basket loader + deterministic synthetic control) and
:mod:`defense_basket.strategy` (event-window measurement, t / HAC-t, placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
