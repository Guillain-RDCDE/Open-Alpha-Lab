"""Study 361 — Zweig Breadth Thrust (the rare "all-clear" buy signal).

Marty Zweig's Breadth Thrust: when the 10-day EMA of (NYSE advancing issues / total
issues) climbs from below 0.40 to above 0.615 within ~10 trading days, a powerful bull
move is supposed to follow — historically very rare (~a dozen ever) and "never wrong."

True NYSE breadth is not on yfinance, so we CONSTRUCT a transparent breadth proxy from a
large basket of liquid US large-caps (the share advancing each day), detect thrust events
on the proxy, and measure forward 3/6/12-month SPY returns vs the unconditional base rate.
The decisive finding is statistical, not directional: a signal with ~a dozen lifetime
occurrences cannot be a tradable strategy — the sample is too small to reject the null at
any honest confidence (cf. Study 167 Hindenburg-Omen, inverted: there the rare signal
"always" precedes crashes; here it "always" precedes rallies — both are sample-size
illusions).

See :mod:`zweig_breadth_thrust.data` (real basket loader + deterministic synthetic control)
and :mod:`zweig_breadth_thrust.strategy` (thrust detection, forward-return inference,
bootstrap / placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
