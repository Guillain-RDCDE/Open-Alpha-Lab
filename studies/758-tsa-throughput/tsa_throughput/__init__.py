"""Study 758 — TSA-Throughput (do checkpoint volumes nowcast the travel trade?).

The alt-data folklore: TSA checkpoint throughput is a daily, government-published,
real-time read on travel demand, so when TSA volumes are **accelerating** the travel
sector — airlines (JETS) and hotels (MAR/HLT) — should have tailwinds you can trade
before the official data or earnings confirm them. We rebuild the believers' signal on a
monthly TSA-throughput tape (a hardcoded, LABELLED PROXY snapshot of TSA's public daily
checkpoint numbers, since the TSA site is firewalled here) and measure forward
1/3/6/12-month travel-basket returns conditional on rising vs falling TSA momentum,
against the unconditional base rate, with a Welch t, a placebo null, a lead/lag scan, an
explicit market-beta control, and a tradable timing overlay.

The decisive finding is about *regime and timing*, not direction: TSA and the travel
trade are tangled around the COVID collapse-and-reopen, but travel equities priced the
recovery first — so a monthly TSA uptick is a coincident-to-lagging reopening echo, and
ex-COVID the "nowcast" is indistinguishable from noise.

See :mod:`tsa_throughput.data` (hardcoded TSA snapshot + basket loader + deterministic
synthetic control) and :mod:`tsa_throughput.strategy` (momentum signal, forward-return
inference, placebo null, lead/lag, market-beta control, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
