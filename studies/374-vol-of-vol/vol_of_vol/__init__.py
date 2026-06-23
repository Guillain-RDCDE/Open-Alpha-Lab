"""Study 374 — Vol-of-Vol (does VVIX time equity risk better than the VIX?).

VVIX is the CBOE "volatility of the VIX": the VIX construction applied to VIX *options*.
The believers' claim is that when the vol-of-vol spikes, option markets are pricing a tail
the *level* of vol hasn't shown yet — so high VVIX should precede weak / drawdown-y forward
equity returns, **and** add information beyond the VIX itself.

This study fetches ^VVIX, ^VIX and SPY (yfinance, 2007→), measures forward SPY
returns/drawdowns after high-VVIX days vs the base rate, and — the decisive test — runs a
HAC/Newey-West regression of forward returns on standardized VIX *and* VVIX to ask whether
vol-of-vol carries **incremental** signal once VIX is controlled. A deterministic synthetic
control with a planted *incremental* edge knob proves the engine isolates vol-of-vol from
VIX leaking through.

See :mod:`vol_of_vol.data` (real loader + synthetic control) and
:mod:`vol_of_vol.strategy` (signal, conditional forward returns, the incremental HAC test,
placebo null, costed timing rule)."""

from . import data, strategy

__all__ = ["data", "strategy"]
