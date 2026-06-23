"""Study 387 — Economic-Surprise-Index 🎯 (should you buy when data keeps beating?).

The pitch: when the macro data keeps coming in *better than expected* — payrolls beat,
factory orders beat, sentiment beats — the economy has positive momentum the market hasn't
fully priced, so you should be long stocks. Citi's Economic Surprise Index (CESI) is the
famous commercial gauge of exactly this; the folklore is "ride the surprise."

Citi's CESI is proprietary, so we **CONSTRUCT a transparent surprise index** from a handful
of public FRED real-activity series. The trick the whole study turns on: there is no free
record of what analysts *expected*, so we proxy the consensus with each series' own
**trailing-12-month average growth** (a naive, unbiased, clearly-labelled stand-in for the
Street's forecast). The surprise is actual-minus-proxy, z-scored and averaged into a
composite ESI. We then ask whether a high ESI predicts higher forward SPY returns.

The decisive finding is a confound, not a directional miss: a surprise index built against a
*trailing-average* consensus is, by construction, a slow **mean-reversion / autocorrelation**
signal on macro momentum — and most of its apparent edge is the market's own well-known
post-recession recovery drift, not a tradable nowcast. Strip the proxy's mechanical
persistence and the excess collapses toward the base rate.

See :mod:`economic_surprise_index.data` (real FRED+SPY loader + a deterministic synthetic
positive control with a PLANTED edge knob) and :mod:`economic_surprise_index.strategy`
(the surprise index, forward-return inference, Welch t / placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
