"""Study 757 — Cass-Freight 🚚 (does freight lead the market?).

The pitch, straight from the freight-macro commentariat: the **Cass Freight Index** —
a monthly gauge of the dollar-weighted volume of shipments Cass Information Systems
processes for its freight-payment clients — is a **real-economy leading indicator**.
Trucks and rail move the physical economy, so when freight rolls over, the argument
goes, the slowdown in output, earnings and the stock market is already baking. "Watch
the freight" is the folklore: a monthly print that supposedly *leads* the cycle and,
if you believe it, hands you a tradable overlay on transports (IYT) and the broad
market (SPY).

The real Cass shipments index is a published-but-not-free monthly series (Cass press
releases; FRED mirror ``FRGSHPUSM649NCIS``), so we **hardcode a small, cited,
approximate** annual path of the shipments level and interpolate it to a monthly
**LABELLED PROXY** — never presented as the live tape. The equity tapes (SPY, IYT) are
real yfinance closes. We then ask the two honest questions: (1) does the freight cycle,
known only *after* Cass publishes it, actually **lead** forward equity returns — or does
it *lag* a market that already turned? and (2) does a "own-stocks-when-freight-expands"
overlay beat buy-and-hold once you pay the publication delay, the execution lag and the
costs?

The decisive object is a lead-lag cross-correlation: a genuine leading indicator peaks
its correlation with equities at a **positive** lead (freight today predicts stocks
tomorrow). We test whether freight leads — or whether, like most "real-economy"
dashboards, equities lead *it*.

See :mod:`cass_freight.data` (the hardcoded Cass proxy + real SPY/IYT loaders + a
deterministic synthetic positive control with a PLANTED edge knob) and
:mod:`cass_freight.strategy` (freight YoY, forward-return inference, a Welch t / placebo
null, the lead-lag cross-correlation, and the net-of-cost timing overlay)."""

from . import data, strategy

__all__ = ["data", "strategy"]
