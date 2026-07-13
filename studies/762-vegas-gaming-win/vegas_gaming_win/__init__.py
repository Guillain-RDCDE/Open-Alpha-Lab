"""Study 762 — Vegas-Gaming-Win (does Strip GGR momentum lead casino stocks?).

The gaming-sector folklore: the Las Vegas **Strip gross gaming revenue** (GGR) tape is the
fundamental pulse of the casino business, so when Strip GGR **momentum turns up** the casino
equities (MGM, Caesars, Las Vegas Sands, Wynn, Boyd, Penn) are supposedly about to run — a
top-down, single-number sector-timing edge. We rebuild the believers' signal on the monthly
Strip-GGR tape (a hardcoded, clearly-labelled **approximate reconstruction** of the Nevada
Gaming Control Board's "Las Vegas Strip" line, since the NGCB PDFs aren't machine-fetchable
here) and measure forward 1/3/6/12-month **casino-basket** returns conditional on rising vs
falling GGR momentum, against the unconditional base rate, with a Welch *t*, a placebo null,
a lead/lag scan, and a tradable timing overlay.

The decisive finding is about *timing*, not direction: casino share prices and Strip GGR are
tangled around the cycle, but the **monthly GGR report is a backward-looking print released
~5 weeks late**, and the stocks — being forward-looking, liquid, and analyst-covered — turn
*first*. So GGR momentum is a coincident-to-lagging echo of a move the equities already made,
not a leading signal you can allocate to.

See :mod:`vegas_gaming_win.data` (hardcoded GGR reconstruction + casino-basket loader +
deterministic synthetic control) and :mod:`vegas_gaming_win.strategy` (TTM-momentum signal,
forward-return inference, placebo null, lead/lag, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
