"""Study 722 — Logo-Rebrand (a rebrand: signal of renewal, or of a floundering firm?).

Market lore pulls two ways at once. The **renewal camp** says a new name or logo marks a
turnaround — a fresh identity that re-rates the stock (Google -> Alphabet, Facebook -> Meta):
*buy the rebrand.* The **skeptic camp** says a rebrand is exactly what a *floundering* firm
does to distract from a rotting business — a vanity red flag: *fade it.* Both are directional
bets on the abnormal drift after a rebrand, so one event study adjudicates them.

We make it falsifiable with a small **event study** over a hardcoded, transparent table of
~26 real corporate rebrands / logo changes 2010-2025 (name changes, identity refreshes, and
logo-only redesigns). Around each reveal we measure the abnormal (excess-of-SPY) return on a
short **announce** leg and a longer **drift** leg, then judge both against a Welch *t*, a
placebo null sized to the event count, and a base-rate win-rate. The decisive finding is
statistical: ~24 priced events is too few to certify either sign, and the surviving tape (the
worst-outcome rebrands delisted / went private, leaving no series) is biased *against* the
floundering thesis — so "renewal vs red flag" resolves to *neither*, a coin flip that a
buy-the-rebrand book cannot turn into an edge once even small large-cap costs hit.

See :mod:`logo_rebrand.data` (the rebrand table + real loader + deterministic synthetic
control) and :mod:`logo_rebrand.strategy` (event windows, abnormal-return inference, placebo
null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
