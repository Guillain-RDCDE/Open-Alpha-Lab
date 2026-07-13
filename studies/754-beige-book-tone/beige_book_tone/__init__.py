"""Study 754 — Beige-Book-Tone (does the Fed's anecdote-tone lead the market?).

The claim: the Federal Reserve's **Beige Book** — the qualitative, anecdote-driven
summary of regional business conditions published eight times a year, ~two weeks before
each FOMC meeting — carries a readable *sentiment*. Score its text with the
Loughran-McDonald finance tone dictionary (net optimism = positive − negative words over
their sum), and the folklore says a *positive-tone* Beige Book precedes an equity **drift
up** in the days after release. We rebuild that signal on the real Beige-Book release
calendar aligned to daily SPY, with a strict release-date lag (you only know the tone once
the book is public), and ask whether the anecdote-tone leads the tape.

The tone series here is a **labelled proxy** — a small, hardcoded, narrative-anchored
reconstruction of the LM net-tone on cited Beige-Book excerpts (the full-text scrape +
dictionary count is the beat-7 extension), never presented under a real-tape banner. The
release *calendar* is real (snapped to the Beige-Book Wednesday cadence) and SPY is real
(yfinance daily adjusted close). A deterministic synthetic control proves the event-study
engine recovers a *planted* tone→drift link and does not manufacture one from noise.

See :mod:`beige_book_tone.data` (release calendar + LM-tone proxy + SPY loader +
synthetic control) and :mod:`beige_book_tone.strategy` (event-window drift, Welch *t*,
placebo null, a Newey-West tone→drift regression, and a costed event overlay).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
