"""Study 717 — Person-of-the-Year (the magazine-cover-curse, tested on TIME's honorees).

Folklore: landing on a magazine cover marks a peak — the *cover curse*. TIME's
**Person of the Year**, announced every mid-December, is the most-watched cover of all.
When the honoree runs (or is the face of) a public company, does the stock *drift down*
after the coronation? We pin the question with a textbook long-horizon **event study**:
around each mid-December announcement, the stock's **cumulative abnormal return** (CAR
over 1 / 3 / 6 / 12-month windows, market-model adjusted vs SPY) — is it reliably
negative (a curse), and can you short it?

The catch is the *count*. In a quarter-century of Persons of the Year, only a **handful**
of honorees ran a tradable public company — most picks are politicians or abstract groups
("The Protester", "You", "The Silence Breakers"). We hardcode a transparent, cited table
of the business honorees (ticker, announcement date) and event-study the post-coronation
drift. The decisive finding is statistical: with ~four tradable events, dominated by two
bubble-era icons (AMZN'99, TSLA'21), the "curse" is indistinguishable from selection —
TIME crowns people at their *zenith*, and zenith stocks mean-revert.

See :mod:`person_of_the_year.data` (hardcoded honoree table + yfinance loader +
deterministic synthetic positive control with a plantable curse edge) and
:mod:`person_of_the_year.strategy` (market-model long-horizon CAR, prior-run-up confound,
Welch t / placebo null, borrow-aware short costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
