"""Study 751 — Fortune-500-Inclusion (does making, or falling off, the list move the stock?).

Market lore, borrowed from the *real* S&P-500 index-inclusion effect: being **added to** a
prestigious list draws a wave of attention and buying, and being **dropped** sheds it — so a
company's **debut on the Fortune 500** (or its **exit**) should print an abnormal return
around the June list reveal. It sounds like the index-inclusion trade (Study 249), dressed in
a magazine cover.

The catch — and the whole point of the teardown — is that the two look alike but aren't. The
S&P 500 is an **investable index**: additions force **index funds to buy**, a genuine demand
shock. The Fortune 500 is a **media ranking by prior-year revenue** — no fund tracks it, no
one is *forced* to buy, and the revenue that decides the ranking was public months earlier.
Strip out the demand shock and the new information, and all that's left is a pure
**attention / prestige** effect. Does prestige alone move price?

We make it falsifiable with a textbook short-window **event study** over a hardcoded,
transparent, *cited* table of notable Fortune-500 **debuts** and **exits**, each snapped to
its year's list-reveal date. Around each event we fit a **market model** (stock = α + β·SPY)
on a clean pre-event window and cumulate the **abnormal return** (CAR) over the reveal window
— then confront it with a placebo null sized to the event count, a one-day execution lag, and
a deterministic synthetic power control. With ~a dozen events per bucket and no demand shock,
the honest answer is a small-sample near-zero.

See :mod:`fortune_500_inclusion.data` (the add/drop table + real loader + synthetic control)
and :mod:`fortune_500_inclusion.strategy` (market-model CAR, placebo null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
