"""Study 745 — Corporate-Jet-Index (is a jet-loving CEO a governance short?).

David Yermack's *Flights of Fancy* (2006, JFE) found that firms which first disclose
their CEO's **personal use of the company aircraft** go on to *underperform* the market
by roughly **4%/year**, risk-adjusted — the corporate jet as a canary for entrenched,
value-destroying management. The believers' trade writes itself: **short the flying
CEOs, buy the frugal ones**, and harvest the governance discount.

We steelman that as a **long/short characteristic sort**. A hardcoded, cited table tags
~24 large-caps as **heavy-perk** (a documented CEO personal-aircraft red flag — Ellison,
Musk, Zuckerberg, Adelson, Wynn, Jeffries, Irani, Icahn…) or **low-perk** (matched
frugal-reputation peers — Buffett, Costco, Walmart, TI…). We form an equal-weight
**low − heavy** book (long the frugal basket, short the flyers), each name entering only
*after* its perk is public (a one-month execution lag, no look-ahead), and judge the
monthly long/short return with a **Newey-West (HAC)** *t* on the raw spread and on a
**market-model alpha** (is it just beta?), net of costs and short borrow.

The decisive confound is stated up front and named on the Signal axis: **survivorship**.
The archetypal corporate-jet abusers — Tyco (Kozlowski), WorldCom (Ebbers), Enron,
Chesapeake (McClendon) — *went to zero and delisted*, so a survivor tape can only hold
the flyers that lived. And the flyers that lived are disproportionately **founder-led
growth compounders** (Oracle, Tesla, Meta, Alphabet) — a growth/founder tilt that has
nothing to do with jets. Both pull the naive basket in opposite directions; we separate
them.

See :mod:`corporate_jet_index.data` (hardcoded perk table + yfinance monthly loader +
deterministic synthetic control with a plantable governance-discount edge) and
:mod:`corporate_jet_index.strategy` (long/short construction, HAC *t*, market-model
alpha, one-way costs + short borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
