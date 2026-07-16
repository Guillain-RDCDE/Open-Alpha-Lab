# References & literature map — Study 771 (Box-Office-Bomb)

## The claim under test

- **The folklore.** "A movie bombs, sell the studio." A perennial financial-media and retail
  reflex: a notorious box-office flop — *John Carter*, *The Lone Ranger*, *The Marvels*,
  *Snow White* — is a public, humiliating, nine-figure write-down in the making, so The Walt
  Disney Company (DIS) should sag in the days and weeks after the disappointing opening
  weekend is known.
- **Why it's a clean calendar test.** A wide release opens on a known Friday; the weekend
  box-office estimate is public by Sunday and actuals by Monday, so the first trading session
  at which "it flopped" is common knowledge is the **Monday after opening** — a "short DIS on
  that Monday, cover K sessions later" rule is calendar-known and zero-look-ahead by
  construction. Opening dates are hardcoded from Box Office Mojo / studio releases
  ([`data.py`](../box_office_bomb/data.py)).
- **The efficient-markets prior.** One film — even a $200M write-down — is a rounding error
  for a ~$200bn conglomerate dominated by parks, ESPN, streaming and consumer products, and
  the flop is a *scheduled, public* disappointment. Semi-strong efficiency (Fama, 1970,
  *Efficient Capital Markets*, **Journal of Finance**) says it should already be priced.

## What the literature actually says

- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, **JAR**); Bernard &
  Thomas (1989, **JAR**; 1990, **JAE**). The canonical "prices drift *after* a scheduled
  information event." A box-office weekend is a product-news event, not an earnings print,
  but the "sell after the bad news" reflex borrows PEAD's intuition; our test asks whether
  any post-event drift is present around a *flop* specifically.
- **Product-market news and stock returns** — Chen, Da & Zhao (2013, **RFS**, "What Drives
  Stock Price Movements?") and the broader cash-flow-news literature motivate *why* a
  revenue-relevant product event could move a stock, while emphasising that a single product
  is a small share of a diversified firm's cash flows.
- **Box-office information and studio stocks** — a thin but real strand studies whether
  opening-weekend surprises move media-company equity (e.g. Joshi & Hanssens, 2009,
  *Journal of Marketing*, on the effect of advertising and box-office on firm value; and
  event-study work on entertainment-industry announcements). None of it establishes a
  tradable "sell the flop" edge on a mega-cap distributor.
- **Attention & sentiment** — Barber & Odean (2008, **RFS**) on attention-driven trading and
  Da, Engelberg & Gao (2011, **JF**) on search-based attention motivate why a high-profile
  flop draws retail attention (and a possible over-reaction), but attention is not, by
  itself, a tradable edge.
- **Diversification / one-line-of-business dilution** — the intuition that a single film's
  P&L is swamped by a conglomerate's other segments is the core reason the prior is a
  non-event; see any corporate-finance treatment of segment materiality.

## Data & method

- **Real tape:** `DIS` and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. We measure the
  *abnormal* return `DIS − SPY` so the test is not just Disney's market beta.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  flop events (the correct unit — not a daily panel); Wilson hit-rate interval; a 20-seed ×
  200-draw random-window placebo per cut; a leave-one-out jackknife; a costed short leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  post-flop sell-off (and optional pre-release drift) — the detector must recover a planted
  drop monotonically and stay quiet on the null. See
  [`strategy.py`](../box_office_bomb/strategy.py).

*Fama, E. (1970). **Journal of Finance**. · Ball, R. & Brown, P. (1968). **JAR**. · Bernard,
V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Chen, L., Da, Z. & Zhao, X. (2013). **RFS**. ·
Joshi, A. & Hanssens, D. (2009). **Journal of Marketing**. · Barber, B. & Odean, T. (2008).
**RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**.*
