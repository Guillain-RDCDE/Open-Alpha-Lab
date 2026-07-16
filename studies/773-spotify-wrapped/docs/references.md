# References & literature map — Study 773 (Spotify-Wrapped)

## The claim under test

- **The folklore.** "Buy Spotify into Wrapped — every December the personalised
  year-in-review takes over social feeds, a free viral marketing blitz that reminds
  everyone how sticky the product is, so the stock rallies into the launch." A perennial
  retail / fintwit talking point: because **Spotify Wrapped** is the company's single most
  visible consumer moment of the year (hundreds of millions of shareable cards), the stock
  is supposed to *rally into* it and perhaps *fade after* once the buzz is spent.
- **Why it's a clean calendar test.** Wrapped ships in the same **late-Nov/early-Dec slot**
  every year (the rollout has fallen between Nov 29 and Dec 6 since the "Wrapped"-branded
  edition began in 2016), so the event is **known in advance** — a "buy K sessions before,
  sell on the launch" rule is calendar-known and zero-look-ahead by construction. The dates
  are hardcoded from Spotify Newsroom / Wikipedia ([`data.py`](../spotify_wrapped/data.py)).
  Because SPOT only lists via a NYSE **direct listing on 2018-04-03**, only 8 of the 10
  Wrapped seasons have a tradable tape.
- **The efficient-markets prior.** A recurring, calendar-fixed marketing event that everyone
  can see coming is exactly what a semi-strong-efficient market should already price, and —
  unlike an earnings print — Wrapped carries essentially *no new fundamental information*
  (it is a re-packaging of usage data that Spotify already has). The desk's prior is that
  any "rally into Wrapped" is a story, not a cash flow — see Fama (1970, *Efficient Capital
  Markets*, JF).

## What the literature actually says about event drift & attention

- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard & Thomas
  (1989, *JAR*; 1990, *JAE*). The canonical "prices drift *after* a scheduled information
  event." A Wrapped launch is a marketing event, not an earnings print, but the folklore
  borrows PEAD's intuition; our test asks whether any drift is present around the *Wrapped*
  date specifically. (It isn't.)
- **"Buy the rumour, sell the news" / anticipation effects** — the idea that a known
  catalyst is bid up beforehand and sold once realised is old market lore with a thin formal
  record; the closest academic cousins are the pre-announcement drift and
  scheduled-announcement premium literatures (e.g. Savor & Wilson, 2016, *JFQA*, on
  scheduled macro announcements). None of it says a recurring *marketing* campaign pays.
- **Investor attention & media effects** — Barber & Odean (2008, *RFS*) on attention-driven
  buying; Da, Engelberg & Gao (2011, *JF*, "In Search of Attention") on search-based
  attention and short-run price pressure. These motivate *why* a viral consumer moment might
  briefly draw retail eyeballs to SPOT — but attention spikes are transient and, as this
  study finds, do not translate into a tradable pre-event drift.
- **Sentiment / social-media price effects** — Tetlock (2007, *JF*) on media pessimism and
  stock prices; Bollen, Mao & Zeng (2011, *J. Computational Science*) on Twitter mood and
  the market. Wrapped is a textbook social-media event, so these frame the *mechanism* the
  folklore imagines; the null result here is consistent with attention/sentiment being
  already-priced or too diffuse to move a mid-cap ahead of a known date.

## Data & method

- **Real tape:** `SPOT` and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel from SPOT's
  2018-04-03 direct listing. SPOT is a high-beta single name, which is why we measure the
  *abnormal* return `SPOT − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  Wrapped years (the correct unit — not a daily panel), n = 8; Wilson hit-rate interval; a
  20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net
  leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-launch run-up (and optional post-event fade) — the detector must recover a planted
  bump and stay quiet on the null. See [`strategy.py`](../spotify_wrapped/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Ball, R. & Brown, P.
(1968). **JAR**. · Bernard, V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. &
Wilson, M. (2016). **JFQA**. · Tetlock, P. (2007). **JF**. · Bollen, J., Mao, H. & Zeng, X.
(2011). **J. Computational Science**.*
