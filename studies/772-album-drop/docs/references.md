# References & literature map — Study 772 (Album-Drop)

## The claim under test

- **The folklore.** "Buy Spotify (SPOT) when a blockbuster album drops — a Taylor Swift,
  Drake, Adele or Bad Bunny record that shatters single-day / single-week Spotify streaming
  records must send the stock up." A perennial retail / financial-media idea: because these
  drops dominate the platform and the streaming-record headlines, the *stock* is supposed to
  rally into the release and pop after it — the textbook "buy the rumour, sell the news."
- **Why it's a clean calendar test.** Major albums are announced weeks ahead (pre-order
  pages, teaser singles, launch dates), so the event is **known in advance** — a "buy K
  sessions before, sell on the day" rule is calendar-known and zero-look-ahead by
  construction. The dates are hardcoded from artist/label press releases
  ([`data.py`](../album_drop/data.py)); all 27 drops post-date Spotify's 2018-04-03 direct
  listing, so every event has SPOT price history.
- **The efficient-markets prior.** Spotify monetises via subscription (ARPU × MAU), not a
  per-stream royalty that a single mega-album meaningfully moves; one record week is a
  rounding error against a ~600M-user base, and the release date is common knowledge. A
  semi-strong-efficient market should already price all of this — see Fama (1970,
  *Efficient Capital Markets*, JF).

## What the literature actually says about event drift & attention

- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard &
  Thomas (1989, 1990, *JAR / JAE*). The canonical "prices drift *after* a scheduled
  information event." An album drop is a product/attention event, not an earnings print, but
  the folklore borrows PEAD's intuition; our test asks whether any drift is present around
  the *release* specifically.
- **Attention & investor-catalyst effects** — Barber & Odean (2008, *RFS*) on
  attention-driven buying; Da, Engelberg & Gao (2011, *JF*, "In Search of Attention") on
  Google-search-based attention predicting short-run returns. These motivate *why* retail
  might crowd into a headline-grabbing streaming record — but attention is not, by itself, a
  tradable edge, and a music-fan spike need not touch the equity.
- **Product-launch / new-product event studies** — Chaney, Devinney & Winer (1991, *Journal
  of Business*) find announcements of new products can move the *maker's* stock; the twist
  here is that the album is a *customer's* product on Spotify's platform, one step removed
  from Spotify's own cash flows, which is exactly why the prior is a flat zero.
- **"Buy the rumour, sell the news" / anticipation effects** — the idea that a known
  catalyst is bid up beforehand and sold once realised is old market lore with a thin formal
  record; the closest academic cousins are the pre-announcement-drift and
  scheduled-announcement-premium literatures (e.g. Savor & Wilson, 2016, *JFQA*, on
  scheduled macro announcements). None of it says a *third-party product* release pays.

## Data & method

- **Real tape:** `SPOT` and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. SPOT's high beta
  to SPY is why we measure the *abnormal* return `SPOT − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  album events (the correct unit — not a daily panel); Wilson hit-rate interval; a 20-seed ×
  200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-drop run-up (and optional post-drop fade) — the detector must recover a planted bump
  and stay quiet on the null. See [`strategy.py`](../album_drop/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Ball, R. & Brown, P.
(1968). **JAR**. · Bernard, V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Chaney, P.,
Devinney, T. & Winer, R. (1991). **Journal of Business**. · Savor, P. & Wilson, M. (2016).
**JFQA**.*
