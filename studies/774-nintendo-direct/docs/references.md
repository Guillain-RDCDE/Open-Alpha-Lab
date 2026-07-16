# References & literature map — Study 774 (Nintendo-Direct)

## The claim under test

- **The folklore.** "Buy the hype into a Nintendo Direct." A Nintendo Direct is Nintendo's
  flagship self-produced video presentation — the ~40-minute broadcast where new games,
  hardware and release dates are revealed. Gaming/finance chatter says NTDOY *rallies into*
  a Direct as anticipation builds, then either keeps ripping on a great showing or *sells
  the news* once it airs. It is the "buy the rumour, sell the news" reflex bolted onto
  Nintendo's biggest owned-media catalyst.
- **Why the calendar test is honest but imperfect.** The big *general* Directs cluster
  seasonally (February, an E3/June slot, September), so the *season* is guessable — but a
  Direct is typically **announced only ~1-3 days ahead**, not 7-10 like an Apple press
  invite. So a literal "buy 10 sessions before the Direct" rule is **not** truly
  calendar-known: you had to know the exact date to place the trade. We measure the run-up
  because it is exactly the folklore, and we count the look-ahead against tradability rather
  than pretending it away. Dates are hardcoded from Wikipedia's Nintendo Direct presentation
  tables and Nintendo Life's full broadcast-history list ([`data.py`](../nintendo_direct/data.py));
  2020 is intentionally absent (no traditional general Direct that year — COVID).
- **The efficient-markets prior.** A recurring, well-telegraphed product showcase is exactly
  the kind of catalyst a semi-strong-efficient market should already price — see Fama (1970,
  *Efficient Capital Markets*, JF). The desk's prior is that any tradable "rally into" is
  arbitraged away, especially in a thin OTC ADR.

## What the literature actually says about event drift

- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard & Thomas
  (1989, 1990, *JAR / JAE*). The canonical "prices drift *after* a scheduled information
  event." A Direct is a product showcase, not an earnings print, but the folklore borrows
  PEAD's intuition; our test asks whether any drift is present around the *Direct*
  specifically.
- **"Buy the rumour, sell the news" / anticipation effects** — the idea that a known catalyst
  is bid up beforehand and sold once realised is old market lore with a thin formal record;
  the closest academic cousins are the pre-announcement drift and scheduled-announcement
  premium literatures (e.g. Savor & Wilson, 2016, *JFQA*, on scheduled macro announcements).
  None of it says a *product* showcase pays.
- **Attention & investor-catalyst effects** — Barber & Odean (2008, *RFS*) on attention-driven
  buying; Da, Engelberg & Gao (2011, *JF*) on search-based attention. These motivate *why*
  retail and gaming fans might crowd into a high-profile Direct, and therefore why a
  disappointment/reversal is plausible — but attention is not, by itself, a tradable edge.
- **ADR / home-market microstructure** — Gagnon & Karolyi (2010, *JFE*) on cross-listed
  ADR–underlying price dynamics. NTDOY is a thin OTC ADR whose US close can lag the Tokyo
  (7974.T) tape, so a "US-close-to-US-close" abnormal return around a Japan-timed broadcast
  is partly a stale-price artefact — another reason to distrust a fragile signal here.

## Data & method

- **Real tape:** `NTDOY` (Nintendo Co., Ltd. US ADR) and `SPY` daily adjusted
  (total-return) closes via [yfinance](https://github.com/ranaroussi/yfinance), one combined
  panel. We measure the *abnormal* return `NTDOY − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  Directs (the correct unit — not a daily panel); Wilson hit-rate interval; a 20-seed ×
  200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-Direct run-up (and optional post-broadcast fade) — the detector must recover a planted
  bump and stay quiet on the null. See [`strategy.py`](../nintendo_direct/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Ball, R. & Brown, P.
(1968). **JAR**. · Bernard, V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. &
Wilson, M. (2016). **JFQA**. · Gagnon, L. & Karolyi, G. A. (2010). **JFE**.*
