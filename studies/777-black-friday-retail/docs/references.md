# References & literature map — Study 777 (Black-Friday-Retail)

## The claim under test

- **The folklore.** "Buy retail before Black Friday — the sector always runs up into the
  biggest shopping day of the year on strong holiday-sales hopes, then sells the news."
  A perennial financial-media / retail-trader trade: because Black Friday (the Friday
  after US Thanksgiving) is the traditional kick-off of the holiday shopping season and
  the year's single biggest catalyst for the consumer-discretionary complex, the SPDR S&P
  Retail ETF (`XRT`) is supposed to *rally into* it and *fade after* — the retail flavour
  of "buy the rumour, sell the news."
- **Why it's a clean calendar test.** Black Friday is fixed by statute (the day after the
  fourth Thursday of November), so the event is **known years in advance** — a "buy K
  sessions before, sell on the day" rule is calendar-known and zero-look-ahead by
  construction. The dates are hardcoded from the 4th-Thursday-of-November rule, cross-
  checked against the NYSE holiday calendar ([`data.py`](../black_friday_retail/data.py)).
- **The efficient-markets prior.** A catalyst everyone can put in their calendar is exactly
  what a semi-strong-efficient market should already price. The desk's prior is that any
  "rally into" is arbitraged away — see Fama (1970, *Efficient Capital Markets*, JF).

## What the literature actually says

- **Calendar/seasonal anomalies & data-mining.** The Halloween/"Sell in May" effect
  (Bouman & Jacobsen, 2002, *American Economic Review*) and the January/turn-of-year
  effects (Rozeff & Kinney, 1976, *JFE*; Keim, 1983, *JFE*) are the canonical seasonal
  claims — and the cautionary tale is Sullivan, Timmermann & White (2001, *JoE*): once you
  correct calendar effects for the *universe of rules you could have tried*, most shrink
  toward noise. A single "run up into Black Friday" window is exactly such a rule.
- **Holiday / pre-holiday effects.** Ariel (1990, *Journal of Finance*, "High Stock Returns
  Before Holidays") and Lakonishok & Smidt (1988, *RFS*, "Are Seasonal Anomalies Real?")
  document elevated pre-holiday returns in the broad market — motivating *why* the days
  around Thanksgiving might carry a tilt, but they concern the whole market, not a
  *sector-relative* Black-Friday retail trade, and both flag fragility.
- **"Buy the rumour, sell the news" / anticipation effects.** The idea that a known
  catalyst is bid up beforehand and sold once realised is old market lore with a thin
  formal record; the closest academic cousins are the pre-announcement drift and
  scheduled-announcement premium literatures (e.g. Savor & Wilson, 2016, *JFQA*, on
  scheduled macro announcements). None of it says a *shopping holiday* pays a sector.
- **Attention & retail-catalyst effects.** Barber & Odean (2008, *RFS*) on attention-driven
  buying; Da, Engelberg & Gao (2011, *JF*) on search-based attention. These motivate *why*
  retail traders might crowd into the highest-attention retail week of the year — but
  attention is not, by itself, a tradable edge.
- **Retail sales & consumer-discretionary returns.** The link from holiday-sales surprises
  to retailer returns is real but realises *after* the sales prints (post-earnings-
  announcement drift; Bernard & Thomas, 1989, *JAR*), not in a pre-Black-Friday calendar
  window — which is why the "sell the news" half is the more plausible one, and it is
  precisely the half that comes up empty here.

## Data & method

- **Real tape:** `XRT` (SPDR S&P Retail ETF, inception 2006-06-19) and `SPY` daily
  adjusted (total-return) closes via [yfinance](https://github.com/ranaroussi/yfinance),
  one combined panel. XRT's ~1 beta to SPY is why we measure the *abnormal* return
  `XRT − SPY`, netting out the broad market's own November seasonality.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  Black-Friday years (the correct unit — not a daily panel); Wilson hit-rate interval; a
  20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife; a costed
  net leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-Black-Friday run-up (and optional post-event fade) — the detector must recover a
  planted bump and stay quiet on the null. See
  [`strategy.py`](../black_friday_retail/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Bouman, S. &
Jacobsen, B. (2002). **American Economic Review**. · Rozeff, M. & Kinney, W. (1976).
**JFE**. · Keim, D. (1983). **JFE**. · Sullivan, R., Timmermann, A. & White, H. (2001).
**Journal of Econometrics**. · Ariel, R. (1990). **Journal of Finance**. · Lakonishok, J.
& Smidt, S. (1988). **RFS**. · Bernard, V. & Thomas, J. (1989). **JAR**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. &
Wilson, M. (2016). **JFQA**.*
