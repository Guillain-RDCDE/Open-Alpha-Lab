# References & literature map — Study 781 (Quad-Witching-Hangover)

## The claim under test

- **The folklore.** "The week after quad-witching is a hangover — the market drifts lower
  after the quarterly expiration churn." Quadruple witching is the simultaneous expiration
  of stock-index futures, stock-index options, single-stock options and single-stock futures
  on the **third Friday of March, June, September and December**. Dealers unwind large,
  offsetting hedges into that close; the trader-lore says the market is left depleted and
  soft the following week. A perennial financial-media talking point every quarter.
- **Why it's a clean calendar test.** The event is a fixture of the exchange calendar,
  literally "third Friday of the quarter's last month," so it is **known years in advance** —
  a "sit out / short the week after" rule is calendar-known and zero-look-ahead by
  construction. The 84 dates 2005→2025 are hardcoded from the CBOE/CME quarterly expiration
  calendar ([`data.py`](../quad_witching_hangover/data.py)).
- **The efficient-markets prior.** A perfectly scheduled, decades-old, everyone-knows-it
  calendar event is exactly what a semi-strong-efficient market should already price — see
  Fama (1970, *Efficient Capital Markets*, JF). The desk's prior is None/Mirage.

## What the literature actually says about expiration effects

- **Expiration-day effects on volume, volatility and price** — Stoll & Whaley (1987, *JF*;
  1991, *Journal of Futures Markets*) documented elevated volume and small, largely
  reversing price effects *on* the S&P expiration day itself. The effects are intraday and
  reverse quickly — they are about the *Friday close*, not a multi-day *following-week* drift.
- **Triple/quadruple witching microstructure** — Chow, Yung & Zhang (2003, *Journal of
  Futures Markets*) and later work find the witching-day print distortions have shrunk as
  markets matured and as more expiration was moved to the open (the "a.m. settlement" for
  index products). None of this literature claims a tradable *next-week* return.
- **Turn-of-the-quarter / options-expiration return seasonality** — the closest cousins are
  the options-expiration-week studies (e.g. Stivers & Sun on OpEx-week return patterns) and
  the broader calendar-anomaly literature (Lakonishok & Smidt, 1988, *RFS*, "Are seasonal
  anomalies real?"), which repeatedly find that once you correct for data-snooping and
  transaction costs, calendar patterns are fragile or gone.
- **Data-snooping caution** — Sullivan, Timmermann & White (1999, *JF*) and Bailey et al.
  (2014) on backtest overfitting: calendar folklore is the canonical multiple-testing trap,
  and a quarterly "hangover" tested across a couple of windows is exactly the kind of pattern
  that survives only until it is measured honestly.

## Data & method

- **Real tape:** `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one tape. Because SPY *is* the index,
  this is a **self-benchmarked single-tape** study: we test SPY's own forward return, and the
  random-window placebo (against SPY's own history) supplies the "is it abnormal vs the usual
  drift?" yardstick.
- **Statistics:** one-sample *t* of the forward return across independent, non-overlapping
  quarterly events (the correct unit — not a daily panel); Wilson up-rate interval; a 20-seed
  × 200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded single-name drift world with a *planted*
  post-event hangover dip — the detector must recover a planted dip (turn clearly negative)
  and stay controlled on the null. See [`strategy.py`](../quad_witching_hangover/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Stoll, H. & Whaley, R.
(1987). Program Trading and Expiration-Day Effects. **JF**; (1991). **J. Futures Markets**. ·
Chow, Y., Yung, H. & Zhang, H. (2003). **J. Futures Markets**. · Lakonishok, J. & Smidt, S.
(1988). Are Seasonal Anomalies Real? **RFS**. · Sullivan, R., Timmermann, A. & White, H.
(1999). Data-Snooping, Technical Trading Rule Performance. **JF**.*
