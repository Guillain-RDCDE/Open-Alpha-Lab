# References & literature map — Study 783 (IPO-Deal-Of-Year)

## The claim under test

- **The folklore.** "Fade the banks' *IPO of the year*." Every year the league-table press
  (*International Financing Review* / IFR runs formal "IPO of the Year" awards; the wider
  financial media anoint a marquee debut) crowns the biggest, splashiest, most heavily
  oversubscribed listing. The retail/desk lore — a loud cousin of the academic
  IPO-underperformance result — says the loudest deals then lag: the hype prices the pop,
  the sell-side victory lap marks the top, and the newly-public stock drifts down against
  the tape.
- **Why the test is *descriptive*, not tradable.** The "IPO of the year" crown is awarded
  *after* the fact (IFR's awards land in December/January; press retrospectives even later),
  so you cannot buy the basket at its opens. Our basket is therefore selected **ex post by
  design** ([`data.py`](../ipo_deal_of_year/data.py)): it answers "did the celebrated debuts
  underperform?", not "here is a live rule." That caps tradability at **Mirage** before a
  single number is computed.
- **The efficient-markets prior.** Underwriters set the offer price and the first print is a
  fair auction of public information, so a semi-strong-efficient market should not leave a
  free, calendar-known "short the famous IPO" edge — any drift should be small, skewed and
  swamped by name-specific noise. See Fama (1970, *Efficient Capital Markets*, JF).

## What the literature actually says about post-IPO returns

- **Long-run IPO underperformance** — Ritter (1991, *JF*, "The Long-Run Performance of
  Initial Public Offerings") and Loughran & Ritter (1995, *JF*, "The New Issues Puzzle"):
  IPOs underperform matched firms over 3–5 years. Our 12-month cut is a compressed version
  (so 2023/24 debuts still qualify), so we expect a *weaker* signal than Ritter's 3-year one.
- **First-day pop and its reversal** — Ibbotson (1975, *JFE*) on IPO underpricing; Ritter &
  Welch (2002, *JF*, "A Review of IPO Activity, Pricing, and Allocation") on the pop-then-fade
  pattern. This is why our 3-month window can be *positive* (the pop still live) while the
  6–12-month windows turn negative.
- **The skew / lottery nature of IPOs** — Green & Hwang (2012, *Management Science*) on
  IPOs as lottery-like, positively-skewed bets; Barberis & Huang (2008, *AER*) on
  probability-weighting and skewness preference. This is the crux of *our* result: the
  median marquee debut lags, but a few extreme right-tail winners (Palantir, Reddit, Arm)
  drag the *mean* back to noise — exactly the skew these papers describe.
- **Hot-market / sentiment-driven issuance** — Loughran & Ritter (2000, *JFE*); Baker &
  Wurgler (2006, *JF*, investor sentiment): the most-hyped deals cluster in hot markets and
  are the worst subsequent performers, motivating *why* a "marquee" tag might predict
  disappointment even if the basket mean is a wash.

## Data & method

- **Real tape:** 17 marquee US-listed debuts (Facebook → Reddit) and `SPY`, daily adjusted
  (total-return) closes via [yfinance](https://github.com/ranaroussi/yfinance). We measure
  each name's **abnormal** forward return `name − SPY` at 63 / 126 / 252 sessions (~3 / 6 /
  12 months) from its first trading close.
- **Statistics:** one-sample *t* of the forward abnormal return across independent debut
  events (the correct unit — not a daily panel); Wilson hit-rate ("beat SPY") interval; a
  20-seed × 200-draw random-window placebo drawn from **each name's own** post-listing tape;
  a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (name, benchmark) world with a *planted*
  post-IPO forward drift — the detector must recover a planted under-performance
  monotonically and stay quiet on the null. See [`strategy.py`](../ipo_deal_of_year/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Ibbotson, R. (1975).
**JFE**. · Ritter, J. (1991). The Long-Run Performance of IPOs. **Journal of Finance**. ·
Loughran, T. & Ritter, J. (1995). The New Issues Puzzle. **JF**; (2000) **JFE**. · Ritter, J.
& Welch, I. (2002). A Review of IPO Activity. **JF**. · Barberis, N. & Huang, M. (2008).
**AER**. · Green, T. C. & Hwang, B. (2012). **Management Science**. · Baker, M. & Wurgler, J.
(2006). Investor Sentiment. **JF**.*
