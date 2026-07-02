# References & literature map — Study 554 (Airline-Bookings)

## The claim, at full strength

- **The alt-data lead pitch (vendor literature).** Card-panel and travel-data aggregators
  (e.g. transaction-panel providers, GDS/ARC ticketing feeds) market a **flight-bookings index**
  as a *leading* indicator for airline and travel equities: their panel "sees the demand" weeks
  before airlines report it, so booking momentum should forecast the sector's returns. This study
  tests the tradable core of that pitch — does the booking signal lead the *forward* return?

## Why the pitch usually fails — the efficient-market trap

- **Fama (1970)**, *"Efficient Capital Markets: A Review of Theory and Empirical Work."*
  *Journal of Finance* 25(2). The semi-strong form: prices already reflect public information. If
  a booking signal is public (or the demand it measures is), the *forward* return carries no edge —
  the lead is contemporaneous, not predictive. That is precisely the pattern this study plants and
  detects (huge same-month correlation, dead forward one).
- **Grossman & Stiglitz (1980)**, *"On the Impossibility of Informationally Efficient Markets."*
  *American Economic Review* 70(3). Information is only worth acquiring if it is *not yet* in the
  price; a widely-sold alt-data feed erodes its own forward edge as it is priced in.
- **The alt-data decay literature.** A recurring empirical finding across satellite, card-panel and
  web-scrape signals: a raw contemporaneous correlation to fundamentals that does **not** survive
  as a *forward* predictor of returns net of costs and crowding. Booking momentum is a clean
  instance of the genus.

## The predictive-regression method

- **Newey & West (1987)**, *"A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix."* *Econometrica* 55(3). The HAC standard error /
  *t* used for the predictive slope — the desk's inference bar for a serially-correlated monthly
  regression.
- **Stambaugh (1999)** / predictive-regression bias — the caution that persistent predictors and
  overlapping returns inflate significance; here the block-shuffle placebo guards the *t* directly
  rather than trusting the asymptotic SE.
- **Circular block bootstrap** (Politis & Romano 1992) — the placebo null: rotate the signal in
  blocks (preserving its own autocorrelation) against forward returns and read the predictive
  *t*'s tail probability.

## Neighbours on this bench (the dedup map)

- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** / **[Study 335 —
  Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** / **[Study 392 —
  Glassdoor-Sentiment](../../392-glassdoor-sentiment/)** — other *alt-data / sentiment leads*. This
  study is the **travel-booking** instance and its specific failure mode: a strong
  *contemporaneous* co-move with airline equities but a dead *forward* signal.
- **[Study 273 — Lego-Returns](../../273-lego-returns/)** / **[Study 275 — Whisky-Cask](../../275-whisky-cask/)** —
  fellow **synthetic-only** studies where the free, point-in-time data does not exist, so the
  Signal axis is capped at `WEAK` and the data-availability limitation is named openly.

## Shared method

- **Newey-West (1987)** HAC *t*; **Fama (1970)** semi-strong efficiency (the myth being checked).
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a *real* tape for `REAL`; synthetic-only ⇒ `WEAK` ceiling), one documented execution
  lag (signal at *t*, traded return at *t+1*), gross **and** net labelled, and shorts paying borrow.
