# References & literature map — Study 904 (Shareholder-Yield + Quality)

## The claim under test

- **The shareholder-yield construct.** Meb **Faber & Richardson (2011), _The
  Shareholder Yield_** (and the Cambria Shareholder Yield ETF whitepaper): the "correct"
  cash-return-to-shareholders metric is dividends **plus net buybacks plus net debt
  paydown**, not dividend yield alone. Buybacks became the dominant channel after the
  2000s, so a dividends-only screen misses most of the payout.
  https://mebfaber.com/shareholder-yield/
- **Buybacks are not all equal — the "dilution theatre" problem.** A gross repurchase
  that merely offsets stock-based-compensation grants leaves the **net** share count flat;
  only a *net* reduction returns value. **Fried & Wang (2019), "Short-Termism, Shareholder
  Payouts, and Investment in the EU"** and much of the SBC-accounting literature document
  how gross-buyback headlines overstate real payout. The quality overlay is the attempt to
  screen for firms whose buybacks are *funded and net*.
- **The buyback / repurchase anomaly.** **Ikenberry, Lakonishok & Vermaelen (1995),
  "Market Underreaction to Open Market Share Repurchases"** (JFE 39): firms announcing
  repurchases earn positive abnormal returns over the following years — strongest among
  cheap ("value") firms, weak among glamour firms. The valuation caveat ("at reasonable
  valuations") is baked into the original evidence.
- **Quality as the screen.** **Asness, Frazzini & Pedersen (2019), "Quality Minus Junk"**
  (Review of Accounting Studies): profitable, growing, safe, well-managed firms earn higher
  risk-adjusted returns. The QUAL overlay proxies this — the hypothesis is that quality
  keeps the *funded, value-accretive* repurchasers and drops the theatre.
- **The live-vehicle test.** We grade the claim not on a paper factor but on the
  fee-paying products a retail/small-institutional investor can actually hold: the Invesco
  Buyback Achievers ETF (PKW) as the raw shareholder-yield leg and the iShares MSCI USA
  Quality Factor ETF (QUAL) as the overlay, raced against SPY on excess-of-cash Sharpe.

## What we measure, and the honesty rails

- **Excess-vs-excess Sharpe races.** Every sleeve is measured in excess of **BIL**, whose
  monthly total return *is* the realized cash return — no modelled risk-free. Cash cancels
  in a sleeve-minus-sleeve difference, so the QSY−RAW and QSY−SPY spreads are cash-free and
  we put a HAC *t* directly on their means.
- **One documented rebalance lag.** Sleeves are rebalanced to equal weight at each
  month-end using only within-month drift (returns known at *t*); no look-ahead. A
  single-member sleeve (RAW = PKW) never rebalances.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly spread; a
  paired **moving-block bootstrap** CI on the Sharpe **gap** itself (so serial and cross
  correlation survive); an era cut (split 2020-01); a 20-seed synthetic positive control.
- **Costs graded separately.** One-way spread × realized turnover, long-only, no borrow —
  the honest test of whether a thin monthly edge survives friction (here it trivially does;
  the problem is significance, not cost).
- **Short-history / survivorship named on the Signal axis.** QUAL lists 2013-07, so the
  race is one ~13-year mostly-bull regime with two drawdowns; the ETFs are survivors. BUYB
  (a standalone buyback ETF) lists 2026-05 and is too young to race — named, not tested.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on every monthly spread).
- **Lo, A. (2002), "The Statistics of Sharpe Ratios"** — why Sharpe races are run
  excess-vs-excess and why a gap needs a standard error / bootstrap, not eyeballing.
- **Politis & Romano (1994)** — the stationary / moving-block bootstrap behind the paired
  Sharpe-gap CI.
- Repo-wide protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar
  (HAC *t* ≥ 2 on the real tape for a `Real` stamp), excess-vs-excess races, one documented
  lag, synthetic controls as machinery proofs only.

## Data sources

- **yfinance** (Yahoo! Finance, public, no key) — daily auto-adjusted (total-return,
  net-of-fee) closes: PKW, QUAL, SPYD, SPY, BIL, BUYB. https://pypi.org/project/yfinance/
- **PKW** — Invesco BuyBack Achievers ETF (inception 2006-12, ER 0.62%): US firms that
  reduced net shares outstanding ≥5% over the trailing 12 months.
- **QUAL** — iShares MSCI USA Quality Factor ETF (inception 2013-07, ER 0.15%): high ROE,
  stable year-over-year earnings growth, low financial leverage.
- **SPYD** — SPDR Portfolio S&P 500 High Dividend ETF (inception 2015-10): a raw
  high-dividend-yield screen (the dividend-only shareholder-yield context leg).
- **SPY** — SPDR S&P 500 (the plain cap-weight market); **BIL** — SPDR 1-3 Month T-Bill
  (the tradable cash leg); **BUYB** — a standalone buyback ETF (lists 2026-05, too young).

## Related desk studies (the dedup map — what this study is NOT)

- [368-buyback-drift](../../368-buyback-drift/) — the **event-study** abnormal drift after
  individual buyback *authorizations* (Ikenberry underreaction). This study holds no event
  window: it races a live *shareholder-yield ETF* against a *quality overlay* and the market.
- [233-shareholder-yield](../../233-shareholder-yield/) — the **cross-sectional factor**
  (dividends + net buybacks) built long/short on S&P survivors. This study is the *product
  race* — a buyback wrapper with vs without a quality screen, long-only, vs SPY.
- [519-net-share-issuance](../../519-net-share-issuance/) — the **net-share-issuance
  factor** (issuers underperform, repurchasers outperform; Pontiff-Woodgate) on a large-cap
  cross-section. This study does not build an issuance signal; it grades the packaged
  buyback vehicle and a quality overlay on it.
- [900-quality-income](../../900-quality-income/) — the sibling *quality-vs-yield* dividend
  product race (SCHD+NOBL vs SPHD+VYM). This study swaps the axis to *buyback* shareholder
  yield (PKW) with a *quality* overlay (QUAL) and asks the beat-the-market question; it
  reuses that study's live-ETF template.

None of the siblings race a **live buyback / shareholder-yield ETF with vs without a
quality overlay against the plain market on excess-of-cash Sharpe** — this study's own axis.
