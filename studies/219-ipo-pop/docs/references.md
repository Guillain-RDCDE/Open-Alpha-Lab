# References & literature map — Study 219 (IPO-Pop)

## The claims under test

### (A) The first-day pop

- **Ibbotson, Sindelar & Ritter (1988).** *Initial Public Offerings* — Journal of Applied
  Corporate Finance 1(2), 37–45. Documents the consistent phenomenon of IPO underpricing:
  on average, shares are sold below their first-day market price. The expected first-day
  return in the US has varied between 10% and 70% depending on the decade and market
  conditions. This is the foundational reference for Part A of our study.

- **Loughran & Ritter (2004).** *Why Has IPO Underpricing Changed Over Time?* —
  Financial Management 33(3), 5–37. Documents the "partial adjustment" phenomenon: the
  offer price adjusts only partially to pre-IPO demand, leaving money on the table.
  Shows that underpricing increased dramatically during the 1999–2000 dot-com bubble.

- **Rock (1986).** *Why New Issues Are Underpriced* — Journal of Financial Economics
  15(1–2), 187–212. The "winner's curse" model: uninformed investors receive full
  allocations only on overpriced (bad) IPOs; informed investors crowd the underpriced
  ones. Issuers underprice deliberately to attract uninformed capital. Explains why
  retail allocation in hot IPOs is a lottery with near-zero expected profits.

### (B) Long-run underperformance

- **Ritter (1991).** *The Long-Run Performance of Initial Public Offerings* — Journal of
  Finance 46(1), 3–27. The canonical study: using 1,526 US IPOs from 1975–1984, Ritter
  documents that IPOs underperform a matching portfolio of seasoned firms by −29% to −47%
  over the three years following the IPO. The "hot issue" periods (1980–1981) show the
  worst long-run performance. This is the central hypothesis for Part B.

- **Loughran & Ritter (1995).** *The New Issues Puzzle* — Journal of Finance 50(1),
  23–51. Extends Ritter (1991) to 1970–1990 data and finds similarly poor long-run
  performance (−30% over five years vs a size-matched benchmark). Attributes the puzzle
  to windows-of-opportunity: managers time issuance to periods of investor overoptimism.

- **Brav & Gompers (1997).** *Myth or Reality? The Long-Run Underperformance of Initial
  Public Offerings: Evidence from Venture and Nonventure Capital-Backed Companies* —
  Journal of Finance 52(5), 1791–1821. Finds that IPO underperformance is concentrated
  in small, non-venture-backed IPOs; venture-backed and large-cap IPOs show less
  underperformance when properly benchmarked. Relevant caveat for our large-cap table.

## Harvestability and allocation

- **Aggarwal (2003).** *Allocation of Initial Public Offerings and Flipping Activity* —
  Journal of Financial Economics 68(1), 111–135. Documents that institutional investors
  receive the bulk of IPO allocations in hot offerings; retail receives <10% of shares
  in the most oversubscribed deals. The empirical backbone for our "unharvestable" label.

- **Reuter (2006).** *Are IPO Allocations for Sale? Evidence from Mutual Funds* —
  Journal of Finance 61(5), 2289–2324. Shows that allocations correlate with brokerage
  commissions paid — a quid-pro-quo. Retail investors without brokerage relationships
  have effectively zero allocation in hot IPOs.

## Why long-run underperformance is hard to exploit

- **Ritter & Welch (2002).** *A Review of IPO Activity, Pricing, and Allocations* —
  Journal of Finance 57(4), 1795–1828. Comprehensive review showing that while first-day
  underpricing is large and well-documented, the long-run underperformance is inconsistent
  across samples and measurement methodologies. Short-selling newly listed companies is
  also difficult (limited borrow, high cost).

- **Schultz (2003).** *Pseudo Market Timing and the Long-Run Underperformance of IPOs* —
  Journal of Finance 58(2), 483–517. Argues that apparent long-run underperformance is
  partly an artefact of "pseudo market timing" (more IPOs happen when the market is high,
  so average post-IPO returns look bad simply due to reversion). Our small, survivorship-
  biased sample cannot adjudicate between Schultz and Ritter.

## Survivorship and sample biases

- **Brown, Goetzmann, Ibbotson & Ross (1992).** *Survivorship Bias in Performance
  Studies* — Review of Financial Studies 5(4), 553–580. The canonical reference for
  the survivorship problem we explicitly name: our table contains only tickers still
  tradeable on yfinance, biasing long-run returns upward.

- **Fama (1998).** *Market Efficiency, Long-Term Returns, and Behavioral Finance* —
  Journal of Financial Economics 49(3), 283–306. Cautions that long-horizon event-study
  results are highly sensitive to methodology (size/BM benchmarks, equal vs value
  weighting, exclusion of delistings). Our simple "return vs zero" test is the weakest
  possible form.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  implemented in [`strategy._hac_tstat`](../ipo_pop/strategy.py).

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`) — adjusted closes for post-IPO price
  history. IPO offer prices and first-day closes from publicly available SEC S-1 filings
  and historical market data. Window: 1997–2026.

## Related desk studies

- **[Study 142 — Split-Drift](../../142-split-drift/)**: another corporate-action event
  study with the same "buy the announcement, hold for 6-12m" structure and the same
  cautionary tale about market efficiency eating the edge before it reaches retail.
- **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)**: a calendar effect that,
  like IPO chasing, attracts retail attention but shows no durable net signal.
- **[Study 34 — Aftershock](../../34-aftershock/)**: post-earnings drift — the closest
  anomaly in spirit (buy after a surprising public announcement, hold for 3–12 months).
