# References & literature map — Study 249 (Index-Inclusion)

## The claim under test

- **Shleifer (1986).** *Do Demand Curves for Stocks Slope Down?* — Journal of Finance
  41(3), 579–590. The foundational paper: when Shleifer examined additions to the S&P
  500, he found a significant positive abnormal return around the effective date, showing
  that index demand shifts create real price pressure. The average addition pop was
  roughly +3% in 1966–1983 data, inconsistent with perfectly elastic demand.

- **Harris & Gurel (1986).** *Price and Volume Changes in the S&P 500: New Evidence for
  the Existence of Price Pressures* — Journal of Finance 41(4), 815–829. Documented
  volume surges and price increases around S&P 500 additions, arguing for a short-lived
  price pressure effect followed by a reversal over weeks.

- **Lynch & Mendenhall (1997).** *New Evidence on Stock Price Effects Associated with
  Changes in the S&P 500 Index* — Journal of Business 70(3), 351–383. Showed that
  after Standard & Poor's switched to announcing additions in advance (starting 1989),
  the price effect split: an announcement effect (prices rise on news) and an effective-
  date effect (further rise as index funds buy), with a partial reversal after. This is
  the precise mechanism our pop and give-back windows attempt to capture.

- **Wurgler & Zhuravskaya (2002).** *Does Arbitrage Flatten Demand Curves for Stocks?*
  — Journal of Business 75(4), 583–608. Linked the size of the inclusion pop to the
  availability of close substitutes for arbitrageurs. Stocks with few substitutes show
  larger pops — demand curves slope more steeply.

## Evidence of decay and crowding

- **Chen, Noronha & Singal (2004).** *The Price Response to S&P 500 Index Additions
  and Deletions: Evidence of Asymmetry and a New Explanation* — Journal of Finance
  59(4), 1901–1929. Found that the price increase following addition has become permanent
  (not fully reversed) while the deletion effect reverses. Argued for an "information"
  component: addition signals quality.

- **Petajisto (2011).** *The Index Premium and Its Hidden Cost for Index Funds* — Journal
  of Financial Economics 102(3), 627–654. Quantified the trading cost S&P 500 index
  funds incur when they must buy new additions at elevated prices due to indexing demand.
  He estimated this "index premium" at ~0.2%/yr for passive S&P 500 funds, paid to the
  arbitrageurs who front-run additions. Consistent with our finding that the pop is
  arbitraged away over time.

- **Madhavan (2003).** *The Russell Reconstitution Effect* — Financial Analysts Journal
  59(4), 51–64. Similar inclusion-effect analysis for the Russell indices; the annual
  reconstitution generates a larger and more predictable pop (known universe beforehand).
  Contrast with S&P (committee-driven, less predictable).

- **Cai (2007).** *What's in the News? Information Content of S&P 500 Additions* — This
  literature strand argues that inclusions convey information (quality screen by the S&P
  committee) — part of the price increase is permanent. Our study cannot separate the
  information component from the pure demand-pressure component.

## Why the pop has faded

- **Passive-fund competition.** The growth of index ETFs (SPY since 1993, IVV, VOO, etc.)
  has made the index-buying pressure on effective date larger and more predictable — but
  also more intensely front-run by arbs. As more capital chases the same trade (buy
  before effective date, sell to index funds), the available pop is shared across a
  larger base and compressed.

- **Announcement-day effect capture.** Practitioners increasingly buy on announcement
  day (or even pre-announcement via probabilistic models), moving the return to the
  announcement window. By the time anyone measures announce→effective, most alpha
  is already gone for later entrants.

- **S&P methodology changes.** S&P has extended the announcement-to-effective window
  over time, giving more time for arbitrage to flatten the pop.

## Our methodology and related work

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  implemented in [`strategy._hac_tstat`](../index_inclusion/strategy.py).

- **Event-study framework.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment of
  Stock Prices to New Information* (International Economic Review) — foundational
  event-study template we follow.

- **Survivorship / selection bias.** Our table is a manually curated set of notable
  additions; it is not a systematic sample of all S&P 500 inclusions. Small-cap,
  low-float, or sector-rotation additions are underrepresented. This bias is named
  in results.md; it likely *overstates* the pop because we selected well-known,
  liquid names.

## Related desk studies

- **[Study 142 — Split-Drift](../../142-split-drift/)**: another corporate-action event
  study (stock splits) where yfinance gives effective dates only — same caveat
  (announcement vs effective) applies.
- **[Study 138 — Random-Forest](../../138-random-forest/)**: ML approach to identifying
  structural breaks in anomalies — relevant to understanding whether the inclusion pop
  has decayed post-2010.
- **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)**: calendar event with
  index-rebalancing mechanics — a structurally similar "known-date buying pressure" story.
