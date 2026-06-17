# References & literature map — Study 277 (Trading-Cards)

## The claim under test

The post-2020 thesis that **graded collectible cards (Pokemon, vintage sports)
are an investable alternative asset class** — equity-like or better returns, low
correlation to stocks, and an inflation hedge you can physically hold. The pitch
was amplified by fractional-ownership platforms (Rally, Collectable, Otis/Public),
card-focused indices (PWCC "500"/"100" indices, Card Ladder), and a wave of
auction-house marketing during the 2020–2021 mania.

## Why the index can look so good — and why it's a mirage

- **Survivorship bias (named on the Signal axis).** Every collectible price index
  tracks the cards that *kept trading* — the iconic PSA-10 Charizard, the Jordan
  rookie — not the billions of "junk wax" commons and ungraded duds that are worth
  less than the toploader they sit in. A random buyer's realized return is far
  below any published index. This biases every collectible index *upward*.

- **One boom.** The entire "cards beat stocks" record lives in 2020–2021: stimulus
  cash, lockdown boredom, influencer hype, and a flood of new buyers roughly
  tripled blue-chip graded prices, then the froth left and the same comps
  round-tripped through 2022–2023. With ~35 annual observations and one giant
  spike, any naïve statistic is dominated by it. Dropping just 2020–2021 collapses
  the index CAGR to ~2%/yr.

- **Brutal frictions.** Collectibles trade by auction: buyer's premiums of
  ~15–20%, seller's commissions on top, grading/slabbing fees (PSA/BGS/SGC), and
  the wide bid–ask of an illiquid, slow-settling market. A buy-and-hold-then-sell
  round trip can cost 25–40% before any holding-period appreciation. We amortise a
  generous single round trip over a 5-year hold; even that turns a positive gross
  return negative.

- **No yield.** Cards pay no dividend or coupon — they cost money to store, insure,
  and protect. The honest comparison is therefore the S&P **price** return, not
  total return; adding dividends widens the equity lead further.

## Academic & industry literature on collectibles as assets

- **Burton, B. J. & Jacobsen, J. P. (1999).** "Measuring Returns on Investments in
  Collectibles." *Journal of Economic Perspectives*, 13(4), 193–212. Surveys
  collectible-asset returns (stamps, art, coins) and finds them generally **below**
  equities once measurement and selection biases are corrected.

- **Dimson, E. & Spaenjers, C. (2011).** "Ex Post: The Investment Performance of
  Collectible Stamps." *Journal of Financial Economics*, 100(2), 443–458. A
  rigorous long-run study; collectibles earn a modest real return, well under
  equities, with high idiosyncratic risk and large transaction costs.

- **Mei, J. & Moses, M. (2002).** "Art as an Investment and the Underperformance
  of Masterpieces." *American Economic Review*, 92(5), 1656–1668. The famous
  art-index paper: returns below equities, and the "trophy" pieces underperform —
  a direct analogue to chasing grail cards.

- **Goetzmann, W. N. (1993).** "Accounting for Taste: Art and the Financial Markets
  over Three Centuries." *American Economic Review*, 83(5), 1370–1376. Art tracks
  the broad economy with a lag and offers little diversification when it matters.

- **Pénasse, J. & Renneboog, L. (2022).** "Speculative Trading and Bubbles:
  Evidence from the Art Market." *Management Science*. Documents bubble dynamics in
  collectibles — relevant to the 2020–2021 card mania and its round-trip.

## Method lineage

- **Newey-West (HAC) regression.** $C_t = \alpha + \beta M_t + \varepsilon_t$ with
  heteroskedasticity-and-autocorrelation-consistent standard errors (Newey & West,
  1987). The intercept's HAC t-stat is the bar for a *real* structural edge; we
  require |t| ≥ 2 on the real tape.
- **Block bootstrap.** Stationary block resampling (Politis & Romano, 1994) of the
  paired annual returns, block length 3, to put a confidence interval on the alpha
  when n is tiny and one boom dominates.
- **Friction accounting.** A round-trip cost of `2 × one_way` amortised over a
  typical holding period, subtracted from each annual return — deliberately
  generous (one round trip per multi-year hold, not per year).

## Data sources

- **S&P 500 price (^GSPC).** Cached at repo-level `_cache/^GSPC_split_only.parquet`;
  December-to-December price returns 1991–2025. Price-only, label-consistent with
  the card *price* index. No network needed.
- **Graded-card index.** Curated and hardcoded in `data.py` (base 100 = 1990).
  There is no clean, machine-readable, survivorship-corrected total-return index
  for trading cards, so this is a stylised composite anchored to publicly reported
  moves from PWCC/Card Ladder-style market indices and widely-cited auction comps
  (Goldin, PWCC, Heritage). It is an honest best effort, not a vendored vendor
  index; the verdict does not hinge on its exact levels (see the bootstrap and the
  synthetic positive control).

## Related desk studies

- **Folklore / single-boom anomalies** elsewhere in the desk share this structure:
  a striking sample, a tiny n, and a result that does not survive costs or
  out-of-sample testing.
