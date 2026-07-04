# References & literature map — Study 622 (Thematic-ETF-Curse)

## The claim under test

- **The seminal paper.** Itzhak Ben-David, Francesco Franzoni, Byungwook Kim & Rabih Moussawi,
  *Competition for Attention in the ETF Space* (2023, **Review of Financial Studies** 36(3),
  987–1042; earlier NBER WP 28369, 2021). Their headline finding: **specialized (thematic)
  ETFs hold attention-grabbing, overvalued stocks and deliver about −5%/yr risk-adjusted
  returns over their first five years**, with the underperformance starting at launch —
  issuers launch these products near the **peak of the theme's hype** because that is when
  attention (and inflows) are easiest to harvest. The broad, plain-vanilla segment of the ETF
  market shows no such drag. <https://doi.org/10.1093/rfs/hhac048>
- **The mechanism.** Launch timing is endogenous to attention: a theme gets hot → valuations
  and media coverage peak → the issuer packages it → the ETF is born holding the theme at its
  most expensive, right before mean reversion. The launch date is therefore *itself* the sell
  signal — the claim we test.

## Adjacent evidence

- Kenechukwu Anadu et al. / Morningstar's periodic **"Big Ideas / thematic funds"** reviews:
  most thematic funds fail to survive and beat a global benchmark over any trailing decade
  (survival rates < 50%, success rates ~10%). E.g. Morningstar, *Global Thematic Funds
  Landscape* (annual). <https://www.morningstar.com/lp/global-thematic-fund-landscape>
- Sam Huber & Martin Lettau et al. on specialized-ETF underperformance post-2020; and the
  broader IPO-timing literature (Ritter's long-run IPO underperformance) — the same
  launch-at-peak-attention logic one asset class over. Ritter, *Initial Public Offerings:
  Updated Statistics* <https://site.warrington.ufl.edu/ritter/ipo-data/>
- Barber & Odean (2008), *All That Glitters: The Effect of Attention and News on the Buying
  Behavior of Individual and Institutional Investors*, RFS 21(2) — the attention-driven-buying
  foundation the ETF paper builds on.

## Named desk siblings (dedup guard)

- [Study 334 — ARK-Innovation](../334-ark-innovation/) is **one fund's** boom-bust and the
  dollar-weighted buy-the-top machinery. This study is the **category-level launch-timing
  claim** across ~48 thematic launches — ARKK is one row here (and the ex-ARK robustness drops
  it entirely).
- [Studies 393](../393-ai-datacenter-basket/)–[396](../396-reshoring-basket/) (AI-datacenter,
  [defense](../394-defense-basket/), [quantum](../395-quantum-computing-basket/),
  [reshoring](../396-reshoring-basket/)) test **today's** thematic baskets as current trades.
  This study tests the **historical launch-date rule** — what a dollar put in at every thematic
  launch since 2005 earned, risk-adjusted.

## Data sources

- **yfinance** (Yahoo Finance) — daily auto-adjusted (total-return) closes for the 48 thematic
  ETFs, the 13 broad index-ETF launches, SPY (market) and ^IRX (13-week T-bill discount yield,
  the risk-free proxy). <https://github.com/ranaroussi/yfinance>
- **Expected inception months** hardcoded per ticker from the issuers' fund pages / prospectus
  summaries (Global X, ARK, First Trust, iShares, VanEck, Invesco, Roundhill, KraneShares,
  WisdomTree, Amplify, ProShares, Defiance, Renaissance, U.S. Global, Wedbush, AdvisorShares,
  Bitwise, Schwab, Vanguard, SSGA, JPMorgan) — used only as a **ticker-reuse guard** (a Yahoo
  symbol recycled from an unrelated fund would splice the wrong history onto the launch date;
  SPLG and ONLN were dropped by this guard).
- **Survivorship caveat**: delisted thematics (SNSR, YOLO, MOON, UFO, AWAY, NERD, …) are
  absent from yfinance — named on the Signal axis; the bias runs **against** the finding.

## Method citations

- Newey & West (1987), *A Simple, Positive Semi-definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3) — the HAC t on the
  calendar-time alpha (Bartlett kernel, lags 6 and 12).
- Fama (1998), *Market Efficiency, Long-Term Returns, and Behavioral Finance*, JFE 49(3) —
  why long-horizon event studies belong on a **calendar-time portfolio** footing (pooled
  event-month regressions pseudo-replicate the same calendar months).
- Sharpe (1964) CAPM — the one-factor risk adjustment; alphas are excess-vs-excess
  (fund − T-bill on SPY − T-bill).
