# References & literature map — Study 218 (SPAC-Performance)

## The claim under test

- **SPAC boom, 2020–2021.** Goldman Sachs, Morgan Stanley, Citigroup, and hundreds of
  celebrity/blank-check sponsors marketed SPACs as a superior IPO mechanism: faster
  (~3 months vs 6-12 for traditional IPO), cheaper (no roadshow), less dilutive (fixed
  target price), and generating better post-merger returns than traditional IPOs. The
  Defiance Next Gen SPAC Derived ETF (SPAK, launched Oct 2020) was purpose-built to
  provide index-level access to the claimed opportunity.
- **Media coverage 2020-2021.** *Bloomberg* ("The SPAC Boom Is the Democratisation of
  IPOs," 2020), *WSJ* ("Everyone Is Getting Rich on SPACs. Should You?" 2021), *FT*
  multiple pieces positioning SPAC sponsors as sophisticated dealmakers with aligned
  incentives. The marketing narrative was: sponsors have skin-in-the-game via promote
  shares and will only target quality businesses.

## The prior academic evidence on SPAC underperformance

- **Klausner, Ohlrogge & Ruan (2022).** *A Sober Look at SPACs*, Yale Journal on
  Regulation 39(1). The definitive pre-bust academic work: median SPAC investors who
  held through merger lost ~50% of invested capital within 12 months of deal close;
  the 20% promote plus warrants creates ~12-13% dilution at merger for public
  shareholders, even before any operational underperformance.
- **Gahng, Ritter & Zhang (2021).** *SPACs*, University of Florida working paper.
  SPACs underperform industry- and size-matched IPOs by roughly 35-50 percentage
  points over 12 months post-merger; the underperformance is concentrated in deals
  where SPAC sponsors have fewer conflicts of interest disclosures.
- **Loughran & Ritter (1995).** *The New Issues Puzzle*, Journal of Finance 50(1).
  Traditional IPOs underperform matched non-issuers by ~30% over 5 years post-IPO;
  SPACs layer an additional structural dilution on top of this well-documented pattern.
- **Jog & Sun (2007).** *Blank Check IPOs: A Home Run for Management*, Journal of
  Financial Economics 85(3). Earlier evidence that blank-check vehicles structurally
  transfer value from public investors to insiders; confirmed for the modern SPAC era
  by Klausner et al. (2022).

## The structural mechanics driving underperformance

- **Sponsors' "promote" (20% founder shares).** Sponsors receive ~20% of the combined
  company at no cost at merger. If SPAC raises $100m, sponsors get $25m of equity
  for free, diluting public investors immediately. Klausner et al. quantify this as
  ~8.5% dilution at median for 2019-2020 SPACs.
- **Warrants dilution.** Each SPAC unit (at $10) typically includes a fraction of a
  warrant struck at $11.50. If exercised these dilute further; if not, they were a
  cost of capital to the SPAC. Klausner et al. attribute ~3.7% dilution from warrants
  at median.
- **Redemption mechanics.** Investors who wish can redeem at ~$10 (trust value) before
  the merger vote; heavy redemptions (sometimes 95%+) deplete the SPAC's cash,
  making the acquired company undercapitalized at inception.
- **Forward projections.** De-SPACs (Lucid, Rivian, Clover Health, etc.) used 5-year
  forward revenue projections in their merger presentations — a practice not permitted
  in traditional IPO registration statements — inflating valuation at merger.

## The ETF-level evidence (SPAK)

- **SPAK prospectus (2020).** Defiance ETFs / Exchange Traded Concepts. SPAK tracked
  an index of post-merger de-SPAC equities (60%) and SPAC trusts pre-merger (40%).
  ER 0.45%. Launched 2020-10-01; delisted 2022-09-01 after assets fell below
  economically viable levels — itself a signal of category destruction.
- **SPAK delisting (2022).** ETF Trends, September 2022: Defiance closed SPAK due to
  shrinking AUM (from ~$350m peak to under $10m at close) driven by SPAC category
  collapse. The delisting date caps the observable window at 1.92 years.

## Related desk studies

- **[Study 139 — AI-Powered-ETF](../../139-ai-powered-etf/)**: AIEQ (IBM Watson),
  another "themed" active ETF that promised to beat the market and delivered persistent
  underperformance; same Jensen alpha methodology.
- **[Study 142 — Split-Drift](../../142-split-drift/)**: corporate actions and return
  effects; structural mechanics distorting price-based signals, analogous to SPAC
  sponsor economics distorting apparent returns.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's mechanical
  value + quality screener — the type of fundamental discipline SPACs systematically
  avoided in their merger targets.
- **[Study 196 — Long-Term-Reversal](../../196-long-term-reversal/)**: DeBondt-Thaler
  long-horizon reversal; SPAC mania is a textbook case of overpricing with subsequent
  reversal, consistent with the long-term reversal literature.

## Method lineage

- **Jensen (1968).** *The Performance of Mutual Funds in the Period 1945-1964*,
  Journal of Finance 23(2). Jensen's alpha as the CAPM intercept — the standard
  measure used here.
- **Newey & West (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3). HAC t-stat via
  Bartlett kernel — the inference engine for both tapes.
- **Lo (2002).** *The Statistics of Sharpe Ratios*, Financial Analysts Journal 58(4).
  Motivates the finite-sample uncertainty on Sharpe ratios reported here.
