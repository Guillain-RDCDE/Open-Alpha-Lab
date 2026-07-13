# References & literature map — Study 745 (Corporate-Jet-Index)

## The claim under test

- **The source.** David Yermack (2006), *Flights of Fancy: Corporate jets, CEO
  perquisites, and inferior shareholder returns* (Journal of Financial Economics 80,
  211–242). Yermack hand-collects the first disclosures of CEO **personal use of company
  aircraft** and finds that these firms **underperform market benchmarks by ~4% per year**,
  risk-adjusted — interpreting the perk as a visible symptom of agency problems and weak
  governance. This is the canonical "the corporate jet is a sell signal" result and the
  exact claim we steelman as a long/short.
- **The believers' trade.** If a disclosed personal-jet perk really forecasts
  underperformance, the deployable version is a governance long/short: **long frugal-CEO
  firms, short jet-loving-CEO firms**, and collect the discount. We test whether that
  spread is real, and whether it is *governance* or something else.

## The governance / perks literature (context)

- **Perks as agency cost.** Jensen & Meckling (1976), *Theory of the firm* (JFE) — the
  managerial-perquisite / agency-cost foundation. Rajan & Wulf (2006), *Are perks purely
  managerial excess?* (JFE) — perks can be productivity-enhancing, not just waste, muddying
  a clean "perk = bad" reading. Grinstein, Weinbaum & Yehuda (2017) and Yermack's later
  disclosure work — the post-2006 SEC perquisite-disclosure regime that itemises personal
  aircraft use in the DEF 14A "All Other Compensation" table (our labelling source).
- **Governance and returns, the harder question.** Gompers, Ishii & Metrick (2003),
  *Corporate Governance and Equity Prices* (QJE) — the "G-index" long/short that looked
  like a huge governance premium in-sample; Bebchuk, Cohen & Wang (2013), *Learning and the
  disappearing association between governance and returns* (JFE) — showed that premium
  **decayed to zero out-of-sample once the market learned it**. Direct precedent for our
  finding: a governance sort that looks alpha-generating but is confounded / non-robust.

## Why the significant number here is *beta*, not jets (the confound)

- **Betting-against-beta / low-volatility anomaly.** Frazzini & Pedersen (2014),
  *Betting against beta* (JFE); Baker, Bradley & Wurgler (2011), *Benchmarks as limits to
  arbitrage* (FAJ) — low-beta stocks earn positive CAPM alpha, high-beta stocks negative.
  Our frugal basket is low-beta staples/retail (β ≈ 0.87) and our flyer basket is high-beta
  growth (β ≈ 1.31), so a long-low/short-high book is a **−0.45-beta** portfolio whose CAPM
  intercept is a BAB/low-vol premium — the mechanism behind the lone *t* > 2.
- **Founder-CEO / growth tilt.** Fahlenbrach (2009), *Founder-CEOs, investment decisions,
  and stock market performance* (JFQA) — founder-led firms tilt toward high-growth, high-
  beta profiles. The surviving jet-perk names (Tesla, Alphabet, Meta, Oracle) are precisely
  such founder-growth compounders, so the perk *correlates with* a growth factor unrelated
  to governance quality.

## The method (the shared engine)

- **Long/short characteristic sort + market model.** Sharpe (1964) CAPM / market-model
  alpha; Fama & French (1993) factor framing for "is it alpha or a known premium." We fit
  `LS = α + β·SPY` and read α as the risk-adjusted governance claim.
- **HAC / Newey-West inference.** Newey & West (1987), *A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix* (Econometrica) — the
  Bartlett-kernel long-run variance behind our *t* on the monthly mean and on the OLS α;
  lag rule ``floor(4·(n/100)^(2/9))``. Mirrors [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Survivorship.** Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship bias in
  performance studies* (RFS) — here the bias points *against* the claim (the delisted
  abusers are the missing disasters), so a survivor tape that still fails to short them is a
  conservative test, named on the Signal axis.

## Method lineage (this study's code)

- **Perk table + eligibility.** [`data.JET_FIRMS`](../corporate_jet_index/data.py) and
  [`strategy._eligible_from`](../corporate_jet_index/strategy.py) — a heavy name is
  shortable only from the year after its perk is public (no look-ahead).
- **Long/short construction.** [`strategy.long_short_panel`](../corporate_jet_index/strategy.py)
  and [`strategy.basket_returns`](../corporate_jet_index/strategy.py) — equal-weight
  low − heavy, excess of market.
- **Inference.** [`strategy.hac_tstat`](../corporate_jet_index/strategy.py) (HAC *t* on the
  raw spread) and [`strategy.market_model_alpha`](../corporate_jet_index/strategy.py) (HAC
  *t* on the beta-adjusted α).
- **Costs + borrow.** [`strategy.net_of_costs`](../corporate_jet_index/strategy.py) — one-way
  turnover on both legs + short borrow on the heavy leg; gross and net.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../corporate_jet_index/data.py) plants a known heavy-basket
  discount; the engine must recover it and must not fabricate significance under the null.

## Data sources used here

- **Hardcoded perk table** (`corporate_jet_index.data.JET_FIRMS`): ~24 large-caps (ticker,
  heavy/low, public-year, note), compiled from DEF 14A proxy perquisite tables, SEC
  enforcement, and WSJ / Reuters / Bloomberg / NYT / Forbes coverage. True governance-perk
  panels (ISS, proxy databases) are not free; the labelled table is the transparent
  stand-in. The delisted abusers (`data.DELISTED_ABUSERS`) are named but unpriced.
- **yfinance** monthly total-return closes for each firm + SPY, cached under `_cache/`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: a sibling corporate-governance
  event study (does firing the CEO move the stock?) with the same small-sample / labelled-
  table honesty problems.
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: another hardcoded,
  survivor-biased corporate cross-section where the loudest cases delisted and the survivor
  tape can't certify the folklore.
