# References & literature map — Study 526 (Intangible-Value)

## The claim under test

- **Intangibles-adjusted value (the believers' / academic version).** Plain **book-to-market**
  (B/M) value investing has decayed because GAAP **expenses** the intangible capital — R&D, brand,
  organisational know-how — that increasingly drives firm value. Reported book equity therefore
  *understates* the true capital of intangible-heavy firms and contaminates the B/M value sort. The
  fix: **capitalise** historical R&D and a share of SG&A into an *intangible-adjusted book* and
  re-run the value sort. A portfolio **long the cheap (high adjusted-B/M) names and short the
  expensive (low adjusted-B/M) names** should harvest a sharper value premium than plain B/M.
- **The academic backbone.** Baruch Lev & Anup Srivastava (2019/2022), *Explaining the Recent
  Failure of Value Investing*, NYU Stern / SSRN working paper (and the *Financial Analysts Journal*
  follow-ups) — argue the post-2007 collapse of the value premium is largely an *accounting*
  artefact of un-capitalised intangibles, and that an intangibles-adjusted book restores much of the
  value spread. This study replicates the **adjustment contrast** (plain B/M vs intangible-adjusted
  B/M) on a clean survivor basket.

## How the intangible capital is built — and the conventions

- **Knowledge capital from R&D.** Baruch Lev & Theodore Sougiannis (1996), *The capitalization,
  amortization, and value-relevance of R&D*, **Journal of Accounting and Economics** 21 — the
  founding empirical case that expensed R&D distorts book values and that a capitalised R&D stock is
  value-relevant. We use a perpetual-inventory R&D stock with 5-year straight-line amortisation.
- **Organisation capital from SG&A.** Andrea Eisfeldt & Dimitris Papanikolaou (2013), *Organization
  Capital and the Cross-Section of Expected Returns*, **Journal of Finance** 68(4) — show that a
  capitalised stock of past SG&A (organisation capital) carries a return premium. The standard
  convention (also in Peters-Taylor) capitalises ~30% of SG&A; we amortise it over 3 years.
- **Total intangible capital and Tobin's Q.** Ryan Peters & Lucian Taylor (2017), *Intangible
  capital and the investment-q relation*, **Journal of Financial Economics** 123(2) — the canonical
  recipe for adding knowledge (R&D) + organisation (SG&A) capital to book, which this study follows.
- **The value premium itself.** Eugene Fama & Kenneth French (1992, 1993), *The Cross-Section of
  Expected Stock Returns* (**JF** 47) and *Common Risk Factors in the Returns on Stocks and Bonds*
  (**JFE** 33) — the book-to-market value factor whose modern decay motivates the whole exercise.
- **Intangibles, broadly.** Baruch Lev (2001), *Intangibles: Management, Measurement, and Reporting*
  (Brookings) — the book-length argument that intangibles break conventional financial statements.

## Inference & honesty (the desk's shared method)

- **HAC (Newey-West) t-stat.** [`strategy.hac_tstat`](../intangible_value/strategy.py) — the
  Signal-axis test on the monthly long-short and long-minus-SPY spreads. Newey & West (1987), *A
  Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, **Econometrica** 55. `REAL` requires `t ≥ 2` on the real tape **and** survival of a
  label-shuffle placebo; a sub-2 *t* with literature support reads `WEAK`.
- **Label-shuffle placebo.** [`strategy.placebo_null`](../intangible_value/strategy.py) permutes the
  cross-sectional signal labels across names and rebuilds the long-short; the real spread must sit in
  the tail. This kills "any split of a heterogeneous field would do it." Here the real spread sits at
  the 85th percentile (p = 0.31) — it does **not** clear the placebo.
- **The adjustment contrast as the third axis.** The head-to-head
  `adjusted − plain` spread ([`strategy.race`](../intangible_value/strategy.py),
  `test_intan_minus_plain`) isolates *what the intangibles correction adds*, independent of whether
  value works at all. This is the cleanest test of the Lev-Srivastava mechanical claim.
- **Multiple testing on a famous factor.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns*, **RFS** 29; McLean & Pontiff (2016), *Does Academic Research Destroy Stock
  Return Predictability?*, **Journal of Finance** 71 — value and its variants have been public for
  decades and shrink out of sample.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../intangible_value/data.py) plants
  a *known* annual value premium via the `edge` knob; at `edge = 0` the long-short must stay
  insignificant (mean *t* = 0.00 over 20 seeds), and a large planted edge must light up
  (mean *t* = +4.23 over 20 seeds). This is a faithful-engine check only — never cited for the tape.
- **One reporting lag + one execution lag, costs, borrow.**
  [`strategy.signal_books`](../intangible_value/strategy.py) forms each month's book from
  fiscal-year-(Y-1) fundamentals and the contemporaneous price (no look-ahead), enters the *next*
  month (one execution lag), charges one-way turnover × NAV at a stated bps, and **charges borrow on
  the short leg** (a long/short pays to be short the expensive growth names).

## Data sources used here

- **SEC EDGAR companyfacts** — annual `StockholdersEquity`, `ResearchAndDevelopmentExpense`,
  `SellingGeneralAndAdministrativeExpense` and shares-outstanding for 40 large-caps (10-K
  full-fiscal-year facts only), 2005–2026 fiscal years, cached under
  `_cache/{equity,rd,sga,shares}.parquet`.
- **yfinance** — monthly total returns and auto-adjusted closes (for the market-cap denominator) for
  the 40 names + SPY, 2005-02 → 2026-05, cached under `_cache/{returns,prices}.parquet`. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 525 — R-And-D-Intensity](../525-r-and-d-intensity/)**: the Chan-Lakonishok-Sougiannis
  R&D/market-cap premium — a sibling intangibles teardown on the *same* survivor-basket machinery.
  This study (526) adds the SG&A organisation-capital layer and frames the question as a
  **book-to-market** adjustment rather than a raw R&D tilt.
- **[Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/)** and
  **[Study 238 — Betting-Against-Beta](../238-betting-against-beta/)**: sibling cross-sectional
  long-short teardowns (HAC t, placebo null, costs + borrow) on the same large-cap survivor basket.
