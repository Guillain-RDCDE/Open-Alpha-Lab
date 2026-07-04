# References & literature map — Study 628 (Buffett's Alpha)

## The claim under test

- **The seminal paper.** Andrea Frazzini, David Kabiller & Lasse Heje Pedersen, *Buffett's
  Alpha* (2018, **Financial Analysts Journal** 74(4), 35–55; first NBER/AQR working paper
  2013). On 1976–2011 they find Berkshire Hathaway delivered a Sharpe of **0.76**, roughly
  double the market's, with a significant CAPM alpha (***t* > 3**) at a market beta of only
  **~0.7**. <https://doi.org/10.2469/faj.v74.n4.3>
- **The mechanism.** FKP decompose the alpha into (i) **quality** (profitable, growing,
  well-run stocks — Asness, Frazzini & Pedersen, *Quality Minus Junk*, RAF 2019), (ii)
  **low-beta** exposure (Frazzini & Pedersen, *Betting Against Beta*, JFE 2014), and (iii)
  **~1.7× leverage at below-T-bill cost** via the insurance float. Against MKT+SMB+HML+UMD+
  BAB+QMJ the alpha becomes statistically insignificant — genius, but a *describable* genius.
- **The fade.** FKP's own sub-samples, Berkshire's shareholder letters (Buffett has warned
  for years that size kills outperformance), and the post-publication-decay literature
  (McLean & Pontiff 2016, JF) all predict the alpha shrinks as AUM grows. Our tape: the
  post-2010 alpha change is **−10.98 pp/yr** (HAC *t* = −2.38) and the last 15 years show
  *t* = 0.89.

## What we measure, and the honest deviations from FKP

- **Sample.** FKP use 1976–2011 (CRSP + their 13-F work). Free public data (yfinance) has
  BRK-A from **1980-03**, so our window is 1980-04 → 2026-06 — we *miss 1976–1980* (strong
  Buffett years; our full-sample alpha is, if anything, biased **down** vs FKP) and *add
  2012–2026* (the fade years). On the overlapping era (→ 2011-12) we reproduce their
  headline: alpha +12.36%/yr, HAC *t* = 3.40, beta 0.68.
- **Market total return, spliced and labeled.** SPY adjusted close (total return) from
  1993-02; before that ^GSPC month-end price return **plus the Shiller monthly dividend
  yield** (D/12/P) — the standard way to rebuild a pre-ETF S&P total return from free data.
  Berkshire pays no dividend (since 1967), so its adjusted price *is* total return.
- **Risk-free.** ^IRX (13-week T-bill discount yield) observed at the **previous** month-end,
  /12 — the lockable rate, no look-ahead. All alphas are excess-vs-excess.
- **Factor-lite, not factor-complete.** The academic QMJ/BAB long-short factors are not
  freely available on our stack; we use the **investable ETF proxies** QUAL (2013-07 →) and
  USMV (2011-10 →) and label the attribution accordingly. It can test the *mechanism's
  signature* in the ETF era (BRK loads +0.89 on USMV, *t* = 3.77) but **cannot** re-run FKP's
  full-sample factor regression — we say so instead of pretending.

## Method

- **Newey & West (1987, Econometrica)** — HAC (Bartlett-kernel) standard errors for all
  alpha regressions; rule-of-thumb truncation lag `4·(n/100)^(2/9)`.
- **Selection on success.** Berkshire is studied *because it won* — the max-Sharpe stock among
  long-lived US names (FKP, Table 1). The *t*-stat certifies *this fund's* realized alpha, not
  an ex-ante strategy of picking conglomerates; named on the Signal axis.
- **Sub-period contrast with uncertainty.** "Has it faded?" is tested as one regression with a
  post-2010 dummy (level + interaction) and a HAC *t* on the **difference** — a justified
  split (FKP's sample ends 2011), not a snooped one (METHODOLOGY → the inference bar).
- **Synthetic control.** A seeded (market, manager) world with a planted CAPM alpha and AR(1)
  idiosyncratic noise, averaged over **20 seeds**; the null must stay quiet and the planted
  alpha must be recovered (machinery proof only).

## Data sources used here

- **yfinance** (no key): BRK-A, SPY, ^GSPC, ^IRX, QUAL, USMV — monthly frame cached at
  [`_cache/ba_monthly.csv`](../_cache/ba_monthly.csv), as-of 2026-06-30, fingerprint
  `96d18b13a5da`. <https://finance.yahoo.com/>
- **Shiller long S&P series** (price + dividends, for the pre-1993 market dividend yield):
  Robert Shiller, *Irrational Exuberance* data, via the repo's shared cache.
  <http://www.econ.yale.edu/~shiller/data.htm>
- Frazzini, Kabiller & Pedersen (2018), AQR version: <https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha>

## Related desk studies (the dedup map)

- [238-betting-against-beta](../238-betting-against-beta/) and
  [242-quality-minus-junk](../242-quality-minus-junk/) — the two **factors** FKP use to
  explain Buffett; this study is the **fund-level audit** of the single most famous alpha,
  never before on this bench.
- [264-buffett-indicator](../264-buffett-indicator/) — Buffett's *macro valuation ratio*
  (market cap / GDP), a completely different claim sharing only the name.
- [627-thirteen-f-cloning](../627-thirteen-f-cloning/) — copying managers' 13-F disclosures;
  the practical cousin of "just buy what Buffett buys". Here we test the source itself.
