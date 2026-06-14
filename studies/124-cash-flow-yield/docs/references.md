# References & literature map — Study 124 (Cash-Flow-Yield)

## The claim under test

The OCF-yield literature argues that operating cash flow (OCF) is a **harder-to-manipulate
proxy for earnings quality** than reported net income: accrual-heavy earnings can be "managed"
by shifting timing of revenues and expenses, but cash actually in the bank is harder to fake.
The testable form is a standard value sort: rank S&P 500 members by OCF/market-cap (the cash
equivalent of the P/E multiple's inverse), go long the cheap (high-yield) quintile and short
the expensive (low-yield) quintile, hold one year, rebalance. The secondary claim is that OCF
yield has *incremental* predictive power over the simpler earnings yield (NI/market-cap).

## The canonical value-factor literature

- **Fama & French (1992)**, *The Cross-Section of Expected Stock Returns* (Journal of Finance).
  The foundational evidence that "value" — measured by book/market — is a robust equity return
  predictor. B/M is conceptually related to yield metrics as both measure cheapness relative to
  a fundamental anchor.
- **Fama & French (1993)**, *Common Risk Factors in the Returns on Stocks and Bonds* (Journal
  of Financial Economics). The three-factor model; the value factor (HML) is the formal
  systematic version of what a yield sort tries to capture.
- **Lakonishok, Shleifer & Vishny (1994)**, *Contrarian Investment, Extrapolation, and Risk*
  (Journal of Finance). Documents that sorting on cash-flow/price, earnings/price, and
  book/market all produce a value premium in the US; cash-flow/price is one of their central
  signals — the direct ancestor of OCF yield.
- **Asness, Frazzini & Pedersen (2019)**, *Quality Minus Junk* (Review of Accounting Studies).
  Cash-flow quality is one of the "quality" dimensions in QMJ; the study finds that quality
  (partly proxied by cash earnings) predicts returns positively.

## OCF yield and cash-flow quality specifically

- **Sloan (1996)**, *Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About
  Future Earnings?* (Accounting Review). The seminal accruals paper: high accruals (low cash
  backing of earnings) predict poor future returns. This is the mirror image of OCF yield —
  sorting on OCF/price separates high-cash-quality firms (our Q5) from low-quality (Q1).
- **Richardson, Sloan, Soliman & Tuna (2005)**, *Accrual Reliability, Earnings Persistence, and
  Stock Prices* (Journal of Accounting and Economics). Decomposes accruals and finds the cash
  component of earnings is more persistent and better priced.
- **Desai, Rajgopal & Venkatachalam (2004)**, *Value-Glamour and Accruals Mispricing: One
  Anomaly or Two?* (Accounting Review). Asks whether the value premium and the accruals anomaly
  are the same thing in disguise — relevant to our head-to-head OCF vs EY test.

## Why it might be gone

- **McLean & Pontiff (2016)**, *Does Academic Publication Destroy Stock Return Predictability?*
  (Journal of Finance). Documents that ~half of published anomalies decay substantially after
  publication; the accruals / cash-flow quality trade (Sloan 1996) is widely known and widely
  arbed.
- **Green, Hand & Zhang (2013)**, *The Supraview of Return Predictability Research* (Review of
  Accounting Studies). Meta-study of 330+ predictors; most show significant post-publication
  decay or are confined to small-cap / illiquid stocks not present in the S&P 500.
- **Chordia, Subrahmanyam & Tong (2014)**, *Have Capital Market Anomalies Attenuated in the
  Recent Era of High Liquidity and Trading Activity?* (Journal of Accounting and Economics).
  The rapid increase in quant-fund AUM and liquidity since 2000 has materially eroded
  fundamental-factor returns in large-cap equities — consistent with our NONE result.
- **Survivorship bias.** Our sample is the *current* S&P 500 projected backwards, which
  excludes the worst historical outcomes. Finding NONE even in this survivorship-inflated
  sample is a strong negative for the practical case.

## Method lineage (the desk's shared engine)

- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica). The HAC t-stat used in
  `strategy.summary`.
- **Spearman (1904)**, *The Proof and Measurement of Association between Two Things* (American
  Journal of Psychology). The Spearman rank IC, `scipy.stats.spearmanr`, in `strategy.pairwise_ic`.

## Related desk studies

- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: the accruals anomaly (NI − CFO / assets).
  The direct predecessor; OCF yield uses the same EDGAR caches with a market-cap denominator.
- **[Study 65 — Scorecard](../../65-scorecard/)**: the Piotroski F-score includes cash-flow
  quality as one of nine components; a more composite fundamental signal on the same data.
- **[Study 44 — Growth-Spurt](../../44-growth-spurt/)**: revenue growth as a cross-sectional
  predictor — the "momentum" alternative in the fundamental-factor family.
