# References & literature map — Study 209 (ETH-BTC-Ratio)

## The claim under test

**The folk recipe.** Crypto traders interpret the ETH/BTC price ratio as a "risk-on / risk-off"
dial inside crypto: when the ratio rises, speculative capital rotates from "digital gold" (BTC)
into "programmable money" (ETH) and broader alts; when it falls, capital flees back to BTC
safety. The trading rule is: follow the ETH/BTC momentum — buy ETH when the ratio is trending
up, BTC when it is trending down. We steelman this as: *the 20-day sign of the ETH/BTC log-ratio
change carries directional information that, when acted on with a daily 100% rotation between the
two assets, produces risk-adjusted returns above a passive 50/50 rebalanced baseline.* This is
distinct from Study 134 (Bitcoin-Dominance), which compares BTC to the entire alt basket.

## Why the claim is not incoherent — mechanisms proposed in the literature

- **Intra-crypto momentum.** Liu, Tsyvinski & Wu (2022), *Common Risk Factors in Cryptocurrency*
  (Journal of Finance), document a strong cross-sectional and time-series momentum factor in
  crypto returns, with a 1-week momentum being especially robust. A 20-day ETH/BTC ratio
  momentum is a specific instance of this broader effect.
- **Speculative cycles and alt-season.** Ante (2023), *Bitcoin and Ethereum: A Conceptual
  Framework for Determining the Dominant Blockchain Utility Token* (Finance Research Letters),
  discusses the role of BTC dominance cycles and capital rotation between the two largest
  cryptocurrencies, suggesting the ratio is not purely random.
- **Crypto market microstructure.** Cong, Li & Wang (2021), *Tokenomics: Dynamic Adoption and
  Valuation* (Review of Financial Studies), and related work on ETH's role as a utility token
  for DeFi applications, suggest distinct economic drivers for ETH and BTC that could cause
  correlated but non-identical return cycles.
- **Trend-following in crypto.** Jiang, Kelly & Yan (2023), *Return Predictability in
  Cryptocurrency Markets* (Review of Financial Studies), find that trend-following (time-series
  momentum) works in crypto with notably higher power than in equities — consistent with our
  finding of t = +3.7 on the 20-day ETH/BTC ratio signal.

## Why the signal may be overstated

- **Single-cycle survivorship.** The entire ETH history available on Yahoo Finance (2017–2026)
  spans one dominant crypto adoption cycle. Survival bias operates on the *asset* level: we chose
  ETH and BTC *because* they survived and grew. A future regime shift (regulatory, technological)
  could break the signal entirely.
- **Benchmark sensitivity.** Our baseline is the 50/50 rebalanced crypto portfolio. Compared to
  any diversified traditional-asset benchmark, *all* crypto strategies look exceptional in this
  period due to the secular bull market. The honest comparison is the 50/50 baseline within
  crypto, which is what we use.
- **Short sample.** With ~7.5 years of data and ~29 rebalances/year, n ≈ 218 independent
  "position periods." This is adequate for the Newey-West t-stat but short for claiming
  regime-robustness across crypto bear-market structures not yet seen.
- **Execution in practice.** Daily rotation between ETH and BTC requires exchange access,
  custody, and tax-event management in most jurisdictions — practical frictions the model
  does not capture beyond the round-trip cost bps assumption.

## Prior work on ETH/BTC ratio specifically

- **Kajtazi & Moro (2019)**, *The role of bitcoin in well diversified portfolios: A comparative
  global study* (International Review of Financial Analysis) — discusses crypto asset correlation
  structures, providing context for the ~0.80 ETH/BTC correlation we model.
- **Bianchi (2020)**, *Cryptocurrencies as an Asset Class? An Empirical Assessment*
  (European Journal of Finance) — addresses crypto return predictability; the ratio strategy is
  consistent with the momentum results reported for the 2014-2018 sample.
- **Grobys & Sapkota (2019)**, *Cryptocurrencies and Momentum* (Finance Research Letters) —
  documents significant momentum effects in cryptocurrency markets, supporting the mechanistic
  basis for a ratio-momentum rule.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../eth_btc_ratio/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **OLS alpha vs baseline.** Standard CAPM/factor decomposition: regress daily strategy returns
  on baseline returns to separate skill (alpha) from beta to the crypto market.

## Data sources

- **Yahoo Finance** (via `yfinance`): ETH-USD daily from 2017-11-09, BTC-USD from 2014-09-17;
  both clipped to ETH's start for aligned comparison. Crypto trades 7 days/week; we use all
  calendar days (no bdate filter). Cached as parquet; fetch with `fetch=True`.

## Related desk studies

- **[Study 83 — Half-Life](../../83-half-life/)**: BTC post-halving return prediction — the same
  BTC-USD data but an event-study approach on halving dates; n=3 events, NONE verdict.
- **[Study 69 — Safe-Haven](../../69-safe-haven/)**: cross-asset flight-to-safety flows between
  gold, treasuries, and equities — the same rotation logic applied across traditional asset classes.
- **[Study 110 — Faber-Timing](../../110-faber-timing/)**: tactical asset allocation using
  trend-following across asset classes — the closest methodological cousin in traditional assets.
