# References & literature map — Study 147 (FX-Momentum)

## The claim under test

- **Menkhoff, Sarno, Schmeling & Schrimpf (2012)** — *"Currency Momentum Strategies"*,
  Journal of Financial Economics 106(3), pp. 660–684. The canonical academic paper on FX
  cross-sectional momentum. Using 48 currencies from 1976 to 2010, they document a large
  and significant momentum factor: sorting currencies monthly on their trailing 12-1 month
  return and going long the top and short the bottom decile earns roughly 10%/yr gross with
  a Sharpe above 0.5. Distinct from carry (Study 36 / Study 27): the effect survives
  controlling for interest rate differentials, bid-offer spreads, and transaction costs (at
  institutional rates). The study also documents a momentum crash risk (2008-style reversals)
  and a volatility-scaling improvement.

## Why the original result is coherent

- **Jegadeesh & Titman (1993)** — *"Returns to Buying Winners and Selling Losers"*, Journal
  of Finance 48(1). The foundational equity momentum paper. FX momentum is the direct
  cross-asset analogue: rank on past returns, long winners, short losers.
- **Asness, Moskowitz & Pedersen (2013)** — *"Value and Momentum Everywhere"*, Journal of
  Finance 68(3), pp. 929–985. Confirms cross-sectional momentum in FX alongside equities,
  bonds, and commodities, suggesting a common risk factor. Uses 1978-2011 data for 24
  developed-market currency pairs.
- **Okunev & White (2003)** — *"Do Momentum-Based Strategies Still Work in Foreign Currency
  Markets?"*, Journal of Financial and Quantitative Analysis 38(2). Earlier evidence of FX
  trend momentum in a smaller universe; documents that it was present in the 1980s-1990s.

## Why the edge has decayed in recent data

- **McLean & Pontiff (2016)** — *"Does Academic Research Destroy Stock Return
  Predictability?"*, Journal of Finance 71(1), pp. 5–32. Documents systematic
  post-publication decay of anomalies: returns drop ~32% after working-paper release and
  ~58% after journal publication, consistent with informed arbitrage. FX momentum, published
  as a famous 2012 paper, fits this pattern.
- **Filippou, Gozluklu & Taylor (2018)** — *"Global Political Risk and Currency
  Momentum"*, Journal of Financial and Quantitative Analysis 53(5). Documents that FX
  momentum is particularly exposed to political risk episodes (2014-2018 period), which
  partly explains its post-2012 weakness.
- **Kim, Tse & Wald (2016)** — *"Time Series Momentum and Volatility Scaling"*, Journal
  of Financial Markets. Notes that crowding in momentum strategies (momentum as a 'factor
  risk' rather than anomaly) leads to mean-reversion in extreme drawdowns.

## Relationship to carry and other FX factors

- **Lustig, Roussanov & Verdelhan (2011)** — *"Common Risk Factors in Currency Markets"*,
  Review of Financial Studies 24(11). Defines the "dollar" factor (average return of all
  currencies vs USD) and the "carry" (HML) factor. Currency momentum loads differently on
  these factors than carry does — it is conceptually distinct but can co-vary in crises.
- **Study 36 (Greenback)** and **Study 27 (Steamroller)** in this repo: carry-based
  strategies. Momentum (this study) vs carry: the former bets on recent return persistence,
  the latter on interest rate differentials; they are correlated in trending regimes and
  diverge in sudden reversals.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"*, Econometrica —
  [`strategy.summarize`](../fx_momentum/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *"The Stationary Bootstrap"*,
  JASA 89 — [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **12-1 momentum signal.** Standard in the momentum literature (Jegadeesh & Titman 1993;
  Menkhoff et al. 2012): skip the most recent month to avoid the short-term reversal bias
  documented in Lo & MacKinlay (1990).

## Data sources used here

- **Yahoo Finance G10 FX spot rates** (via `yfinance`): 9 currency pairs vs USD
  (EURUSD, GBPUSD, JPY, AUDUSD, USDCAD, USDCHF, NZDUSD, USDNOK, USDSEK), daily,
  from approximately 2003 to present. Yahoo coverage is less complete than Bloomberg for
  FX — NOK and SEK data in particular may have gaps — and the G10 universe of 9 is smaller
  than the 48-currency universe of Menkhoff et al. (2012). These structural differences
  reduce power relative to the original paper.

## Related desk studies

- **[Study 36 — Greenback](../../36-greenback/)** and
  **[Study 27 — Steamroller](../../27-steamroller/)**: carry-based FX strategies — rank on
  interest differentials rather than past returns; the two strategies sometimes align and
  sometimes offset.
- **[Study 103 — Turtle-Trader](../../103-turtle-trader/)**: a time-series trend rule that
  shares the momentum intuition but operates on a single asset's breakout, not cross-sectional
  ranking.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: equity MA crossover — the single-asset
  momentum family on equities, also showing post-publication decay.
