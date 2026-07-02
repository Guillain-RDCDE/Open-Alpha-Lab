# References & literature map — Study 590 (Sharpe-Hacking)

## The claim, at full strength

The pitch every allocator eventually hears: *"Our Sharpe is 2 — look how smooth the equity curve
is."* A high, smooth Sharpe is the single most persuasive number in asset management, and it is also
the easiest to manufacture. Three pieces of pure financial engineering — **return smoothing**
(reporting stale/illiquid marks), **leverage**, and **volatility targeting** — are routinely credited
with "improving risk-adjusted returns" when at best they change nothing and at worst they only inflate
the *reported* metric. This study makes the trap undeniable by running the transforms on a synthetic
tape we *built* to carry a modest honest Sharpe and **zero genuine edge**, so any Sharpe *gain* is,
by construction, an artefact.

## The Sharpe ratio and how it is (mis)measured

- **Sharpe (1966; 1994)**, *"Mutual Fund Performance"* / *"The Sharpe Ratio."* *Journal of Business* /
  *Journal of Portfolio Management*. The ratio itself, and the standard √T annualisation that assumes
  **iid** returns — the assumption every smoothing game exploits.
- **Lo (2002)**, *"The Statistics of Sharpe Ratios."* *Financial Analysts Journal* 58(4). The core
  result this study leans on: when returns are serially correlated, the naive √q annualisation is
  **wrong**, and the correct factor is η(q) = q / √( q + 2·Σ (q−k)·ρ_k ). Positive autocorrelation
  makes η(q) < √q, deflating an inflated Sharpe back toward the truth. This is the **honest Sharpe**
  used throughout.
- **Getmansky, Lo & Makarov (2004)**, *"An Econometric Model of Serial Correlation and Illiquidity in
  Hedge Fund Returns."* *Journal of Financial Economics* 74(3). The definitive treatment of **return
  smoothing**: illiquid hedge funds mark slowly, so reported returns are a moving average of true
  returns — exactly the AR(1) smoothing modelled here — which mechanically *lowers* measured
  volatility, injects positive autocorrelation, and **inflates the reported Sharpe** without any real
  performance. The empirical spine of this demo.

## Leverage and volatility targeting

- **Leverage invariance of the Sharpe ratio.** A textbook identity: scaling every return by a
  constant L scales both the mean and the standard deviation by L, leaving the Sharpe ratio
  unchanged. Leverage moves you along the capital-market line — more expected return *and* more risk
  in equal measure — it does not improve risk-adjusted return (Sharpe 1964; Tobin 1958,
  separation theorem). The study's null lever.
- **Moreira & Muir (2017)**, *"Volatility-Managed Portfolios."* *Journal of Finance* 72(4). The case
  *for* vol-targeting: scaling exposure down when volatility is high *can* raise the Sharpe — but the
  gain depends entirely on the (weak, unstable) relationship between conditional volatility and future
  returns. On a tape without that relationship (as here), vol-targeting adds nothing and, net of the
  turnover it generates, can lose. The honest counterpoint.
- **Harvey, Hoyle, Rattray, Sargaisson, Balloch & Van Hemert (2018)**, *"The Impact of Volatility
  Targeting."* *Journal of Portfolio Management* 45(1). Vol-targeting mostly helps by taming tail
  risk and drawdowns for assets with a strong negative vol/return link (equities); for many series the
  Sharpe effect is small or negative — matching this study's finding that the lever is fragile, not
  free.

## Metric-gaming and the multiple-comparisons cousins

- **Bailey & López de Prado (2014)**, *"The Deflated Sharpe Ratio."* *Journal of Portfolio Management*
  40(5). Corrects a Sharpe for selection/trial count and for non-normality — the sibling correction
  to the autocorrelation one used here (both say: a raw Sharpe is not what it looks like).
- **Ingersoll, Spiegel, Goetzmann & Welch (2007)**, *"Portfolio Performance Manipulation and
  Manipulation-Proof Performance Measures."* *Review of Financial Studies* 20(5). The formal statement
  that most performance metrics (the Sharpe included) are **manipulable** by dynamic strategies — the
  theoretical charter for this whole study.

## Neighbours on this bench (the dedup map)

- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)** — inflating a Sharpe by
  **searching** (grid search over rules) and haircutting it with the Deflated Sharpe Ratio. Study 590
  inflates the Sharpe by **transforming the reported returns** (smoothing/leverage/vol-target), not by
  searching — a different fake, corrected by the autocorrelation adjustment rather than the trial
  count.
- **[Study 589 — Genetic-Algo-Overfit](../../589-genetic-algo-overfit/)** — the same overfitting-by-
  search trap via an evolutionary optimiser. 590 is the *measurement* cousin: no search at all, just
  three accounting transforms and the honest vs naive Sharpe.

## Shared method

- **The autocorrelation-corrected (honest) Sharpe** — Lo (2002) η(q) factor, the yardstick that
  cannot be gamed by smoothing.
- **Circular-block bootstrap** (Politis & Romano 1992) — confidence bands on the smoothing inflation
  that preserve short-range structure, showing the honest-inflation band straddles zero while the
  naive one is firmly positive.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a synthetic
  control is a machinery proof, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real tape —
  which a synthetic-only demo can never provide), gross/net labelling with costs one-way × NAV, and
  the ≥ 20-seed rule for any synthetic-dependent claim.
