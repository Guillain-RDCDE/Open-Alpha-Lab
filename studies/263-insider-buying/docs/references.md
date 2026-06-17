# References & literature map -- Study 263 (Insider-Buying)

## The canonical claim

- **Lakonishok, J. & Lee, I. (2001).** *Are Insider Trades Informative?* Review of
  Financial Studies, 14(1), 79--111. The reference paper on **aggregate** insider
  activity. Finds that insider purchases have modest predictive power for the
  cross-section and for the aggregate market, but the effect is concentrated in
  small firms and operates over **6--12 month** horizons -- not the one-month
  aggregate-timing horizon most folklore versions imply. Insider *selling* is
  largely uninformative (driven by liquidity/diversification/option exercise).

- **Seyhun, H. N. (1998).** *Investment Intelligence from Insider Trading.* MIT
  Press. The book-length treatment. Documents that aggregate net insider buying
  rises near market bottoms and falls near tops, and argues the aggregate buy/sell
  ratio is a contrarian sentiment indicator. The predictive content is weak and
  slow-moving; Seyhun himself stresses the long horizon and the noise.

- **Seyhun, H. N. (1988).** *The Information Content of Aggregate Insider Trading.*
  Journal of Business, 61(1), 1--24. The earliest aggregate-level study: net insider
  trading predicts subsequent market returns with a small, positive, long-horizon
  coefficient.

## Mechanism and why the one-month aggregate version fails

- **Jeng, L. A., Metrick, A. & Zeckhauser, R. (2003).** *Estimating the Returns to
  Insider Trading: A Performance-Evaluation Perspective.* Review of Economics and
  Statistics, 85(2), 453--471. Insider *purchases* earn abnormal returns of
  ~6%/yr over the following six months; *sales* earn nothing abnormal. The edge is
  a slow, firm-level drift -- it does not aggregate into a sharp one-month market
  timing signal, which is exactly what this study finds (HAC *t* = 0.49 next-month).

- **Cohen, L., Malloy, C. & Pomorski, L. (2012).** *Decoding Inside Information.*
  Journal of Finance, 67(3), 1009--1043. Separates "routine" from "opportunistic"
  insider trades; only the opportunistic subset is informative. Aggregate buy/sell
  ratios mix both and so dilute the signal -- a reason the crude aggregate gauge
  underperforms a carefully filtered one.

## The contrarian-timing folklore

- The popular framing -- "insiders are buying, so the bottom is in" -- conflates the
  documented *contemporaneous* tendency of insiders to buy on weakness with a
  *predictive* market-timing rule. This study's contemporaneous correlation
  (ratio vs same-month return = -0.08) confirms the reflex; the next-month
  correlation (+0.04) and regression (t = 0.49) confirm it does not forecast.

## Method lineage

- **HAC / Newey-West t-stat.** Newey, W. K. & West, K. D. (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix.* Econometrica, 55(3), 703--708. Used for the predictive
  regression slope and all return-series t-stats.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the folklore template -- a
  famous "signal" that evaporates against the right benchmark (here, buy-and-hold).
- **[Study 260 -- Margin-Debt](../../260-margin-debt/)**,
  **[Study 261 -- Put-Call-Ratio](../../261-put-call-ratio/)**,
  **[Study 262 -- Short-Interest](../../262-short-interest/)**: sibling
  sentiment/positioning gauges tested as market-timing overlays.

> **Data honesty.** The aggregate buy/sell ratio used here is a **curated proxy**
> reconstructed from documented episodes (2009/2020 buying spikes, 2007/2021 selling),
> not a clean public Form-4 panel. The Signal axis is capped at WEAK for that reason
> alone; the realised statistics put the verdict at NONE.
