# References & literature map -- Study 265 (IPO-Volume)

## The canonical claim: hot-issue markets and IPO waves

- **Ibbotson, R. G. & Jaffe, J. F. (1975).** *"Hot Issue" Markets.* Journal of
  Finance, 30(4), 1027--1042. The founding documentation that IPO activity comes
  in waves -- months of heavy new-issue volume cluster together. The source of
  the "the IPO window is open / shut" vocabulary the folklore leans on.

- **Ritter, J. R. (1991).** *The Long-Run Performance of Initial Public Offerings.*
  Journal of Finance, 46(1), 3--27. Shows IPOs underperform over the 3 years after
  listing, and that this underperformance is worst for IPOs from high-volume
  ("hot") years -- firms time their issuance to windows of investor over-optimism.
  This is the academic backbone of "a flood of IPOs marks a top." It is a
  statement about *the IPOs themselves*, not about the aggregate market index a
  year later -- a distinction this study makes central.

- **Lowry, M. (2003).** *Why Does IPO Volume Fluctuate So Much?* Journal of
  Financial Economics, 67(1), 3--40. Links aggregate IPO volume to investor
  sentiment and the demand for capital; finds sentiment is a major driver of the
  *volume* but is coincident with, not leading, broad market conditions.

## IPO volume as a sentiment proxy

- **Baker, M. & Wurgler, J. (2006).** *Investor Sentiment and the Cross-Section of
  Stock Returns.* Journal of Finance, 61(4), 1645--1680. IPO volume (and first-day
  returns) is one of the components of the canonical Baker-Wurgler sentiment index.
  High sentiment predicts low *cross-sectional* returns to speculative stocks --
  but the aggregate-index timing signal from the composite is weak and unstable,
  consistent with the flat slope we find for IPO volume alone.

- **Stambaugh, R. F., Yu, J. & Yuan, Y. (2012).** *The Short of It: Investor
  Sentiment and Anomalies.* Journal of Financial Economics, 104(2), 288--302.
  Sentiment matters mostly through the short leg of cross-sectional anomalies, not
  through a clean market-timing rule -- another reason an aggregate IPO-volume
  market-timing bet disappoints.

## The methodological trap this study is about

- **Contemporaneous vs predictive correlation.** IPO volume is high *during* bull
  years (the +0.32 same-year correlation we report) but carries ~0 information
  about the *next* year. Mistaking a coincident indicator for a leading one is the
  core error the folklore makes.

- **Subsample mining / tiny-n.** With only ~41 forecastable annual observations,
  carving out 12-year windows that "work" is exactly the fishing the desk flags.
  See Harvey, Liu & Zhu (2016), *... and the Cross-Section of Expected Returns*
  (Review of Financial Studies), on the multiple-testing inflation of t-stats.

## Data sources

- **Ritter, J. R.** *Initial Public Offerings: Updated Statistics.* University of
  Florida, Warrington College of Business. https://site.warrington.ufl.edu/ritter/ipo-data/
  -- the canonical annual US IPO-count series (operating companies, ex-SPACs/REITs/
  CEFs/ADRs/units/penny). Hardcoded in `data.py`.
- **Shiller, R. J.** *Online Data -- U.S. Stock Markets 1871-Present.* The monthly
  S&P 500 index level used for calendar-year price returns (`_cache/shiller_sp500.parquet`).

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the canonical tiny-n
  folklore predictor; same n ~ 50 annual-observation reckoning.
- **[Study 223 -- Same-Month Seasonality](../../223-same-month-seasonality/)**: the
  structural sibling this study mirrors (synthetic-control + cached real tape).
