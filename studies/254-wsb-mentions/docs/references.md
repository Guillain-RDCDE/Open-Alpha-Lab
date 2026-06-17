# References & literature map -- Study 254 (WSB-Mentions)

## The claim under test

- **r/WallStreetBets mention trackers (2021- ).** A cottage industry of
  dashboards (ApeWisdom, SwaggyStocks, Quiver Quantitative, utradea) count how
  often each ticker is mentioned on WSB and pitch the count as a leading signal:
  *buy the loudest names before they run.* This study hardcodes a curated monthly
  mention table for a 14-name meme basket (2021-2023) and tests whether this
  month's buzz forecasts next month's return.

## The academic case (mostly: it lags, or it's hype-driven and fragile)

- **Bradley, D., Hanousek Jr., J., Jame, R. & Xiao, Z. (2024).** *Place Your Bets?
  The Value of Investment Research on r/WallStreetBets.* Review of Financial
  Studies. Finds WSB "due diligence" posts have *some* short-horizon
  predictive content but it is concentrated, noisy, and economically marginal --
  not a robust monthly cross-sectional signal.

- **Long, C., Lucey, B., Yarovaya, L. (2021).** *"I just like the stock": The role
  of Reddit sentiment in the GameStop share price.* Finance Research Letters.
  Documents that WSB sentiment co-moves *contemporaneously* with meme prices --
  consistent with mentions being driven *by* the move, the core endogeneity
  problem that sinks any forward-looking use of the count.

- **Eaton, G. W., Green, T. C., Roseman, B. S. & Wu, Y. (2022).** *Retail trader
  sophistication and stock market quality: Evidence from brokerage outages.*
  Journal of Financial Economics. Retail meme flow degrades, not improves, price
  efficiency -- buzz-chasing portfolios buy at the top.

- **Barber, B. M., Huang, X., Odean, T. & Schwarz, C. (2022).** *Attention-Induced
  Trading and Returns: Evidence from Robinhood Users.* Journal of Finance.
  Attention-driven (herding) retail buying predicts *negative* subsequent
  returns -- the "buy the rumor, sell the news" reversal we also test.

- **Da, Z., Engelberg, J. & Gao, P. (2011).** *In Search of Attention.* Journal of
  Finance. The canonical attention-proxy paper (Google search volume): retail
  attention predicts a short-run run-up followed by reversal. A mention count is
  the WSB-era analogue of this attention proxy.

## Why a hype proxy is a treacherous "predictor"

- **Endogeneity / reverse causality.** Mentions spike *because* the price already
  moved. Using a contemporaneous attention shock as a *forward* predictor
  mechanically confounds lead and lag (see Long et al. 2021; Da et al. 2011).

- **Survivorship via delisting.** A meme basket that drops names with no price
  (WISH, BBBY post-bankruptcy) quietly excludes the worst blow-ups from the short
  leg -- inflating any "short the hype" reversal. Shumway (1997), *The Delisting
  Bias in CRSP Data* (Journal of Finance), is the canonical caveat.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the folklore template --
  a hardcoded event table joined to real returns, tested against the correct
  baseline.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  cross-sectional monthly-sort template this study mirrors.
- Other meme/retail-attention studies in the Fun/Folklore family.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), Econometrica.
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).
- **Rank information coefficient (Spearman rank-IC).** Standard cross-sectional
  signal-evaluation metric (Grinold & Kahn, *Active Portfolio Management*).
