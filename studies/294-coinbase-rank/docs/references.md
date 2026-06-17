# References & literature map -- Study 294 (Coinbase-Rank)

## The claim under test

> *When the Coinbase app hits #1 on the Apple App Store, retail FOMO has peaked
> and crypto is about to top.* Sell (or short) when your neighbour is downloading
> Coinbase.

App-Store rank is one of the most-watched real-time proxies for retail crypto
attention. Every major blow-off (December 2017, May 2021, the late-2024 $100k
break) coincided with Coinbase atop the free-apps chart, which is exactly why the
omen is so sticky -- and why a hindsight-selected list of "Coinbase hit #1"
moments is biased *toward* the story.

## Why "attention proxy near tops" is theoretically plausible

- **Barber, B. M. & Odean, T. (2008).** "All That Glitters: The Effect of
  Attention and News on the Buying Behavior of Individual and Institutional
  Investors." *Review of Financial Studies*, 21(2), 785-818. Individual investors
  are net buyers of attention-grabbing assets; attention-driven buying clusters
  near sentiment peaks -- the behavioural engine behind a "retail tops" signal.

- **Da, Z., Engelberg, J. & Gao, P. (2011).** "In Search of Attention." *Journal
  of Finance*, 66(5), 1461-1499. Google-search volume (a sibling of App-Store rank)
  proxies retail attention and predicts short-run price pressure followed by
  reversal -- the same shape the Coinbase-rank omen predicts.

- **Liu, Y., Tsyvinski, A. & Wu, X. (2022).** "Common Risk Factors in
  Cryptocurrency." *Journal of Finance*, 77(2), 1133-1177. Documents strong
  attention/size/momentum effects in crypto; retail attention is a first-order
  driver of crypto returns, lending the omen a real mechanism.

## Why a real-looking effect can still be a mirage

- **Event-study method.** MacKinlay, A. C. (1997). "Event Studies in Economics and
  Finance." *Journal of Economic Literature*, 35(1), 13-39. The canonical reference
  for abnormal returns, the market model `R = a + b*R_mkt + e`, CAR windows, and the
  cross-sectional t on event-day abnormal returns. We use ETH as the market factor
  so a BTC move that was just the whole complex falling is not mistaken for a
  rank-spike effect.

- **Fat tails, tiny n, and the permutation/t gap.** BTC daily vol is ~60-80%
  annualized and the return distribution is extremely fat-tailed. With only 15
  events, a couple of genuine blow-offs (Dec-2017, mid-2021) can manufacture a
  permutation-significant *mean* forward CAR while the **cross-sectional t stays
  under |2|** and the *median* event is near flat. We therefore report both and
  let the t-stat / outlier-drop be decisive -- and they do not clear the bar.

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "... and the Cross-Section of
  Expected Returns." *Review of Financial Studies*, 29(1), 5-68. The relevant
  warning: a *curated* list of the famous spikes is a hindsight-selected sample.
  The appropriate hurdle for a "discovered" effect is higher than t = 2; we report
  the result as an upper bound and lean on the fragility and un-tradability.

## Why "no short" is the expected tradability outcome

- **Efficient near-instant repricing.** App-Store rank is *public*; if it called
  tops, the call would be in the price immediately. Any drift a daily trader could
  short after a one-day lag must be small -- which is what we find.
- **Shorts pay borrow.** BTC's unconditional drift is strongly positive; shorting
  it requires a fee on entry/exit **plus** a daily borrow charge against NAV. We
  charge both. A ~53%-hit, sub-2-sigma short on the strongest-trending major asset
  is negative-expectancy once those frictions are paid.

## Method lineage

- **Market model / abnormal returns.** OLS of BTC on ETH over non-event days
  (`numpy.linalg.lstsq`); abnormal = BTC - (a + b*ETH).
- **Forward CAR.** Sum of abnormal returns over [+1, +5] (the omen is a *forward*
  top, so we measure the days *after* the spike, never the spike day itself).
- **Permutation test.** Reassign the event-day count to random in-sample positions
  10,000 times; the two-sided p-value is the fraction of shuffles with |mean
  forward CAR| >= observed. Distribution-free -- but still mean-driven, so paired
  with a cross-sectional t and an outlier-drop check.
- **Cross-sectional t-test.** `mean / (sd/sqrt(n))` on per-event forward CAR,
  reported raw and after dropping the single most-negative event for robustness.
- **Execution lag, costs & borrow.** One-day lag (short at spike-day close), 20 bps
  one-way on entry and exit against NAV, 5 bps/day borrow over the hold, no leverage.

## Data sources

- **BTC-USD / ETH-USD daily closes.** Yahoo! Finance via `yfinance`, auto-adjusted,
  cached at `_cache/coinbase_rank_btc_daily.parquet`. 2017-11 -> 2026-06,
  close-to-close UTC, price-only = total-return (no dividend).
- **Coinbase rank-spike table.** Hardcoded in `data.py`. Sources: contemporaneous
  reporting (CNBC, Bloomberg, Reuters, The Block, Decrypt) and App Annie / Sensor
  Tower / data.ai chart snapshots reproduced in financial media around each major
  crypto top.

## Related desk studies

- **[Study 291 -- Doge-Tweets](../../291-doge-tweets/)**: the same event-study /
  hardcoded-table machinery applied to Elon Musk's Dogecoin tweets (the structural
  twin of this study).
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the same hindsight-table
  teardown for a folklore predictor.
- **[Study 251 -- Crypto-Reversal](../../251-crypto-reversal/)** and the
  crypto-trend family: other crypto-native tests with the same fat-tail, tiny-n
  caveats.
