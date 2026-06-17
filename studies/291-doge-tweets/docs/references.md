# References & literature map — Study 291 (Doge-Tweets)

## The claim under test

> *Elon Musk's tweets move Dogecoin.* When Musk tweets about DOGE — "Dogecoin is
> the people's crypto", "the Dogefather SNL May 8", swapping the Twitter logo for
> the Shiba Inu — the price spikes. The folk version says you can ride those spikes.

The phenomenon is real enough that it acquired a nickname ("the Musk effect") and
prompted academic study and, eventually, litigation alleging market manipulation.

## Academic / empirical treatment of the Musk-crypto effect

- **Ante, L. (2023).** "How Elon Musk's Twitter activity moves cryptocurrency
  markets." *Technological Forecasting and Social Change*, 186, 122112. Event-study
  evidence that Musk's Bitcoin and Dogecoin tweets are followed by significant
  abnormal returns and volume, with the bulk of the reaction realised within the
  first hours/day — i.e. very fast, consistent with our finding that the move is
  gone by the daily close.

- **Cary, M. (2021).** "Down with the #Dogefather: Evidence of a cryptocurrency
  responding in real time to a crypto-tastemaker." *Journal of Theoretical and
  Applied Electronic Commerce Research*, 16(6). Documents the near-instant DOGE
  reaction to specific Musk tweets, including the SNL ("it's a hustle") sell-off.

- **Huynh, T. L. D. (2022).** "When Elon Musk Changes his Tone, Does Bitcoin
  Adjust? Tone analysis of Musk's tweets and crypto returns." *Finance Research
  Letters.* Sentiment/tone of Musk tweets vs same-day and next-day crypto returns.

## Why "real signal, no trade" is the expected outcome

- **Event-study method.** MacKinlay, A. C. (1997). "Event Studies in Economics and
  Finance." *Journal of Economic Literature*, 35(1), 13–39. The canonical reference
  for abnormal returns, the market model `R = α + β·R_mkt + ε`, AAR/CAR windows,
  and the t-statistic on cross-sectional event-day abnormal returns. We use BTC as
  the market factor so a DOGE move that was just the whole complex rallying is not
  mistaken for a tweet effect.

- **Efficient near-instant repricing.** If a public, observable signal (a tweet to
  millions of followers) moves price, it moves it *immediately*. Any drift a daily
  trader could capture after a one-day lag must therefore be small — which is
  exactly what we find. This is the standard "the news is in the price before you
  are" result, here with a crypto-native, 24/7-market twist.

- **Heavy tails & tiny n.** DOGE daily vol is ~100%+ annualized and the return
  distribution is extremely fat-tailed. A naive cross-sectional t-test on 23 events
  is fragile (one +346% day dominates the variance). We therefore lean on a
  **permutation test** (distribution-free) and an **outlier-robust** t — both of
  which clear the |t| ≥ 2 bar, which the naive t (1.89) just misses.

## Hindsight / data-snooping caveat

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "… and the Cross-Section of Expected
  Returns." *Review of Financial Studies*, 29(1), 5–68. The relevant warning: a
  *curated* list of the famous tweets is a hindsight-selected sample. The
  appropriate hurdle for a "discovered" effect is higher than t = 2; we report this
  as an upper bound and lean on the regime-bound, non-recurring nature of the signal
  on the *tradability* axis.

## Method lineage

- **Market model / abnormal returns.** OLS of DOGE on BTC over non-event days
  (`numpy.linalg.lstsq`), abnormal = DOGE − (α + β·BTC).
- **Permutation test.** Reassign the event-day count to random in-sample positions
  10,000 times; the two-sided p-value is the fraction of shuffles with |mean
  abnormal| ≥ observed. Distribution-free — the right tool for fat-tailed DOGE.
- **Cross-sectional t-test.** `mean / (sd/√n)` on per-event day-0 abnormal returns,
  reported both raw and after dropping the single largest event for robustness.
- **Execution lag & costs.** One-day lag (enter at tweet-day close), 30 bps one-way
  on entry and exit against NAV (round-trip 60 bps), long-only (no borrow).

## Data sources

- **DOGE-USD / BTC-USD daily closes.** Yahoo! Finance via `yfinance`, auto-adjusted,
  cached at `_cache/doge_btc_daily.parquet`. 2019-01 → 2026-06, close-to-close UTC.
- **Musk/Doge tweet table.** Hardcoded in `data.py`. Sources: contemporaneous
  reporting (Reuters, CNBC, Bloomberg, The Verge), Musk's public X/Twitter timeline,
  and the dates referenced in the 2021–2024 DOGE manipulation litigation.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the same event-study /
  hardcoded-table machinery applied to a folklore predictor (the structural twin of
  this study).
- **[Study 251 — Crypto-Reversal](../../251-crypto-reversal/)** and the crypto-trend
  family: other crypto-native tests with the same fat-tail, tiny-n caveats.
