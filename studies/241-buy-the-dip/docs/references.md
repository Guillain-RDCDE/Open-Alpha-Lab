# References & literature map — Study 241 (Buy-the-Dip)

## The claim under test

- **"Buy the dip" as folk wisdom.** The phrase is ubiquitous in retail investing
  communities (Reddit's r/investing, financial Twitter/X, popular YouTube channels).
  The claim: systematic deployment of cash into assets after a pullback of X% from
  the recent high will outperform passive holding, because dips represent discounted
  prices. This is a testable hypothesis: we operationalise it as a drawdown-threshold
  rule on SPY (a liquid, low-cost US equity ETF proxy) and measure the CAGR and
  Sharpe ratio against always-invested buy-and-hold.

## The real effect the rule leans on — mean reversion

- **Short-horizon mean reversion.** Jegadeesh (1990), *Evidence of Predictable
  Behavior of Security Returns* (Journal of Finance). Lehmann (1990), *Fads,
  Martingales, and Market Efficiency* (Quarterly Journal of Economics). Both find
  reversal at short (weekly to monthly) horizons. These effects exist — but they
  operate at different timescales and magnitudes than dip-buying at 5–20% drawdowns.
  The dip-buyer's entry advantage (buying below the ATH) is real but small relative
  to the cash-drag cost of waiting.
- **Investor overreaction.** De Bondt & Thaler (1985), *Does the Stock Market
  Overreact?* (Journal of Finance). Long-horizon reversal (3–5 year) supports the
  qualitative notion that extreme losers recover. At the 1–6 month horizon relevant
  to dip-buying, the effect is much weaker.

## The mechanism that kills dip-buying — time in market > timing

- **Time in market vs market timing.** Dalbar QAIB annual reports (various years)
  document that average equity fund investors systematically underperform their own
  funds due to buy-low/sell-high *timing attempts* — they hold cash waiting for
  dips and miss the up days. The dip-buyer is doing the opposite (waiting to buy
  cheap), but faces the same core problem: in a positively drifting market, being
  out of the market costs compounding.
- **Missing the best days.** Putnam Investments (2022), "The cost of being out of
  the market." If an investor misses the 10 best single days in any 20-year period
  of the S&P 500, their terminal wealth is roughly halved versus staying invested.
  Dip-buyers systematically miss upside during the wait.
- **Dollar-cost averaging vs lump sum.** Vanguard (2012), *Dollar-cost averaging
  just means taking risk later* (Vanguard Research). Across 10-year windows in US,
  UK, and Australian equity markets, lump-sum investing outperforms DCA ~66% of the
  time; the median shortfall from DCA is around 2%/yr. Dip-buying is DCA with an
  even worse timing rule: it forces you into cash for longer.

## Why the strategy's logic seems reasonable (the bias it exploits)

- **Representativeness heuristic.** Kahneman & Tversky (1979), *Prospect Theory*
  (Econometrica). A dip "looks" like a discount — the price is lower than it was,
  so buying "at a discount" feels smart. This anchoring to the recent high masks the
  forward-looking truth: the current price is the market's best estimate of fair
  value, not a discounted version of the old price.
- **Anecdotal survivorship.** Famous dip-buy moments (March 2009, March 2020) are
  salient because they worked spectacularly. The many times a dip continued deeper,
  or the market rallied past the threshold without triggering a buy, are less
  memorable. This is classic availability bias.

## Method lineage

- **Drawdown from running maximum.** Standard definition: $(P_t - \max_{s\le t} P_s)
  / \max_{s\le t} P_s$. Used identically in study 99-safety-net (trailing stop)
  and study 97-balancing-act (rebalancing). The running maximum is the natural
  reference for "the recent high" in the folk rule.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica). Applied here to the daily excess return (dip-buyer minus
  buy-and-hold) to test whether the underperformance is distinguishable from zero.
- **DCA control.** Each monthly tranche is 1/N of total capital, where N is the
  total number of investment months. Chosen as the most natural alternative to lump
  sum — an investor who drip-feeds capital at regular intervals regardless of price.

## Related desk studies

- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: periodic rebalancing on
  a 60/40 portfolio — Real/Investable (the only folk rule in this series that
  genuinely wins). The dip-buying strategy is in the same conceptual family
  (tactical allocation based on recent price action) but reaches the opposite verdict.
- **[Study 99 — Safety-Net](../../99-safety-net/)**: trailing-stop loss — the
  mirror image of dip-buying. Trailing stops exit on drawdown; dip-buyers enter.
  Both are typically dominated by passive holding on equity indices.
- **[Study 75 — Knee-Jerk](../../75-knee-jerk/)**: RSI(2) mean-reversion on
  individual stocks — a Real/Fragile signal that does exploit short-term reversals.
  Unlike dip-buying (which requires a large drawdown), RSI(2) fires on 1–5 day
  oversold conditions and earns a measurable per-trade premium. The key difference:
  the cash-drag penalty for waiting is negligible at that timescale.
- **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)**: calendar-based
  entry timing — also None/Mirage. Systematic timing rules of all kinds tend to
  underperform passive holding on broad equity indices.

## Data source

- **Yahoo! Finance daily bars** for SPY (via `yfinance`), adjusted-close, 1993-01-29
  to 2026-06-16 (8,402 trading days, 33.4 years). All headline numbers are pinned
  with an as-of date and content fingerprint (see [`docs/results.md`](results.md)).
  The offline reproducible core and test suite run on the deterministic
  [`data.synthetic_daily`](../buy_the_dip/data.py) generator, never the network.
