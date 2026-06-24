# References & literature map — Study 403 (Hammer & Hanging Man)

## The claim under test

- **The folk recipe.** The hammer and the hanging man are textbook single-candle reversal
  patterns popularised in the West by **Steve Nison**, *Japanese Candlestick Charting
  Techniques* (1991, 2nd ed. 2001) — the book that brought Munehisa Homma's Edo-era rice-
  trading candles to modern markets. The geometry is identical for both: a small real body at
  the **top** of the session range, a **long lower shadow** (≥ ~2× the body), and little or no
  upper shadow. The folklore splits them purely by **prior trend**: the same shape after a
  *downtrend* is a bullish **hammer** ("the market hammered out a bottom" — buy), and after an
  *uptrend* a bearish **hanging man** ("the rally is hanging by a thread" — sell). We steelman
  the strongest version: *a long lower wick marks a forward-looking reversal — a floor after a
  slide, a top after a rally — beyond each name's own base-rate drift, net of costs.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Intraday rejection of lows.** A long lower wick is a real microstructure fact: price
  probed lower during the session and buyers pushed the close back to the top. That *is*
  information about the day. The leap the folklore makes is that this within-day rejection
  *predicts the next several days* — a much stronger claim that the tape has to earn.
- **Short-horizon reversal.** Jegadeesh (1990), *"Evidence of Predictable Behavior of Security
  Returns"* (Journal of Finance), and Lehmann (1990), *"Fads, Martingales, and Market
  Efficiency"* (QJE), document one-week/one-month reversal at the single-stock level — the
  effect a "buy the dip candle" rule hopes to proxy. But these are weak, cost-fragile effects
  measured on *return*, not on a hand-drawn candle shape.
- **The original context.** Homma traded a single, illiquid, manually-cleared rice market;
  generalising a 1700s rice pattern to 2020s S&P large-caps is an out-of-sample stretch the
  data here do not support.

## The failure mode exposed

- **No edge beyond the base rate.** On 11,816 hammer-shaped bars over six decades the bullish
  hammer's best forward edge is +0.05% at 3 days, HAC *t* = 0.92 — under the bar, negative net
  of costs. The pattern carries the day's information but no *forward* floor.
- **Candlestick anomalies vanish under testing.** Marshall, Young & Rose (2006), *"Candlestick
  Technical Trading Strategies: Can They Create Value for Investors?"* (Journal of Banking &
  Finance), test the full candlestick zoo on the DJIA and find **no value** beyond what costs
  and data-snooping explain — the canonical academic verdict this study replicates. Horton
  (2009), *"Stars, Crows, and Doji: The Use of Candlesticks in Stock Selection"* (Quarterly
  Review of Economics and Finance), reaches the same null.
- **Filter-snooping.** The only way to lift the hammer *t* toward 2 is to pick a shorter trend
  lookback; a longer one or a "purer" (longer) wick kills it. Sullivan, Timmermann & White
  (1999), *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"* (Journal of
  Finance), and Brock, Lakonishok & LeBaron (1992), *"Simple Technical Trading Rules…"*
  (Journal of Finance), document exactly how much apparent edge such tuning manufactures.
- **Weak-form efficiency.** Fama (1970), *"Efficient Capital Markets"* (Journal of Finance):
  past price *shapes* alone should not forecast liquid large-cap returns — borne out here.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../hammer_hanging_man/strategy.py) and
  [`quantlab.analytics`](../../../quantlab/analytics.py). HAC is essential here because nearby
  hammer signals share overlapping multi-day forward windows.
- **Label-shuffle / permutation placebo.** The bootstrap-style null behind
  [`strategy.placebo_pvalue`](../hammer_hanging_man/strategy.py); same spirit as the stationary
  bootstrap of Politis & Romano (1994), *"The Stationary Bootstrap"* (JASA).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily OHLC** (via `yfinance`, `auto_adjust=False`), full available history
  (1962→2026) across 26 liquid US large-caps + SPY. The offline reproducible core and the
  notebooks run on cached parquets; the synthetic positive control
  ([`data.synthetic_panel`](../hammer_hanging_man/data.py)) is deterministic and never touches
  the network. Each headline is pinned with an as-of date and a content fingerprint (see
  [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: an overbought/oversold oscillator — same "does a
  technical shape beat a coin / a base rate?" question, same null.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: band mean-reversion,
  the "buy the dip" cousin of the bullish hammer.
- **[Study 186 — Morning-Star](../../186-morning-star/)**: the multi-candle reversal pattern
  from the same Nison candlestick family.
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)**: the rare event study that *does* clear
  the bar — the contrast that shows what a real edge looks like under this harness.
