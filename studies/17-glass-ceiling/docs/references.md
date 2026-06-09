# References & literature map — Study 17 (Glass-Ceiling)

## The claim under test

- **Koroush AK, *"My Breakout Trading Strategy"*** (X/Twitter long-form, Jan 2026, ~309k views;
  [@KoroushAK](https://x.com/KoroushAK)). The steelmanned setup, mechanized in
  [`glass_ceiling/`](../glass_ceiling): go **long** when price clears a resistance level on **two
  consecutive 1-minute closes**; place the **stop at the swing low**, floored at **1%** of price
  ("if it's less than 1%, use the next swing low down"); take profit at **1R** — the same distance as
  the stop. Trades are graded by three "optimal environment" filters: a slow **staircase** approach
  (not a vertical spike), **building volume**, and a **clean trend** (few crossings of a 30-period
  smoothed moving average, the SMMA). The thesis: in the right environment, breakouts have momentum
  follow-through you can harvest repeatably at 1:1.

- **Why the 1:1 stop/target is the whole game.** A trade that risks 1R to make 1R is a **symmetric
  bracket**. Its per-trade expectancy is exactly ``(2·p − 1) − cost_R`` where ``p`` is the win rate
  and ``cost_R`` the round-trip cost in units of R, so the break-even win rate is ``0.5 + cost_R/2``
  — strictly above a coin flip. The entire case therefore reduces to one measurement: *is the win
  rate reliably above that line?* We make the null exact — a driftless tape, on which the bracket is
  a coin flip by the optional-stopping theorem — and measure how far real breakouts depart from it.

## The relevant market-microstructure evidence

- **Breakouts and short-horizon momentum are weak-to-absent after costs.** The academic record on
  intraday technical breakouts is, charitably, mixed: Sullivan, Timmermann & White, *"Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap"* (**Journal of Finance** 54(5), 1999) show
  that once you correct for the **multiple-testing / data-snooping** induced by searching over many
  rules and parameters (White's Reality Check), the apparent profitability of trading rules on liquid
  indices largely evaporates. Park & Irwin, *"What Do We Know About the Profitability of Technical
  Analysis?"* (**Journal of Economic Surveys** 21(4), 2007) survey the literature and find positive
  results concentrated in early/illiquid samples and fragile to costs and selection — exactly the
  pattern this study reproduces at the single-strategy level.

- **The 1-minute chart is where costs dominate.** Bid-ask bounce (Roll 1984) and the fact that a
  fixed spread is paid on **both** legs of every trade mean transaction costs scale with trade
  frequency, not holding-period return. At a ~1% stop on a 1-minute chart, the round-trip spread is a
  large fraction of R (``cost_R = roundtrip_bps·1e-4 / risk_frac``), so even a few bps moves the
  break-even win rate measurably above 50% — the mechanism behind the `MIRAGE` stamp.

- **The selection illusion in "filtered" setups.** Conditioning trades on after-the-fact-looking
  "quality" filters and then reporting the survivors is a textbook **conditioning-on-the-outcome /
  small-sample** trap: a thinner subset has a noisier win rate, so cherry-picked "A-grade" examples
  routinely show inflated hit rates that vanish out of sample. We test this directly: on the null
  tape the three filters add no win-rate lift beyond their own sampling error while collapsing the
  trade count, and on real tapes the all-filters-pass subset is **1–9 trades**.

## Why the negative result is a *finding*, not a rigged null — the power checks

A test that can only ever say "no" proves nothing. The synthetic core therefore ships tapes where the
answer is **yes** by construction, and the *same* machinery must recover it:

- a **continuation** tape (a small post-breakout drift) on which the win rate rises above 0.5 and net
  expectancy stays positive through realistic costs;
- a **grind-gated** tape where continuation fires **only** after a low-concentration ("staircase")
  approach — built on the *same* grind metric the filter reads — so the staircase filter recovers a
  genuine win-rate lift. The filter works when there is something to find; its failure on the real
  claim is the result, not an artefact.

## Method lineage (the desk's shared engine)

- **Honest interval on a proportion.** The win rate carries a **Wilson score interval**
  ([`strategy.win_rate_ci`](../glass_ceiling/strategy.py)) — well-behaved at moderate ``n`` and
  bounded in [0,1] — because the only question that matters is whether the interval contains 0.5.
- **Cost made concrete.** [`strategy.cost_sweep`](../glass_ceiling/strategy.py) reports net
  expectancy in R across a round-trip-cost grid and locates the break-even, the tradability analogue
  of [`quantlab.backtest.cost_sweep`](../../../quantlab/backtest.py) /
  [`breakeven_cost_bps`](../../../quantlab/backtest.py).
- **Pessimistic intrabar resolution.** When a single bar spans both stop and target, the stop is
  assumed to fill first — the standard conservative assumption, so the strategy is never flattered by
  unknowable within-bar paths.

## Data sources used here

- **Yahoo! Finance intraday OHLCV** (via `yfinance`, `auto_adjust=True`), cached per ticker/interval
  to [`_cache/`](../_cache): **BTC-USD** (Koroush's actual market — crypto trades 24/7, giving the
  deepest real sample), **SPY** and **QQQ**, at the **5-minute** interval over the available ~60-day
  window. **A named limitation, not a detail:** Yahoo serves intraday history only in a short
  trailing window (~7 days at 1-minute, ~60 days at 5-minute), so the real leg is a *small-sample
  sanity check* — the verdict is carried by the synthetic core, where the answer is baked in. A deep
  intraday history would need a paid feed or an MT5 export
  ([`quantlab/brokers/mt5_connector.py`](../../../quantlab/brokers/mt5_connector.py)).

## Related desk studies

- The contrast with **Study 16 — Storm-Shy** (the desk's first `INVESTABLE`) is the lesson in one
  line: Storm-Shy times *risk* (forecastable) and pays; Glass-Ceiling tries to time *direction* at
  1:1 on the noisiest timeframe and doesn't. **Study 11 — Vanishing-Penny** and the desk's other
  cost-and-capacity teardowns share the same killer — an edge that exists on paper and dies on the
  spread.
