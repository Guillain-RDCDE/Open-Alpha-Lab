# References & literature map — Study 663 (Hash-Ribbons)

## The claim under test

- **The folklore.** "Hash Ribbons" — Charles Edwards / Capriole Investments, first published
  2019 ("Bitcoin Hash Ribbons: A New Way to Buy Bitcoin", capriole.com and widely syndicated on
  TradingView, Glassnode and crypto Twitter/X ever since). The mechanism as the believers state
  it: Bitcoin's difficulty adjustment forces the least-efficient, most-levered miners offline
  when profitability drops (a "miner capitulation") — those miners are typically forced sellers
  of their BTC treasury to cover costs before shutting down. Once the weakest hands are gone,
  selling pressure eases and the network's remaining (stronger) miners face better economics.
  The indicator: plot the 30-day and 60-day simple moving averages of network hashrate; a
  capitulation is the stretch where the 30-day SMA sits below the 60-day; the **buy signal**
  fires the day the 30-day SMA crosses back above the 60-day — the "ribbons" widening again
  after being compressed.
- **The economic logic, steelmanned.** Hashrate is a real, hard-to-fake commitment of capital
  (ASICs, electricity contracts) — unlike price, it cannot be wash-traded. A genuine, sustained
  hashrate decline is one of the few on-chain signals that requires real economic pain to
  produce, which is why it has intuitive appeal as a "forced capitulation" marker distinct from
  price-based sentiment indicators.
- **What we are NOT testing.** Whether hashrate *growth* generally leads price (that is
  292-bitcoin-hashrate's continuous regression question, answered **NONE**: HAC *t* = −0.05).
  This study is narrower and more literal: the specific, rare, discrete **crossover event** the
  Hash Ribbons indicator is built to flag, and what happens to BTC in the days/months after it.

## What we measure, and the honesty rails

- **The signal.** SMA(30)/SMA(60) crossover on a **daily-interpolated** hashrate path (see the
  data note below), filtered to crossovers preceded by ≥21 days below the long SMA **and** a
  ≥8% peak-to-trough hashrate decline during that stretch — a magnitude filter meant to
  approximate Capriole's own visual "ribbons visibly compressed" criterion and exclude
  single-digit noise blips. This filter is a modeling choice, stated in the open, not fit to
  the answer: it was chosen to separate the two unmistakably large capitulations in the raw
  crossover list (2019, 2021: −20% and −44% declines) from six much smaller wobbles
  (≤ 8%), and it happens to land near the "4-6 historical Hash Ribbons signals" figure
  informally cited across crypto-analytics sites and Capriole's own published charts.
- **Forward-return event study, not a continuous exposure rule.** Because genuine signals are
  rare (n = 4 on this tape), we do not build a smooth "long while ribbons > 0" backtest — that
  question is already answered by 292-bitcoin-hashrate's Hash-Ribbons crossover run (a 3/6-month
  MA version, which converges to buy-and-hold as exposure approaches 100%, i.e. no incremental
  skill). We instead treat each crossover as a **discrete event** and ask what happened in its
  aftermath: forward BTC returns at 30/90/180/365 days, a Welch *t* against the unconditional
  distribution (honestly noted as barely informative at n=4), and a **random-date placebo**
  (20 seeds × 1,000 draws) as the less model-dependent check.
- **One documented execution lag.** Signal known at the crossover day's close; enter BTC at the
  **next** trading day's close (a single `shift`, applied once) — the study's one documented
  convention. Hash Ribbons carries no sell rule in the folklore; the 180-day fixed hold used in
  the timer backtest is our own modeling choice for building a *comparable* exposure backtest,
  not part of the original claim, and is stated as such.
- **Costs charged one-way × NAV per leg** (10 bps), two legs per holding episode (entry + exit).

## Data sources

- **Hashrate.** Curated month-end EH/s table, digitised to round figures from the public
  Blockchain.com 7-day-average hashrate chart
  (https://www.blockchain.com/explorer/charts/hash-rate) — **the identical hardcoded table**
  used by sibling study [292-bitcoin-hashrate](../292-bitcoin-hashrate/), reused rather than
  re-digitised so the two studies agree on the one input fact they share. Linearly interpolated
  to a daily path in [`data.py`](../hash_ribbons/data.py) — **named limitation**: Capriole's own
  ribbon runs on true daily hashrate (much noisier day to day than a straight line between two
  monthly points), so a signal's exact calendar day here is accurate only to roughly the anchor
  month, not the session. The multi-month capitulation/recovery cycles the ribbon targets — the
  2018 bear, the 2020 COVID crash and May halving, the 2021 China mining-ban exodus, the 2022
  FTX shock — are all visible at monthly resolution, so the *episodes* the indicator is built to
  catch survive the smoothing even though the exact day does not.
- **BTC-USD daily close** — yfinance (no key), cached under `_cache/hr_btc_usd.csv`,
  2014-09-17 → 2026-06-30. Price-only == total-return for BTC (no dividends).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [292-bitcoin-hashrate](../292-bitcoin-hashrate/) — tests whether **hashrate growth predicts
  price** as a continuous monthly regression (**NONE**, HAC *t* = −0.05) and separately backtests
  a **3-month/6-month MA "Hash-Ribbons" crossover as a continuous exposure rule** (long whenever
  the ribbon is "on"), finding it wins on absolute return only by being long even more of the
  time than the plain hashrate-rising rule, with **no incremental skill over buy-and-hold**.
  This study is narrower and literal: the canonical **30-day/60-day** crossover, treated as a
  **rare discrete event** (n=4) with its own forward-return event study, placebo and timer
  backtest — a different question (does the *moment* of the signal matter) from 292's continuous
  exposure question (does being long *whenever the ribbon says so* beat holding).
- [221-mayer-multiple](../221-mayer-multiple/) — a **price**-based valuation indicator
  (price / 200-day SMA); no hashrate involved. Busted for the opposite reason (the "cheap" zone
  is a downtrend filter, not a bargain).
- [323-btc-halving](../323-btc-halving/) — the **halving calendar** (fixed, known years ahead);
  no hashrate measurement, a pre-programmed supply-shock date rather than an observed miner
  capitulation.
- [210-crypto-trend](../210-crypto-trend/) — **200-day price SMA** trend-following (Faber-style);
  price only, no hashrate.
- [633-btc-vol-targeting](../633-btc-vol-targeting/) — a continuous **volatility-sizing overlay**
  on BTC; unrelated to any hashrate or capitulation signal.

None of the siblings test the literal Hash-Ribbons 30d/60d hashrate-SMA crossover as a discrete,
rare buy-signal event — this study's own axis.
