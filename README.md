<div align="center">

# Open-Alpha-Lab

### An open quant research desk.

I take famous trading ideas — anomalies, folk strategies, things people swear by —
put each through the **same brutal protocol**, and publish the verdict: **edge or mirage.**

***Most are mirages. The honest write-up of why is the point.***

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

> Built by someone who ran the real thing — a fully systematic global-macro book scaled
> from sub-\$100M to **\$9B+ in monthly traded notional** — so every idea is judged on the
> two questions most repos skip: **is the signal real?** *and* **does it survive real
> execution and scale?**

## How it works

Every idea goes through the *same* protocol and earns **two stamps**, so results are comparable:

| | |
|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) |
| **Tradability** — does it survive costs, capacity & scale? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |

Robust inference (Newey-West / Lo SEs, bootstrap CIs, White Reality Check for data-snooping),
an honest alpha-vs-beta split, and a square-root market-impact capacity test — the full house
style is written up in **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## The map

The whole bench on one grid — every study is a numbered chip, sorted by its two stamps.
Almost everything ends up bottom-right; two chips are green — and neither one predicts returns.

![The bench map — every study placed on a Signal × Tradability grid](docs/bench_map.png)

The counts, the mortality by family of idea, and the five lessons the bench keeps
teaching are in **[What 71 teardowns taught us](docs/bench.md)**.

---

## The studies

| # | Study | The claim — tested to destruction | Real? | Tradable? |
|:--:|---|---|:--:|:--:|
| **[01](studies/01-overnight-anomaly/)** | **Overnight Anomaly** | Do stocks really make all their money overnight? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[02](studies/02-falling-knife/)** | **Falling-Knife** | Does buying the dip (−3%) beat a random day? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[03](studies/03-fear-gauge/)** | **Fear-Gauge** | Does buying the VIX spike pay? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[04](studies/04-social-oracle/)** | **Social-Oracle** | Does following a viral crowd's stock picks pay? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[05](studies/05-twin-spread/)** | **Twin-Spread** | Does textbook pairs trading still pay after everyone copied it? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[06](studies/06-clockwork-vol/)** | **Clockwork-Vol** | Does the VIX run on a fixed-period cycle you can time? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[07](studies/07-coiled-spring/)** | **Coiled-Spring** | Does the "20-EMA breakout" deliver explosive +30% pops? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[08](studies/08-true-strength/)** | **True-Strength** | Is the "True" Strength Index truer than MACD/RSI? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[09](studies/09-phantom-kernel/)** | **Phantom-Kernel** | Is Avellaneda-Stoikov's "optimal spread" built on a real arrival law? No: under heavy-tailed reach drawn from the band measured on real books, a per-order likelihood test rejects the exponential kernel by ~0.83 nats per order (Vuong V = +220), and the fitted `k` misprices the spread by ~41% at the study's own horizon — yet swapping that phantom `k` for the textbook value leaves AS's P&L unchanged (Sharpe 1.516 vs 1.514): the part that earns its keep, the inventory skew, never depended on `k`. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[10](studies/10-markov-mint/)** | **Markov-Mint** | Can a Markov-chain pipeline "win every trade" on Polymarket? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[11](studies/11-vanishing-penny/)** | **Vanishing-Penny** | How fast does a guaranteed \$40M Polymarket arbitrage close? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[12](studies/12-paper-prophet/)** | **Paper-Prophet** | Does an ARIMA+GARCH stack forecast the SPY, or is it vol-targeting in a trenchcoat? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[13](studies/13-crimson-hour/)** | **Crimson-Hour** | Does a red opening hour + IB-rejection really call the close at 88%? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[14](studies/14-gamma-gospel/)** | **Gamma-Gospel** | Does dealer gamma (GEX) call the day's character, or is it the VIX in a trenchcoat? | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) |
| **[15](studies/15-sigma-sleight/)** | **Sigma-Sleight** | Does length-aware "AdaptiveRSI" beat fixed 70/30, or is the σ-transform a monotone relabel? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[16](studies/16-storm-shy/)** | **Storm-Shy** | Does scaling exposure down when markets get loud actually pay — or is it just selling low? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) |
| **[17](studies/17-glass-ceiling/)** | **Glass-Ceiling** | Does a 1:1 resistance breakout have momentum to harvest, or are you just paying the spread to buy the high? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[18](studies/18-dull-roar/)** | **Dull-Roar** | The low-volatility anomaly: real in the long-run literature, *inverted* on the modern survivor S&P 500 — the wild decile carried the alpha (+7.2%/yr, *t* +2.3; low-minus-high −5.6%/yr), so the textbook trade meant shorting the decade's winners before its unaffordable borrow even bit. What survives is a low-beta defensive tilt, not free alpha. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[19](studies/19-rubber-band/)** | **Rubber-Band** | Internal Bar Strength: does a stock that closes near its low really snap back the next day — and can you still cash it? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[20](studies/20-freight-train/)** | **Freight-Train** | Time-series momentum: does riding the trend across many markets pay — and why would you hold a thin-Sharpe sleeve? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[21](studies/21-fools-gold/)** | **Fools-Gold** | The "golden cross" (50/200 MA): is it a real buy signal, or a trend filter that only shines on the one index everyone quotes? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[22](studies/22-crystal-ball/)** | **Crystal-Ball** | An HP-filter detrending strategy backtests at Sharpe 2 — on a coin flip. Is it an edge, or is the filter peeking at the future? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[23](studies/23-broken-tether/)** | **Broken-Tether** | Pairs trading: two assets drift apart, bet they snap back — but does the cointegration hold out of sample? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[24](studies/24-stampede/)** | **Stampede** | Cross-sectional momentum: do past winners keep winning on the modern S&P 500 — and what does the crash cost? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[25](studies/25-clean-slate/)** | **Clean-Slate** | Residual momentum, the "cleaner cousin" of Stampede's: the alpha point estimate is higher (+5.0%/yr, *t* 1.0, vs total's +4.4%, *t* 0.9) but a paired bootstrap of the skew and Sharpe gaps clears nothing — *cleaner is unproven on this tape* — and the crash only collapses (−70% → −30%) once vol-management is stacked on top. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[26](studies/26-sand-castle/)** | **Sand-Castle** | A "mathematically optimal" stat-arb portfolio (w ∝ C⁻¹E) backtests beautifully — does inverting an estimated covariance help, or maximize the error? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[27](studies/27-steamroller/)** | **Steamroller** | The FX carry trade: borrow cheap, lend dear, pocket the gap. On the real 2001–2024 G10 tape the premium is there but thin (+0.8%/yr, *t* ≈ 0.9) — and it is rent paid for standing in front of a steamroller: a crash that vol-targeting makes *worse*. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[28](studies/28-carousel/)** | **Carousel** | Sector rotation: does chasing the hottest sectors beat just holding all eleven equal-weight — or is it concentration risk for nothing? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[29](studies/29-hedgers-toll/)** | **Hedgers-Toll** | Are speculators really paid (via CFTC positioning) for taking the other side of producers' commodity hedges — or has the toll booth closed? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[30](studies/30-house-edge/)** | **House-Edge** | Lever a vol-targeted dip-buyer and you beat the market — except it never does: funded flat at the bill it *ties* the index (10.1% vs 10.6%, half the drawdown), and the retail CFD markup — charged on the whole notional, not the borrowed slice — takes 2.65 pts/yr more. Risk control real; the mirage is the house's markup. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[31](studies/31-trade-winds/)** | **Trade-Winds** | Cross-asset time-series momentum: the cliché that comes closest to surviving — a century of literature behind it, but our 26-year tape alone can't certify the headline blend (*t* ≈ 1.6; only the 12-month leg clears at *t* ≈ 3.3). Fragile standalone, but crisis alpha — a 30% sleeve lifts a 60/40's Sharpe and halves its drawdown. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[32](studies/32-rip-tide/)** | **Rip-Tide** | Short-term contrarian (fade the move) on the same 18 futures as Trade-Winds — same machinery, opposite sign. Gross Sharpe +0.25 at *t* = 1.3 — noise, confined to one decade — and daily turnover (break-even 0.79 bp) buries the net. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[33](studies/33-slingshot/)** | **Slingshot** | The equity mirror of Rip-Tide: fade each stock against its peers (dollar-neutral). On the S&P 500 the reversal is *real* (gross Sharpe 0.70) — but break-even 3.31 bp, it lives in the least-liquid names, and it's decayed. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[34](studies/34-aftershock/)** | **Aftershock** | Post-earnings drift: the decades-documented premium shows its textbook rise-then-flatten shape on the real EDGAR tape, but on a liquid S&P 500 universe it's thin — gross Sharpe +0.30 (*t* ≈ 1.4), present but not significant — and its 6 bp break-even sits inside realistic costs. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[35](studies/35-contango/)** | **Contango** | Commodity carry / roll yield: long the backwardated curves, short the contangoed. On the real energy tape (front-month vs laddered ETF pairs) the contango tax is economically enormous — USO bled −76% while USL on the same crude was +4% — yet statistically under the bar (weekly roll spread HAC *t* 1.5–1.8 < 2), and harvesting it is a curve-timing book that's flat (Sharpe +0.16) with −83% drawdowns. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[36](studies/36-greenback/)** | **Greenback** | Dollar-carry + the carry⊕momentum combo: the literature calls FX carry real, but this 23-year G10 sample alone reads weak (Sharpe +0.22, *t* ≈ 1.0, bootstrap CI spanning zero) — and it's a steamroller. The honest fix isn't a vol overlay (Study 27) but diversification: on 2001–2024 the combo cushions the crash exactly as designed, though it can't out-Sharpe carry while FX momentum is losing. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[37](studies/37-barometer/)** | **Barometer** | The trend in macro data (growth, inflation) as a cross-asset signal — long what improving macro momentum favours, tilt to real assets when inflation rises. Right-sided but small and slow: on the real 2007–2025 macro tape every cut lands the predicted way yet none clears the bar (best cut *t* ≈ 0.3), the timed books don't beat a passive hold, and the inflation hedge pays only in the regimes it targets. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[38](studies/38-chorus/)** | **Chorus** | The capstone: blend three weak, decorrelated signals (momentum + reversal + low-vol) into one book. The mechanism is real — the momentum+reversal pair (0.66) beats both parts — but a decorrelated *loser* dilutes, so the naive chorus flattens to ~0, and turnover (break-even 0.02 bp) kills it net. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[39](studies/39-black-box/)** | **Black-Box** | A neural net fed crypto OHLCV scores a dazzling *in-sample* Sharpe (BTC 5.38, 66% accuracy) that collapses to a 51% coin-flip *walk-forward* and goes negative after costs — the in-sample edge was the net memorising noise (shuffled labels score the same). | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[40](studies/40-paper-tiger/)** | **Paper-Tiger** | A vendor's published dual-momentum backtest, mirrored back: the book makes money, but the t≈3.5 belongs to the assets it holds — its timing alpha over its own ingredients is +10 bp/mo at *t* ≈ 0.5 — and the headline that sells it ("beats the market") fails in both Sharpe conventions (excess 0.62 vs SPY's 0.67 and 60/40's 0.68). Its one real gift (−23% vs −51% drawdown) leans on a single 2008 and has decayed since publication. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[41](studies/41-hangover/)** | **Hangover** | The January Barometer ("as goes January, so goes the year"): a base-rate illusion. The rest of the year is up ~76% of the time *regardless*, the omen's 68% directional accuracy is *beaten by always predicting "up"*, and its one faint residue (a weaker year after a down January — real, Fisher p = 0.012) "trades" only by holding less stock, with any post-1972 decay too small to tell from noise. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[42](studies/42-last-call/)** | **Last-Call** | The turn-of-the-month effect: real, large (11 vs 1.9 bp/day, t=5.1), and a trap. Even holding the exact window with the cash leg paid the T-bill, the book makes 5.5%/yr vs buy-and-hold's 10.8% and trails it on excess-of-cash Sharpe (0.41 vs 0.51 gross) — and the premium has *faded* from 11.7 to 0.6 bp/day across 2008 (change *t* ≈ 2.3, modern *t* ≈ 0.1). | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[43](studies/43-free-lunch/)** | **Free-Lunch** | Betting against beta: the "low-risk free lunch" is the leverage bill. Self-financed and measured excess-of-cash like-for-like, the beta-neutral book's *gross* Sharpe (0.32) already trails the market (0.48), it needs 2.77× leverage to be market-neutral, and a realistic financing spread drags the Sharpe to 0.23 (1%) then 0.09 (2.5%). Same lesson as House-Edge — and the glory years were half a levered TLT trade (corr +0.55); the intra-equity book is flat and small (0.23→0.20), so the "decay" was the bond bull ending. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[44](studies/44-growth-spurt/)** | **Growth-Spurt** | The asset-growth effect — the *highest* headline Sharpe on the vendor list (0.835) — rebuilt look-ahead-free on real SEC balance-sheet data for large caps. No measurable premium: the hedge is −3.4%/yr with a 95% CI of [−9.3%, +2.5%], on a survivor panel that censors the short leg's blow-ups and is therefore biased *against* the effect. The premium, where the literature finds it, hides in micro-caps and is subsumed by the investment factor — the illiquid corner where headline Sharpes are made. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[45](studies/45-vanishing-act/)** | **Vanishing-Act** | The size premium (Banz 1981), finance's original factor, on the longest free proxy: never there. Russell 2000 − S&P 500 over 39 years earns +0.03%/yr (Sharpe 0.00, t=0.02), small-caps *trail* large risk-adjusted, and the famous "turn in 2010" is a window choice (bootstrap p = 0.45) on a sample that is *all* post-publication — no decay to mourn, the premium never showed. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[46](studies/46-bargain-bin/)** | **Bargain-Bin** | The value premium (HML) on tradable value/growth ETF pairs: a regime bet, not a dependable edge. Value has *trailed* growth since 2000 (IVE−IVW −1.3%/yr, value Sharpe < growth), strong pre-2007 (a hindsight-framed split) then a brutal 2007–2020 "lost decade" (−5%/yr, Sharpe −0.7). Real but unholdable. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[47](studies/47-paper-moon/)** | **Paper-Moon** | The Fed Model (E/P vs the 10-year yield) on 125 years of Shiller data: a famous timing rule built on a logical bug. It matches buy-and-hold's Sharpe (0.72 vs 0.73) while losing return, and its defining ingredient is *inert* — E/P alone forecasts better (+0.16) than E/P minus the bond yield (+0.12). A real-vs-nominal money illusion (Asness 2003). | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[48](studies/48-groundhog/)** | **Groundhog** | Return seasonality — a stock repeats its calendar-month performance (Heston-Sadka). Sounds like astrology; the *effect* holds up (t=4.1, undecayed, and the other-month control earns −3.3% — same-month-specific). Measured on a survivor panel, so the +7.3%/yr magnitude is an upper bound; counted one-way the book breaks even at ~19 bp (net Sharpe 0.38 at 10 bp, less after borrow). Real but thin. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[49](studies/49-black-gold/)** | **Black-Gold** | Does the oil price predict the stock market (Driesprong 2008)? Not since 2000: on 309 hole-free months of tradable futures data, regressing equity returns on last month's oil gives slope −0.001 (t=−0.03) — *exactly zero* — and the timing rule, even with its cash leg paid the T-bill, trails buy-and-hold (Sharpe 0.32 vs 0.37, excess of T-bill). A documented predictor that vanished out of sample. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[50](studies/50-high-water/)** | **High-Water** | The 52-week-high effect (George-Hwang): momentum wearing a behavioural hat. It's 0.82 correlated with standard 12-2 momentum — not the distinct anomaly it's sold as (the bias-robust half of the verdict) — and on large caps 1996–2026 it showed no premium: the long-short was negative (−8.4%/yr), a sign our survivor panel partly manufactures. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[51](studies/51-blue-chip/)** | **Blue-Chip** | The quality (gross-profitability) premium — the factor that survived where size and value faded (Novy-Marx). On real SEC data it points the right way (high-GP firms beat low-GP by +3.3%/yr, hit 61%) but only glimmers: Sharpe 0.20 (t≈0.8) over the short ~2007+ XBRL window that misses the strong pre-2008 decades. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[52](studies/52-smoke-screen/)** | **Smoke-Screen** | The accruals anomaly (Sloan 1996): firms whose earnings are backed by *cash* beat the *accrual-heavy* ones. Replicates clearly on real SEC data — hedge +5.9%/yr, Sharpe 0.64 (t≈2.7), 72% of years, and the long (cash-backed) leg beats the market. A real quality-of-earnings edge; fragile only on short-side cost and a documented post-2000 fade. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[53](studies/53-jackpot/)** | **Jackpot** | The lottery (MAX) effect — high single-day-gain stocks should underperform (Bali et al.). On large caps it *inverts*: the textbook long-low/short-high trade lost −10.4%/yr (Sharpe −0.49, t −2.5), worsening over time, as high-vol growth survivors led the market. A small-cap effect erased and reversed by survivorship. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[54](studies/54-static/)** | **Static** | The idiosyncratic-volatility puzzle (high-vol → low returns, Ang et al.). Inverts decisively on large caps: the textbook low-vol trade lost −12.1%/yr (Sharpe −0.60, t −3.1), steady across the sample — the volatile growth survivors won. The near-twin of Jackpot (MAX); a small-cap puzzle reversed by survivorship. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[55](studies/55-summer-lull/)** | **Summer-Lull** | "Sell in May": the Halloween seasonal is real and persists for a century (winter +5.3%/yr vs summer +2.7%), but the rule still loses to buy-and-hold (same Sharpe 0.43 vs 0.42, a third less wealth) — because summer is *positive*, so sitting it out just cuts exposure. A durable seasonal, a self-defeating trade. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[56](studies/56-tide-table/)** | **Tide-Table** | The Shiller CAPE genuinely forecasts long-run returns — corr −0.51 with the next-10-year real return (R² 0.28), cheap decades +10.2%/yr vs expensive +4.0%. The valuation signal the Fed Model only pretended to be. But a tide table, not a stopwatch: at 1 year R² is 0.05, so it sets expectations, it doesn't time. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[57](studies/57-yield-trap/)** | **Yield-Trap** | Do high-dividend stocks beat the market? On total return, no: VYM trailed SPY on both return (+9.3% vs +10.8%/yr) and Sharpe (0.67 vs 0.76), with a deeper drawdown. Dividends are payout form, not a premium (Miller-Modigliani); the high-yield screen is a value/sector bet that lagged. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[58](studies/58-bunker/)** | **Bunker** | Does the min-vol ETF (USMV) deliver the low-vol anomaly? Half: it genuinely cuts risk (vol 11.4% vs 14.3%, drawdown −20% vs −24%) but didn't beat the market on Sharpe over 2011–2026 (0.99 vs 1.05), trailing on return. A fine defensive bunker; not the free Sharpe — its edge lives in bear markets this bull sample lacked. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[59](studies/59-downhill/)** | **Downhill** | Riding the yield curve for the term premium. Real — intermediate Treasuries (IEF) beat cash by +2.2%/yr — but poorly paid: the excess earns Sharpe 0.32 vs cash's 1.82, the Sharpe *falls* as you extend duration, and 2022 carved a −23% drawdown. A real premium the risk-adjusted numbers say to leave on the table. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[60](studies/60-long-shot/)** | **Long-Shot** | The skewness (lottery) effect in commodities — low-skew beats high-skew. Unlike the inverted equity version, it points the *right* way and is stable (+5.2%/yr, hit 53%), but a Sharpe of 0.27 (t 1.1) over a 14-ETF basket, thinned by roll/cost (net 0.19 at 10 bp), can only support it, not confirm it. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[61](studies/61-slow-burn/)** | **Slow-Burn** | Do 3× leveraged ETFs (TQQQ) decay or amplify? The volatility drag is *real* (~8–13%/yr, matching 0.5·L·(L−1)·σ²), but "decays to zero" is too glib — TQQQ turned QQQ's +20% into +44%/yr. The myth is "free amplifier": no risk-adjusted gain (Sharpe 0.90 vs 0.98), an −82% drawdown, −79% in 2022. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[62](studies/62-premium-seller/)** | **Premium-Seller** | Does a covered-call "income" ETF (QYLD) beat the index? No — it trailed the very index it holds (QQQ) by −10.9%/yr (Sharpe 0.76 vs 1.06, t −3.6), because it keeps only 50% of up months but takes 58% of down months. The fat distribution is your own gains given back, not a premium. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[63](studies/63-free-fall/)** | **Free-Fall** | The short-volatility carry: selling vol earns a *real* premium (SVXY +12%/yr, +10% post-2018) but is the textbook pennies-in-front-of-a-steamroller — skew −4.8, a −95% drawdown, an −83% day (Volmageddon). Five crash days wiped 95%; the 1× inverse-VIX funds were liquidated. Harvestable only sized & hedged. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[64](studies/64-share-shuffle/)** | **Share-Shuffle** | Do firms that issue stock underperform the ones buying it back (the net-issuance anomaly)? Not on large caps — it *inverts*. On ~412 S&P names from EDGAR filings the textbook long-buyback/short-issuer hedge lost −2.8%/yr (Sharpe −0.30, t −1.3, 28% hit): the heavy issuers (+22.8%) *beat* the buyers (+20.0%), because 2010–25's diluters were the growth winners. The real effect hides in small caps. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[65](studies/65-scorecard/)** | **Scorecard** | Does Piotroski's 9-point F-score (the famous fundamental-health screen) sort large-cap winners? Not here: long high-F / short low-F lost −3.4%/yr (Sharpe −0.34, t −1.4) on ~237 S&P names from EDGAR. It's a *small-cap value* effect — on profitable large-cap survivors the high-score names are crowded and fully priced, so the spread vanishes (and inverts). | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[66](studies/66-inverted/)** | **Inverted** | Does an inverted yield curve forecast equity downturns? Yes, strongly: after the 10y−3m curve inverts, the next 18 months of stocks averaged +1% vs +16% normal, the next 24 months −9% vs +22%. But rare (~5% of months) and slow — markets often climb for a year first (12m gap only −4%). The best recession barometer; a terrible sell button. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[67](studies/67-fed-drift/)** | **Fed-Drift** | Do stocks drift up before FOMC meetings (Lucca-Moench)? They *did*, hugely: 3% of sessions (the pre-FOMC days) earned 11.5% of SPY's whole cumulative return since 1993 — 19.8% pre-2011. Then publication killed it: the pre-FOMC day decayed from +0.24%/day (t 1.7) to +0.09% (t 0.4) after 2011. A real giant, arbitraged away in plain sight. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[68](studies/68-all-weather/)** | **All-Weather** | Does Dalio's risk-parity portfolio beat everything? It *diversifies* for real — unlevered on SPY/IEF/GLD/DBC it earned the best Sharpe (0.92 vs 0.81 for 60/40, 0.65 for SPY) and a drawdown a third of equities' (−17% vs −55%). But it returned only +6.7%/yr, ~half of SPY: the edge is risk-adjusted, and closing the gap needs leverage + the bond bull that 2022 ended. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) |
| **[69](studies/69-safe-haven/)** | **Safe-Haven** | Does gold hedge inflation and crashes? As an inflation hedge: no — YoY gold vs YoY CPI correlate just +0.09, and gold earned the same in high-inflation months (+10.6%) as low (+10.2%). As a crisis hedge: only half-true — flat in equity crashes (−0.2% vs SPY −10%) and uncorrelated, but up in just 50% of them. A real diversifier, oversold as inflation insurance. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[70](studies/70-digital-gold/)** | **Digital-Gold** | Is bitcoin "digital gold"? No: it's ~uncorrelated with gold (+0.09), its correlation with stocks has *risen* (+0.13→+0.36), and in equity crashes it fell harder than stocks (−18% vs −10%), up in 0% of them — the opposite of a haven. It was a phenomenal risk asset (+64%/yr, −83% drawdown), so a small sleeve lifted Sharpe — but as a leveraged growth bet, not a store of value. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[71](studies/71-ambush/)** | **Ambush** | The bench's own inversion, pre-registered: four edges that died *net* to daily turnover (low IBS, turn-of-month, red close, VIX stress) are gated to fire only at their confluence. The premium is real and monotone (+0.6 → +42 bp next-day across 0…4 signals; HAC t=+3.1 at K≥3, RC p=0.009) and *undecayed* since 2015 — and rarity defeats the costs by design (~25% of gross vs >100% traded daily). But at ~15 trades/yr under a 1%/day risk budget the prize is +1.2%/yr excess, OOS Sharpe +0.28 under the frozen 0.30 bar. A real overlay, not a fund. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |

> **Click any study** for the full teardown — two narrative notebooks (one for the curious,
> one for the quant), reproducible code, and every number behind the two stamps.

*In the queue:* the weekend effect, the payday anomaly, paired switching,
a VIX term-structure follow-up. Suggestions welcome via issues.

---

## Run it

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1   |   *nix:  source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                                            # the engine's test-suite
python studies/01-overnight-anomaly/examples/run_synthetic_demo.py   # offline, no network
```

Then open **[studies/01-overnight-anomaly/](studies/01-overnight-anomaly/)** — start with the
notebook *for the curious*, or read the working paper.

To verify the published numbers byte-for-byte (data caches, fingerprints, release bundle), see **[docs/reproducibility.md](docs/reproducibility.md)**.

<details>
<summary><b>The engine — <code>quantlab/</code></b> (a small, tested, reusable toolkit that powers every study)</summary>

<br>

| Module | Role |
|---|---|
| [`decompose.py`](quantlab/decompose.py) | Exact overnight/intraday/close-close return decomposition + Sharpe summary. |
| [`data.py`](quantlab/data.py) | Yahoo fetch + parquet cache; split/total-return/raw adjustment modes. |
| [`diagnostics.py`](quantlab/diagnostics.py) | Critique layer (offline): compounding, split-artefact injector/detector, synthetic markets. |
| [`backtest.py`](quantlab/backtest.py) | Cost-aware backtest, break-even cost, cost sweep. |
| [`stats.py`](quantlab/stats.py) | Bootstrap Sharpe CIs, alpha-vs-beta (gap-risk) decomposition. |
| [`analytics.py`](quantlab/analytics.py) | HAC & Lo (2002) inference, calendar-time normalization, rolling-Sharpe decay, market-impact capacity. |
| [`universe.py`](quantlab/universe.py) | Firm-level cross-section across an index (S&P 500 breadth). |
| [`simulate.py`](quantlab/simulate.py) | Adversarial steelman of a strategy/manipulator P&L vs capital. |
| [`bayes.py`](quantlab/bayes.py) | Bayesian hypothesis posteriors + White (2000) Reality Check. |
| [`plots.py`](quantlab/plots.py) | Decomposition / grid plots. |
| [`repro.py`](quantlab/repro.py) | Reproducibility stamp: pin an as-of date + content fingerprint so headline numbers reproduce. |
| [`brokers/`](quantlab/brokers/) | Swappable `BrokerBase` + MT5 template (`dry_run=True`). |

```text
Open-Alpha-Lab/
├── quantlab/        # the reusable research engine
├── tests/           # deterministic test-suite (CI on 3.10–3.12)
├── studies/         # one folder per study: notebooks, code, data, docs
└── pyproject.toml · CITATION.cff · LICENSE
```

</details>

A [`CITATION.cff`](CITATION.cff) is provided — use GitHub's **"Cite this repository"** button.

---

<div align="center">

*Built by [**Guillain d'Erceville**](https://github.com/Guillain-RDCDE) — production systems,
trading & market-data plumbing, and a habit of publishing the dead-ends, not just the wins.*

**Not investment advice.** Research & education only. See [LICENSE](LICENSE).

</div>
