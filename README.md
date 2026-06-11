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
| **[09](studies/09-phantom-kernel/)** | **Phantom-Kernel** | Is Avellaneda-Stoikov's "optimal spread" built on a real arrival law? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[10](studies/10-markov-mint/)** | **Markov-Mint** | Can a Markov-chain pipeline "win every trade" on Polymarket? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[11](studies/11-vanishing-penny/)** | **Vanishing-Penny** | How fast does a guaranteed \$40M Polymarket arbitrage close? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[12](studies/12-paper-prophet/)** | **Paper-Prophet** | Does an ARIMA+GARCH stack forecast the SPY, or is it vol-targeting in a trenchcoat? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[13](studies/13-crimson-hour/)** | **Crimson-Hour** | Does a red opening hour + IB-rejection really call the close at 88%? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[14](studies/14-gamma-gospel/)** | **Gamma-Gospel** | Does dealer gamma (GEX) call the day's character, or is it the VIX in a trenchcoat? | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) |
| **[15](studies/15-sigma-sleight/)** | **Sigma-Sleight** | Does length-aware "AdaptiveRSI" beat fixed 70/30, or is the σ-transform a monotone relabel? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[16](studies/16-storm-shy/)** | **Storm-Shy** | Does scaling exposure down when markets get loud actually pay — or is it just selling low? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) |
| **[17](studies/17-glass-ceiling/)** | **Glass-Ceiling** | Does a 1:1 resistance breakout have momentum to harvest, or are you just paying the spread to buy the high? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[18](studies/18-dull-roar/)** | **Dull-Roar** | The low-volatility anomaly: do the market's calmest stocks really out-earn its wildest — and can you bank it? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[19](studies/19-rubber-band/)** | **Rubber-Band** | Internal Bar Strength: does a stock that closes near its low really snap back the next day — and can you still cash it? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[20](studies/20-freight-train/)** | **Freight-Train** | Time-series momentum: does riding the trend across many markets pay — and why would you hold a thin-Sharpe sleeve? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[21](studies/21-fools-gold/)** | **Fools-Gold** | The "golden cross" (50/200 MA): is it a real buy signal, or a trend filter that only shines on the one index everyone quotes? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[22](studies/22-crystal-ball/)** | **Crystal-Ball** | An HP-filter detrending strategy backtests at Sharpe 2 — on a coin flip. Is it an edge, or is the filter peeking at the future? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[23](studies/23-broken-tether/)** | **Broken-Tether** | Pairs trading: two assets drift apart, bet they snap back — but does the cointegration hold out of sample? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[24](studies/24-stampede/)** | **Stampede** | Cross-sectional momentum: do past winners keep winning on the modern S&P 500 — and what does the crash cost? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[25](studies/25-clean-slate/)** | **Clean-Slate** | Residual momentum: does stripping out the market tame momentum's crash — and is the cleaner cousin worth more? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[26](studies/26-sand-castle/)** | **Sand-Castle** | A "mathematically optimal" stat-arb portfolio (w ∝ C⁻¹E) backtests beautifully — does inverting an estimated covariance help, or maximize the error? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[27](studies/27-steamroller/)** | **Steamroller** | The FX carry trade: borrow cheap, lend dear, pocket the gap — a real premium, or rent paid for standing in front of a steamroller? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[28](studies/28-carousel/)** | **Carousel** | Sector rotation: does chasing the hottest sectors beat just holding all eleven equal-weight — or is it concentration risk for nothing? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[29](studies/29-hedgers-toll/)** | **Hedgers-Toll** | Are speculators really paid (via CFTC positioning) for taking the other side of producers' commodity hedges — or has the toll booth closed? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[30](studies/30-house-edge/)** | **House-Edge** | Lever a vol-targeted dip-buyer and you beat the market — until you charge leverage its real full-notional financing. Risk control real; return edge a mirage. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[31](studies/31-trade-winds/)** | **Trade-Winds** | Cross-asset time-series momentum: the one cliché that survives. Fragile standalone, but real crisis alpha — a 30% sleeve lifts a 60/40's Sharpe and halves its drawdown. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[32](studies/32-rip-tide/)** | **Rip-Tide** | Short-term contrarian (fade the move) on the same 18 futures as Trade-Winds — same machinery, opposite sign. No gross premium on deep markets, and daily turnover (break-even 0.24 bp) buries the net. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[33](studies/33-slingshot/)** | **Slingshot** | The equity mirror of Rip-Tide: fade each stock against its peers (dollar-neutral). On the S&P 500 the reversal is *real* (gross Sharpe 0.70) — but break-even 3.31 bp, it lives in the least-liquid names, and it's decayed. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[34](studies/34-aftershock/)** | **Aftershock** | Post-earnings drift: a stock keeps drifting in the direction of its earnings surprise for weeks — a real, decades-documented premium that's small and lives in the illiquid names that cost most to trade. Real-tape run pending an earnings-history fetch. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[35](studies/35-contango/)** | **Contango** | Commodity carry / roll yield: long the backwardated curves, short the contangoed. A real, cheap-to-run premium that's volatile and crash-prone. Real-tape run pending the term-structure data the sandbox can't fetch. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[36](studies/36-greenback/)** | **Greenback** | Dollar-carry + the carry⊕momentum combo: FX carry is real but a steamroller, and the honest fix isn't a vol overlay (Study 27) but diversification — blending carry with momentum lifts Sharpe above either leg and cushions the crash. Real G10 tape pending a FRED rates fetch. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[37](studies/37-barometer/)** | **Barometer** | The trend in macro data (growth, inflation) as a cross-asset signal — long what improving macro momentum favours, tilt to real assets when inflation rises. Real but slow and modest; the inflation hedge pays only in the regimes it targets. Real FRED run pending a reliable fetch. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[38](studies/38-chorus/)** | **Chorus** | The capstone: blend three weak, decorrelated signals (momentum + reversal + low-vol) into one book. The mechanism is real — the momentum+reversal pair (0.66) beats both parts — but a decorrelated *loser* dilutes, so the naive chorus flattens to ~0, and turnover (break-even 0.02 bp) kills it net. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[39](studies/39-black-box/)** | **Black-Box** | A neural net fed crypto OHLCV scores a dazzling *in-sample* Sharpe (BTC 5.38, 66% accuracy) that collapses to a 51% coin-flip *walk-forward* and goes negative after costs — the in-sample edge was the net memorising noise (shuffled labels score the same). | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[40](studies/40-paper-tiger/)** | **Paper-Tiger** | A vendor's published dual-momentum backtest, mirrored back: the signal is *real* (net Sharpe 0.74, t≈3.5) but the headline that sells it — "beats the market" — isn't. It ties buy-and-hold, loses to a naïve 60/40, and its one real gift (−23% vs −51% drawdown) leans on a single 2008, has decayed since publication, and lifts no portfolio's Sharpe. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[41](studies/41-hangover/)** | **Hangover** | The January Barometer ("as goes January, so goes the year"): a base-rate illusion. The rest of the year is up ~76% of the time *regardless*, the omen's 68% directional accuracy is *beaten by always predicting "up"*, and its one faint residue (a weaker year after a down January) has decayed since 1972 and "trades" only by holding less stock. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[42](studies/42-last-call/)** | **Last-Call** | The turn-of-the-month effect: real, large (11 vs 1.9 bp/day, t=5.1), and a trap. A window-only book makes 4%/yr vs buy-and-hold's 11% — you forfeit 60% of the return to sit in cash 81% of the time — and the premium has *faded* from 13.8 to 4.8 bp/day since 2008, now an ordinary day's return. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[43](studies/43-free-lunch/)** | **Free-Lunch** | Betting against beta: the "low-risk free lunch" is the leverage bill. On a tradable ETF cross-section the beta-neutral book's *gross* Sharpe (0.47) already trails the market (0.59), it needs 2.78× leverage to be market-neutral, and realistic financing drags the Sharpe to 0.20 (3%) then 0.02 (5%). Same lesson as House-Edge; decayed 0.70→0.28. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[44](studies/44-growth-spurt/)** | **Growth-Spurt** | The asset-growth effect — the *highest* headline Sharpe on the vendor list (0.835) — rebuilt on real SEC balance-sheet data for large caps. It doesn't replicate: the long-low/short-high hedge is −1.6%/yr (Sharpe −0.12), the sign *reversed*. The effect hides in micro-caps and is subsumed by the investment factor — the illiquid corner where headline Sharpes are made. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[45](studies/45-vanishing-act/)** | **Vanishing-Act** | The size premium (Banz 1981), finance's original factor, on the longest free proxy: gone. Russell 2000 − S&P 500 over 39 years earns +0.1%/yr (Sharpe 0.01, t=0.05), small-caps *trail* large risk-adjusted, and every pair flips sign — positive before 2010, negative since. The cleanest post-publication decay on the bench. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[46](studies/46-bargain-bin/)** | **Bargain-Bin** | The value premium (HML) on tradable value/growth ETF pairs: a regime bet, not a dependable edge. Value has *trailed* growth since 2000 (IVE−IVW −1.1%/yr, value Sharpe < growth), strong pre-2007 then a brutal 2007–2020 "lost decade" (−5%/yr, Sharpe −0.7). Real but unholdable. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[47](studies/47-paper-moon/)** | **Paper-Moon** | The Fed Model (E/P vs the 10-year yield) on 125 years of Shiller data: a famous timing rule built on a logical bug. It matches buy-and-hold's Sharpe (0.72 vs 0.73) while losing return, and its defining ingredient is *inert* — E/P alone forecasts better (+0.16) than E/P minus the bond yield (+0.12). A real-vs-nominal money illusion (Asness 2003). | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[48](studies/48-groundhog/)** | **Groundhog** | Return seasonality — a stock repeats its calendar-month performance (Heston-Sadka). Sounds like astrology; it's the bench's *real* one. Long-short +7.3%/yr (Sharpe 0.81, t=4.1), undecayed, and a clean control (other-month history earns −3.3%) confirms it's same-month-specific. Fragile only on turnover/short-book frictions. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[49](studies/49-black-gold/)** | **Black-Gold** | Does the oil price predict the stock market (Driesprong 2008)? Not since 2000: regressing equity returns on last month's oil gives slope +0.017 (t=0.6) — insignificant *and the wrong sign* — and a timing rule trails buy-and-hold (Sharpe 0.26 vs 0.42). A documented predictor that vanished out of sample. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[50](studies/50-high-water/)** | **High-Water** | The 52-week-high effect (George-Hwang): momentum wearing a behavioural hat. On large caps the nearness long-short *lost* (−8.4%/yr, Sharpe −0.40, t=−2.2), and it's 0.87 correlated with plain 12-month momentum — not the distinct anomaly it's sold as, and it didn't even pay. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[51](studies/51-blue-chip/)** | **Blue-Chip** | The quality (gross-profitability) premium — the factor that survived where size and value faded (Novy-Marx). On real SEC data it points the right way (high-GP firms beat low-GP by +3.3%/yr, hit 61%) but only glimmers: Sharpe 0.20 (t≈0.8) over the short ~2007+ XBRL window that misses the strong pre-2008 decades. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[52](studies/52-smoke-screen/)** | **Smoke-Screen** | The accruals anomaly (Sloan 1996): firms whose earnings are backed by *cash* beat the *accrual-heavy* ones. Replicates clearly on real SEC data — hedge +5.9%/yr, Sharpe 0.64 (t≈2.7), 72% of years, and the long (cash-backed) leg beats the market. A real quality-of-earnings edge; fragile only on short-side cost and a documented post-2000 fade. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[53](studies/53-jackpot/)** | **Jackpot** | The lottery (MAX) effect — high single-day-gain stocks should underperform (Bali et al.). On large caps it *inverts*: the textbook long-low/short-high trade lost −10.4%/yr (Sharpe −0.49, t −2.5), worsening over time, as high-vol growth survivors led the market. A small-cap effect erased and reversed by survivorship. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[54](studies/54-static/)** | **Static** | The idiosyncratic-volatility puzzle (high-vol → low returns, Ang et al.). Inverts decisively on large caps: the textbook low-vol trade lost −12.1%/yr (Sharpe −0.60, t −3.1), steady across the sample — the volatile growth survivors won. The near-twin of Jackpot (MAX); a small-cap puzzle reversed by survivorship. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[55](studies/55-summer-lull/)** | **Summer-Lull** | "Sell in May": the Halloween seasonal is real and persists for a century (winter +5.3%/yr vs summer +2.7%), but the rule still loses to buy-and-hold (same Sharpe 0.43 vs 0.42, a third less wealth) — because summer is *positive*, so sitting it out just cuts exposure. A durable seasonal, a self-defeating trade. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[56](studies/56-tide-table/)** | **Tide-Table** | The Shiller CAPE genuinely forecasts long-run returns — corr −0.51 with the next-10-year real return (R² 0.28), cheap decades +10.2%/yr vs expensive +4.0%. The valuation signal the Fed Model only pretended to be. But a tide table, not a stopwatch: at 1 year R² is 0.05, so it sets expectations, it doesn't time. | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[57](studies/57-yield-trap/)** | **Yield-Trap** | Do high-dividend stocks beat the market? On total return, no: VYM trailed SPY on both return (+9.3% vs +10.8%/yr) and Sharpe (0.67 vs 0.76), with a deeper drawdown. Dividends are payout form, not a premium (Miller-Modigliani); the high-yield screen is a value/sector bet that lagged. | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[58](studies/58-bunker/)** | **Bunker** | Does the min-vol ETF (USMV) deliver the low-vol anomaly? Half: it genuinely cuts risk (vol 11.4% vs 14.3%, drawdown −20% vs −24%) but didn't beat the market on Sharpe over 2011–2026 (0.99 vs 1.05), trailing on return. A fine defensive bunker; not the free Sharpe — its edge lives in bear markets this bull sample lacked. | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |

> **Click any study** for the full teardown — two narrative notebooks (one for the curious,
> one for the quant), reproducible code, and every number behind the two stamps.

*In the queue:* the weekend effect, post-earnings drift, a cointegration-gated pairs variant,
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
