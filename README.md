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
teaching are in **[What 210 teardowns taught us](docs/bench.md)**.

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
| **[09](studies/09-phantom-kernel/)** | **Phantom-Kernel** | Is Avellaneda-Stoikov's "optimal spread" built on a real order-arrival law — and does it matter to the P&L? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[10](studies/10-markov-mint/)** | **Markov-Mint** | Can a Markov-chain pipeline "win every trade" on Polymarket? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[11](studies/11-vanishing-penny/)** | **Vanishing-Penny** | How fast does a guaranteed \$40M Polymarket arbitrage close? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[12](studies/12-paper-prophet/)** | **Paper-Prophet** | Does an ARIMA+GARCH stack forecast the SPY, or is it vol-targeting in a trenchcoat? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[13](studies/13-crimson-hour/)** | **Crimson-Hour** | Does a red opening hour + IB-rejection really call the close at 88%? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[14](studies/14-gamma-gospel/)** | **Gamma-Gospel** | Does dealer gamma (GEX) call the day's character, or is it the VIX in a trenchcoat? | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) |
| **[15](studies/15-sigma-sleight/)** | **Sigma-Sleight** | Does length-aware "AdaptiveRSI" beat fixed 70/30, or is the σ-transform a monotone relabel? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[16](studies/16-storm-shy/)** | **Storm-Shy** | Does scaling exposure down when markets get loud actually pay — or is it just selling low? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) |
| **[17](studies/17-glass-ceiling/)** | **Glass-Ceiling** | Does a 1:1 resistance breakout have momentum to harvest, or are you just paying the spread to buy the high? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[18](studies/18-dull-roar/)** | **Dull-Roar** | The low-volatility anomaly: free alpha on the modern S&P 500, or just a defensive low-beta tilt? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[19](studies/19-rubber-band/)** | **Rubber-Band** | Internal Bar Strength: does a stock that closes near its low really snap back the next day — and can you still cash it? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[20](studies/20-freight-train/)** | **Freight-Train** | Time-series momentum: does riding the trend across many markets pay — and why would you hold a thin-Sharpe sleeve? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[21](studies/21-fools-gold/)** | **Fools-Gold** | The "golden cross" (50/200 MA): is it a real buy signal, or a trend filter that only shines on the one index everyone quotes? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[22](studies/22-crystal-ball/)** | **Crystal-Ball** | An HP-filter detrending strategy backtests at Sharpe 2 — on a coin flip. Is it an edge, or is the filter peeking at the future? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[23](studies/23-broken-tether/)** | **Broken-Tether** | Pairs trading: two assets drift apart, bet they snap back — but does the cointegration hold out of sample? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[24](studies/24-stampede/)** | **Stampede** | Cross-sectional momentum: do past winners keep winning on the modern S&P 500 — and what does the crash cost? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[25](studies/25-clean-slate/)** | **Clean-Slate** | Residual momentum — is the "cleaner cousin" of cross-sectional momentum actually cleaner on this tape? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[26](studies/26-sand-castle/)** | **Sand-Castle** | A "mathematically optimal" stat-arb portfolio (w ∝ C⁻¹E) backtests beautifully — does inverting an estimated covariance help, or maximize the error? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[27](studies/27-steamroller/)** | **Steamroller** | The FX carry trade — borrow cheap, lend dear: a real premium, or rent paid in front of a steamroller? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[28](studies/28-carousel/)** | **Carousel** | Sector rotation: does chasing the hottest sectors beat just holding all eleven equal-weight — or is it concentration risk for nothing? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[29](studies/29-hedgers-toll/)** | **Hedgers-Toll** | Are speculators really paid (via CFTC positioning) for taking the other side of producers' commodity hedges — or has the toll booth closed? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[30](studies/30-house-edge/)** | **House-Edge** | Lever a vol-targeted dip-buyer to beat the market — does the edge survive the retail CFD markup? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[31](studies/31-trade-winds/)** | **Trade-Winds** | Cross-asset time-series momentum — the cliché closest to surviving: does our 26-year tape certify it, or only the crisis-alpha sleeve? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[32](studies/32-rip-tide/)** | **Rip-Tide** | Short-term contrarian on the same 18 futures as Trade-Winds — same machine, opposite sign: does fading the move pay net of turnover? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[33](studies/33-slingshot/)** | **Slingshot** | The equity mirror of Rip-Tide — fade each stock against its peers (dollar-neutral): a real reversal, or eaten by the spread? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[34](studies/34-aftershock/)** | **Aftershock** | Post-earnings drift — the decades-old premium: does it survive on a liquid S&P 500 universe after costs? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[35](studies/35-contango/)** | **Contango** | Commodity carry / roll yield — long backwardation, short contango: is the contango tax harvestable, or a −83%-drawdown curve-timing book? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[36](studies/36-greenback/)** | **Greenback** | Dollar-carry, and the carry⊕momentum combo: can diversification fix the steamroller a vol overlay couldn't? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[37](studies/37-barometer/)** | **Barometer** | The trend in macro data (growth, inflation) as a cross-asset signal: right-sided, but big and fast enough to trade? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[38](studies/38-chorus/)** | **Chorus** | The capstone — blend three weak, decorrelated signals (momentum + reversal + low-vol): do they add up to one tradable book? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[39](studies/39-black-box/)** | **Black-Box** | A neural net fed crypto OHLCV scores a dazzling in-sample Sharpe: does it survive walk-forward, or is it memorising noise? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[40](studies/40-paper-tiger/)** | **Paper-Tiger** | A vendor's published dual-momentum backtest, mirrored back: is the timing alpha real, or does it belong to the assets it holds? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[41](studies/41-hangover/)** | **Hangover** | The January Barometer ("as goes January, so goes the year"): a real omen, or a base-rate illusion? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[42](studies/42-last-call/)** | **Last-Call** | The turn-of-the-month effect — real, large and famous: can you actually beat buy-and-hold with it? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[43](studies/43-free-lunch/)** | **Free-Lunch** | Betting against beta — the "low-risk free lunch": is it free, or is it the leverage bill? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[44](studies/44-growth-spurt/)** | **Growth-Spurt** | The asset-growth effect — the highest headline Sharpe on the vendor list: does it survive look-ahead-free on large caps? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[45](studies/45-vanishing-act/)** | **Vanishing-Act** | The size premium (Banz 1981), finance's original factor: is it still there on the longest free proxy? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[46](studies/46-bargain-bin/)** | **Bargain-Bin** | The value premium (HML) on tradable value/growth ETF pairs: a dependable edge, or a regime bet? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[47](studies/47-paper-moon/)** | **Paper-Moon** | The Fed Model (E/P vs the 10-year yield): a real timing rule, or a money-illusion bug? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[48](studies/48-groundhog/)** | **Groundhog** | Return seasonality — a stock repeats its calendar-month performance (Heston-Sadka): real signal, or astrology? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[49](studies/49-black-gold/)** | **Black-Gold** | Does the oil price predict the stock market (Driesprong 2008) — still, out of sample? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[50](studies/50-high-water/)** | **High-Water** | The 52-week-high effect (George-Hwang): a distinct anomaly, or just momentum wearing a hat? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[51](studies/51-blue-chip/)** | **Blue-Chip** | The quality (gross-profitability) premium — the factor that survived where size and value faded (Novy-Marx): does it, on real SEC data? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[52](studies/52-smoke-screen/)** | **Smoke-Screen** | The accruals anomaly (Sloan 1996) — cash-backed earnings beat accrual-heavy ones: does it replicate on real SEC data? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[53](studies/53-jackpot/)** | **Jackpot** | The lottery (MAX) effect — high single-day-gain stocks should underperform (Bali et al.): do they, on large caps? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[54](studies/54-static/)** | **Static** | The idiosyncratic-volatility puzzle (high-vol → low returns, Ang et al.): does it hold on large caps? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[55](studies/55-summer-lull/)** | **Summer-Lull** | "Sell in May" — the Halloween seasonal: real enough to beat buy-and-hold? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[56](studies/56-tide-table/)** | **Tide-Table** | Does the Shiller CAPE forecast long-run returns — and can it time the market, or only set expectations? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[57](studies/57-yield-trap/)** | **Yield-Trap** | Do high-dividend stocks beat the market on total return? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[58](studies/58-bunker/)** | **Bunker** | Does the min-vol ETF (USMV) deliver the low-vol anomaly — free Sharpe, or just less risk? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[59](studies/59-downhill/)** | **Downhill** | Riding the yield curve for the term premium: real, but worth the duration risk? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[60](studies/60-long-shot/)** | **Long-Shot** | The skewness (lottery) effect in commodities — low-skew beats high-skew: does it point the right way, and pay? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[61](studies/61-slow-burn/)** | **Slow-Burn** | Do 3× leveraged ETFs (TQQQ) decay to zero or amplify — and is there a free lunch? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[62](studies/62-premium-seller/)** | **Premium-Seller** | Does a covered-call "income" ETF (QYLD) beat the index it holds? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[63](studies/63-free-fall/)** | **Free-Fall** | The short-volatility carry — selling vol earns a real premium: can you keep it through the steamroller? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[64](studies/64-share-shuffle/)** | **Share-Shuffle** | Do firms that issue stock underperform the ones buying it back (the net-issuance anomaly) — on large caps? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[65](studies/65-scorecard/)** | **Scorecard** | Does Piotroski's 9-point F-score (the famous fundamental-health screen) sort large-cap winners? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[66](studies/66-inverted/)** | **Inverted** | Does an inverted yield curve forecast equity downturns — and is it a usable sell button? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[67](studies/67-fed-drift/)** | **Fed-Drift** | Do stocks drift up before FOMC meetings (Lucca-Moench) — still, after everyone knows? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[68](studies/68-all-weather/)** | **All-Weather** | Does Dalio's risk-parity portfolio beat everything, or just diversify better? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) |
| **[69](studies/69-safe-haven/)** | **Safe-Haven** | Does gold hedge inflation and crashes — or one, or neither? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[70](studies/70-digital-gold/)** | **Digital-Gold** | Is bitcoin "digital gold" — a crash-and-inflation haven, or just a high-octane risk asset? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[71](studies/71-ambush/)** | **Ambush** | Four edges that each died to daily turnover (low IBS, turn-of-month, red close, VIX stress), gated to fire only at their rare confluence: does the overlay pay? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[72](studies/72-loaded-dice/)** | **Loaded-Dice** | The 5-minute SMA(5/10) crossover scalp — "a coin flip with the trend on your side": is the trend really on your side? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[73](studies/73-first-light/)** | **First-Light** | Does the 5-minute opening-range breakout earn the returns its viral 2023 backtest claims? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[74](studies/74-mind-the-gap/)** | **Mind-the-Gap** | Does an opening gap always get filled? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[75](studies/75-knee-jerk/)** | **Knee-Jerk** | Does Connors' RSI(2) oversold bounce still pay — or did publishing it wear it out? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[76](studies/76-rice-paper/)** | **Rice-Paper** | Do Japanese candlestick reversals predict anything a coin doesn't? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[77](studies/77-golden-mean/)** | **Golden-Mean** | Do Fibonacci levels and round numbers hold better than random price levels? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[78](studies/78-crossed-wires/)** | **Crossed-Wires** | Is the MACD crossover any more than a slower coin flip? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[79](studies/79-sleigh-ride/)** | **Sleigh-Ride** | Is the Santa Claus rally real — and does a failed one warn of trouble ahead? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[80](studies/80-cold-open/)** | **Cold-Open** | Does January's direction really call the rest of the year? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[81](studies/81-four-year-itch/)** | **Four-Year-Itch** | Is year three of the presidential cycle really the market's best? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[82](studies/82-witching-hour/)** | **Witching-Hour** | Does triple-witching expiry move the market, or just the volume? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[83](studies/83-half-life/)** | **Half-Life** | Does Bitcoin reliably pump after each four-year halving? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[84](studies/84-moon-math/)** | **Moon-Math** | Does the Stock-to-Flow model actually predict Bitcoin's price? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[85](studies/85-dr-copper/)** | **Dr-Copper** | Does the copper/gold ratio forecast the economy, or just echo it? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[86](studies/86-tail-radar/)** | **Tail-Radar** | Does the CBOE SKEW index see black swans coming? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[87](studies/87-center-line/)** | **Center-Line** | Does price really get pulled back to the intraday VWAP? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[88](studies/88-dogs-of-the-dow/)** | **Dogs-of-the-Dow** | Buy the ten highest-yielding Dow stocks each January — do you really beat the Dow? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[89](studies/89-turn-of-the-month/)** | **Turn-of-the-Month** | Do almost all the market's gains really land in the ~4 days around month-end? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[90](studies/90-weekend/)** | **Weekend** | Are Mondays really negative, and does buying Tuesday beat the market? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[91](studies/91-death-cross/)** | **Death-Cross** | When the 50-day crosses below the 200-day, should you really sell? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[92](studies/92-easy-money/)** | **Easy-Money** | Vol ETPs bleed lower every day — so can you just short them and collect free money? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[93](studies/93-round-numbers/)** | **Round-Numbers** | Prices pile up at whole dollars — but can you trade the magnetism? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[94](studies/94-level-pegging/)** | **Level-Pegging** | Does equal-weighting the S&P *always* beat cap-weighting? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[95](studies/95-holiday-cheer/)** | **Holiday-Cheer** | Do stocks really pop the day before every market holiday — and can you buy it? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[96](studies/96-new-year-pop/)** | **New-Year-Pop** | Do small-cap stocks really pop versus large caps every January? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[97](studies/97-balancing-act/)** | **Balancing-Act** | Is the classic 60/40 portfolio the sensible default — and do bonds really cushion every crash? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) |
| **[98](studies/98-high-noon/)** | **High-Noon** | Is buying at an all-time high really the riskiest moment to invest? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[99](studies/99-safety-net/)** | **Safety-Net** | Does a trailing stop-loss really protect your capital and improve returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[100](studies/100-melting-ice/)** | **Melting-Ice** | Do 3x ETFs like TQQQ really "decay to zero", so you must never hold them overnight? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[101](studies/101-slow-and-steady/)** | **Slow-and-Steady** | Dollar-cost averaging: safer *and* smarter than going all-in, or just a way to stay less invested? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[102](studies/102-free-rebalance/)** | **Free-Rebalance** | Is rebalancing really a free lunch that adds return on top of the risk control? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[103](studies/103-turtle-trader/)** | **Turtle-Trader** | Did the legendary Turtle breakout actually beat entering at random? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[104](studies/104-bollinger-reversion/)** | **Bollinger-Reversion** | Does piercing the lower Bollinger band beat just buying a random day? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[105](studies/105-coppock-curve/)** | **Coppock-Curve** | Can a 1962 indicator built for grief really time bear-market bottoms? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[106](studies/106-supertrend/)** | **Supertrend** | Does TradingView's most-watched ATR flip actually beat a coin? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[107](studies/107-stochastic-oscillator/)** | **Stochastic-Oscillator** | Does the stochastic oversold crossover beat a coin? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[108](studies/108-adx-filter/)** | **ADX-Filter** | Does "only trade when the trend is strong" (ADX > 25) add anything? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[109](studies/109-obv-divergence/)** | **OBV-Divergence** | Does on-balance volume really precede price? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[110](studies/110-faber-timing/)** | **Faber-Timing** | Does the famous 200-day timing rule beat buy-and-hold, or just cut risk? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[111](studies/111-vix-term-structure/)** | **VIX-Term-Structure** | Does the VIX term-structure slope time the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[112](studies/112-move-index/)** | **Move-Index** | Does bond-market volatility (MOVE) warn of equity drawdowns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[113](studies/113-gold-silver-ratio/)** | **Gold-Silver-Ratio** | Does the gold/silver ratio reliably mean-revert into a trade? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[114](studies/114-dollar-smile/)** | **Dollar-Smile** | Does the dollar's direction forecast stocks, or just move with them? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[115](studies/115-credit-spreads/)** | **Credit-Spreads** | Does widening high-yield credit warn equities before they fall? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[116](studies/116-power-hour/)** | **Power-Hour** | Does the last "power hour" continue the day's move — or fade it? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[117](studies/117-pi-cycle-top/)** | **Pi-Cycle-Top** | Does Bitcoin's Pi-Cycle Top indicator really call the top? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[118](studies/118-fed-model/)** | **Fed-Model** | Does the earnings yield minus the 10-year bond yield tell you when to own stocks? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[119](studies/119-real-rate-regime/)** | **Real-Rate-Regime** | Should you step aside when real long rates are high or rising? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[120](studies/120-excess-cape-yield/)** | **Excess-CAPE-Yield** | Does Shiller's ECY forecast the next decade's equity risk premium? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[121](studies/121-magic-formula/)** | **Magic-Formula** | Does Greenblatt's quality-plus-value rank beat the S&P 500? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[122](studies/122-gross-profitability/)** | **Gross-Profitability** | Does Novy-Marx's gross profitability still earn a spread on large caps? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[123](studies/123-altman-z/)** | **Altman-Z** | Do financially distressed (low Altman-Z) stocks pay a premium? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[124](studies/124-cash-flow-yield/)** | **Cash-Flow-Yield** | Does buying high operating-cash-flow yield beat plain earnings yield? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[125](studies/125-ichimoku-cloud/)** | **Ichimoku-Cloud** | Does the Ichimoku cloud system beat a coin? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[126](studies/126-parabolic-sar/)** | **Parabolic-SAR** | Does Wilder's stop-and-reverse flip beat a coin once you pay the whipsaws? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[127](studies/127-williams-r/)** | **Williams-R** | Does the Williams %R oversold signal predict a bounce? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[128](studies/128-keltner-channel/)** | **Keltner-Channel** | Keltner breakout or Keltner reversion — is either one signal? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[129](studies/129-heikin-ashi/)** | **Heikin-Ashi** | Do smoothed Heikin-Ashi candles filter noise into an edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[130](studies/130-vol-risk-premium/)** | **Vol-Risk-Premium** | Is selling the gap between implied and realized vol a free lunch? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[131](studies/131-utilities-canary/)** | **Utilities-Canary** | Do utilities leading the market warn of a coming drawdown? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[132](studies/132-yield-curve-steepener/)** | **Yield-Curve-Steepener** | Does the curve slope tell you when to own the long bond? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[133](studies/133-crypto-seasonality/)** | **Crypto-Seasonality** | Is "Uptober" — or any month — a real edge in Bitcoin? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[134](studies/134-bitcoin-dominance/)** | **Bitcoin-Dominance** | Does falling Bitcoin dominance call "alt season"? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[135](studies/135-fomc-cycle/)** | **FOMC-Cycle** | Do stock returns really hide in the even weeks of the Fed cycle? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[136](studies/136-mark-twain/)** | **Mark-Twain** | Is October really the most dangerous month to own stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[137](studies/137-mansfield-rs/)** | **Mansfield-RS** | Does Weinstein's Stage-2 relative-strength screen beat the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[138](studies/138-random-forest/)** | **Random-Forest** | Does a walk-forward Random Forest on price features beat a shuffled-label control? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[139](studies/139-ai-powered-etf/)** | **AI-Powered-ETF** | Can IBM Watson's AI ETF beat a dirt-cheap S&P 500 index fund? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[140](studies/140-amihud-illiquidity/)** | **Amihud-Illiquidity** | Does sorting on price-impact illiquidity earn a premium, or just survivorship? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[141](studies/141-turnover-anomaly/)** | **Turnover-Anomaly** | Do low-turnover stocks deliver the Datar-Naik-Radcliffe liquidity premium? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[142](studies/142-split-drift/)** | **Split-Drift** | Do stocks drift up after a split takes effect? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[143](studies/143-dividend-capture/)** | **Dividend-Capture** | Can you pocket a dividend by buying just before the ex-date? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[144](studies/144-permanent-portfolio/)** | **Permanent-Portfolio** | Does Harry Browne's four-way mix really survive every regime? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[145](studies/145-home-bias/)** | **Home-Bias** | Does diversifying abroad still improve a US investor's portfolio? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[146](studies/146-country-momentum/)** | **Country-Momentum** | Does rotating into the strongest-trending country ETFs pay? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[147](studies/147-fx-momentum/)** | **FX-Momentum** | Does currency momentum still pay after costs? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[148](studies/148-lunar-effect/)** | **Lunar-Effect** | Do stocks really return less around the full moon? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[149](studies/149-daylight-saving/)** | **Daylight-Saving** | Does the market really slump after the clocks change? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[150](studies/150-sad-effect/)** | **SAD-Effect** | Do shorter autumn days really depress stock returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[151](studies/151-stocks-for-long-run/)** | **Stocks-For-Long-Run** | Do stocks really always win over the long run? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[152](studies/152-inflation-hedge/)** | **Inflation-Hedge** | Are stocks really an inflation hedge? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[153](studies/153-net-operating-assets/)** | **Net-Operating-Assets** | Does a bloated balance sheet predict low returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[154](studies/154-leverage-anomaly/)** | **Leverage-Anomaly** | Do high-leverage firms earn the higher return theory promises? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[155](studies/155-asset-turnover/)** | **Asset-Turnover** | Do capital-efficient (high asset-turnover) firms beat the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[156](studies/156-martingale/)** | **Martingale** | Does averaging down ("martingale") beat just buying and holding? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[157](studies/157-kelly-sizing/)** | **Kelly-Sizing** | Does Kelly-optimal position sizing beat fixed sizing in practice? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[158](studies/158-super-bowl/)** | **Super-Bowl** | Can a football game predict the stock market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[159](studies/159-presidential-party/)** | **Presidential-Party** | Do stocks really do better under Democrats — or did the crashes just land on one party's watch? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[160](studies/160-skyscraper-curse/)** | **Skyscraper-Curse** | Does the world's next record skyscraper signal a crash? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[161](studies/161-year-ending-five/)** | **Year-Ending-Five** | Are years ending in 5 really the best for stocks? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[162](studies/162-rosh-hashanah/)** | **Rosh-Hashanah** | Should you really "sell Rosh Hashanah, buy Yom Kippur"? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[163](studies/163-friday-13th/)** | **Friday-13th** | Is Friday the 13th unlucky for the stock market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[164](studies/164-mercury-retrograde/)** | **Mercury-Retrograde** | Does Mercury retrograde really wreck the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[165](studies/165-chinese-zodiac/)** | **Chinese-Zodiac** | Is the Year of the Dragon lucky for stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[166](studies/166-first-five-days/)** | **First-Five-Days** | Do January's first five days really forecast the whole year? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[167](studies/167-hindenburg-omen/)** | **Hindenburg-Omen** | Does the Hindenburg Omen actually warn of crashes? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[168](studies/168-advance-decline/)** | **Advance-Decline** | Does a fading advance-decline line call the top? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[169](studies/169-fluent-tickers/)** | **Fluent-Tickers** | Do stocks with pronounceable tickers outperform? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[170](studies/170-alphabetical-bias/)** | **Alphabetical-Bias** | Do early-alphabet tickers get an edge? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[171](studies/171-naive-1-over-n/)** | **Naive-1-Over-N** | Does "optimal" portfolio math beat a naive equal split? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[172](studies/172-hundred-minus-age/)** | **Hundred-Minus-Age** | Is "100 minus your age in stocks" smart, or just less invested? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[173](studies/173-four-percent-rule/)** | **Four-Percent-Rule** | Does the 4% retirement withdrawal rule actually survive history? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[174](studies/174-bitcoin-rainbow/)** | **Bitcoin-Rainbow** | Does Bitcoin's Rainbow Chart actually time the cycle? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[175](studies/175-crypto-weekend/)** | **Crypto-Weekend** | Is there a tradable Bitcoin weekend effect? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[176](studies/176-hot-hand/)** | **Hot-Hand** | After a winning streak, is the market "hot" — or "due" for a reversal? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[177](studies/177-megacap-concentration/)** | **Megacap-Concentration** | Does just buying the biggest stocks beat the index? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[178](studies/178-cci/)** | **CCI** | Does Lambert's CCI oscillator time daily equities? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[179](studies/179-aroon/)** | **Aroon** | Does Chande's Aroon(25) crossover point the right way? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[180](studies/180-trix/)** | **TRIX** | Does triple-smoothing an EMA filter false signals, or just delay the good ones? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[181](studies/181-ultimate-oscillator/)** | **Ultimate-Oscillator** | Does Williams' three-period oscillator beat a coin at the extremes? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[182](studies/182-vortex-indicator/)** | **Vortex-Indicator** | Does the VI+/VI- crossover call the daily trend? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[183](studies/183-fisher-transform/)** | **Fisher-Transform** | Does Ehlers' Fisher Transform sharpen turning points into an edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[184](studies/184-williams-fractals/)** | **Williams-Fractals** | Do Bill Williams' fractals mark tradable swing points? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[185](studies/185-chande-momentum/)** | **Chande-Momentum** | Does the Chande Momentum Oscillator beat a coin? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[186](studies/186-morning-star/)** | **Morning-Star** | Does the morning-star candle pattern call a reversal? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[187](studies/187-three-soldiers/)** | **Three-Soldiers** | Do three white soldiers (or black crows) predict what comes next? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[188](studies/188-head-shoulders/)** | **Head-Shoulders** | Does the head-and-shoulders pattern actually forecast the drop? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[189](studies/189-double-top/)** | **Double-Top** | Do double tops and bottoms call reversals? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[190](studies/190-nr7/)** | **NR7** | Does the narrowest range in 7 days precede a tradable breakout? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[191](studies/191-leap-year/)** | **Leap-Year** | Are leap years good or bad for stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[192](studies/192-tax-day/)** | **Tax-Day** | Is there a tax-day effect around April 15? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[193](studies/193-window-dressing/)** | **Window-Dressing** | Do funds pump winners into quarter-end — and can you trade it? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[194](studies/194-turkey/)** | **Turkey** | Do stocks really feast around Thanksgiving? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[195](studies/195-monthly-opex/)** | **Monthly-OpEx** | Does monthly options-expiration week move the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[196](studies/196-long-term-reversal/)** | **Long-Term-Reversal** | Do five-year losers really beat five-year winners (De Bondt-Thaler)? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[197](studies/197-dividend-payout-ratio/)** | **Dividend-Payout-Ratio** | Does a higher payout ratio really predict higher growth (Arnott-Asness)? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[198](studies/198-cash-holdings/)** | **Cash-Holdings** | Do cash-rich firms earn higher returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[199](studies/199-sales-growth/)** | **Sales-Growth** | Do fast-growing "glamour" stocks underperform (Lakonishok-Shleifer-Vishny)? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[200](studies/200-roe-quality/)** | **ROE-Quality** | Does buying high-ROE quality beat the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[201](studies/201-dividend-growth/)** | **Dividend-Growth** | Do dividend growers outperform? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[202](studies/202-fifty-two-week-low/)** | **Fifty-Two-Week-Low** | Does buying stocks near their 52-week low pay? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[203](studies/203-golden-butterfly/)** | **Golden-Butterfly** | Does the Golden Butterfly beat plain stocks — and its simpler parent? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[204](studies/204-talmud-portfolio/)** | **Talmud-Portfolio** | Does a 2,000-year-old "thirds" rule still hold up? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[205](studies/205-three-fund/)** | **Three-Fund** | Is the Bogleheads three-fund portfolio actually better than 100% stocks? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[206](studies/206-dividend-aristocrats/)** | **Dividend-Aristocrats** | Do the Dividend Aristocrats beat the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[207](studies/207-reits-diversifier/)** | **REITs-Diversifier** | Are REITs a real diversifier, or just leveraged stocks? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[208](studies/208-gold-miners/)** | **Gold-Miners** | Are gold miners really leveraged gold? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |
| **[209](studies/209-eth-btc-ratio/)** | **ETH-BTC-Ratio** | Can you ride the ETH/BTC rotation? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |
| **[210](studies/210-crypto-trend/)** | **Crypto-Trend** | Does a 200-day timing rule tame Bitcoin's crashes? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) |

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
