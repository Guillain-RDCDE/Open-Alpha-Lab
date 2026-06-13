# What 102 teardowns taught us

*One hundred and two famous trading ideas — anomalies, folk strategies, vendor backtests, things
people swear by — each put through the [same protocol](../METHODOLOGY.md) and
stamped twice: **is the signal real?** and **does it survive real execution and
scale?** This page is the view from above. It aggregates; it doesn't re-judge —
every verdict below links back to the study that earned it.*

![The bench map — 102 studies on a Signal × Tradability grid](bench_map.png)

*(Regenerate with `python tools/make_bench_figures.py` — it parses the
[README table](../README.md), so it's always in sync.)*

---

## The score

Of the 102 studies on the bench, 101 carry final stamps (study
[14](../studies/14-gamma-gospel/) is pre-registered, verdict pending):

| | Investable | Fragile | Mirage | |
|---|:--:|:--:|:--:|:--:|
| **Real** | **3** | 10 | 9 | 22 |
| **Weak** | 0 | 14 | 33 | 47 |
| **None** | 0 | 3 | 29 | 32 |
| | **3** | **27** | **71** | **101** |

Read it the way the colours tell you to:

- **22 / 101 signals are statistically real.** About one famous idea in five
  survives autocorrelation-robust inference. The other four-fifths are weak
  (47) or plain noise (32).
- **71 / 101 are mirages once you try to trade them.** Costs, capacity,
  decay, or the discovery that the "edge" was beta all along.
- **Exactly 3 / 101 are investable.** [Storm-Shy](../studies/16-storm-shy/),
  [All-Weather](../studies/68-all-weather/) and [Balancing-Act](../studies/97-balancing-act/)
  — and tellingly, none is a return predictor. One scales exposure down in
  turbulence, the other two are diversification (risk parity and the plain
  60/40). The bench's only greens don't forecast anything.

The single most important cell isn't the green one — it's **Real × Mirage
(9 studies)**. Nine effects that are *genuinely there in the data* and still
can't pay you. That gap between "true" and "tradable" is the bench's whole
thesis, measured.

A quieter cell worth a look: **None × Fragile (3 studies)** — gold
[[69](../studies/69-safe-haven/)] and bitcoin [[70](../studies/70-digital-gold/)]
flunk the *claims* made for them (inflation hedge, digital haven) yet keep a
Fragile stamp as plain diversifiers. The story dies; the asset survives.

---

## Where ideas go to die — mortality by family

We sorted the 102 into rough families. The boundaries are judgement calls (is
the 52-week high a chart pattern or a momentum factor? we said factor) — the
totals below are honest, the taxonomy is approximate.

| Family | Studies | Real | Survived costs* | Investable |
|---|:--:|:--:|:--:|:--:|
| Equity factors & fundamentals — [18](../studies/18-dull-roar/) [34](../studies/34-aftershock/) [38](../studies/38-chorus/) [43](../studies/43-free-lunch/) [44](../studies/44-growth-spurt/) [45](../studies/45-vanishing-act/) [46](../studies/46-bargain-bin/) [50](../studies/50-high-water/) [51](../studies/51-blue-chip/) [52](../studies/52-smoke-screen/) [53](../studies/53-jackpot/) [54](../studies/54-static/) [57](../studies/57-yield-trap/) [58](../studies/58-bunker/) [64](../studies/64-share-shuffle/) [65](../studies/65-scorecard/) [88](../studies/88-dogs-of-the-dow/) [94](../studies/94-level-pegging/) | 18 | 1 | 4 | 0 |
| Vol, hedges & allocation — [03](../studies/03-fear-gauge/) [06](../studies/06-clockwork-vol/) [16](../studies/16-storm-shy/) [30](../studies/30-house-edge/) [61](../studies/61-slow-burn/) [62](../studies/62-premium-seller/) [63](../studies/63-free-fall/) [68](../studies/68-all-weather/) [69](../studies/69-safe-haven/) [70](../studies/70-digital-gold/) [83](../studies/83-half-life/) [86](../studies/86-tail-radar/) [92](../studies/92-easy-money/) [97](../studies/97-balancing-act/) [100](../studies/100-melting-ice/) [101](../studies/101-slow-and-steady/) [102](../studies/102-free-rebalance/) | 17 | 8 | 7 | **3** |
| Technical & chart patterns — [02](../studies/02-falling-knife/) [07](../studies/07-coiled-spring/) [08](../studies/08-true-strength/) [13](../studies/13-crimson-hour/) [15](../studies/15-sigma-sleight/) [17](../studies/17-glass-ceiling/) [19](../studies/19-rubber-band/) [21](../studies/21-fools-gold/) [22](../studies/22-crystal-ball/) [72](../studies/72-loaded-dice/) [73](../studies/73-first-light/) [74](../studies/74-mind-the-gap/) [75](../studies/75-knee-jerk/) [76](../studies/76-rice-paper/) [77](../studies/77-golden-mean/) [78](../studies/78-crossed-wires/) [87](../studies/87-center-line/) [91](../studies/91-death-cross/) [93](../studies/93-round-numbers/) [98](../studies/98-high-noon/) [99](../studies/99-safety-net/) | 21 | 2 | 2 | 0 |
| Momentum & trend — [20](../studies/20-freight-train/) [24](../studies/24-stampede/) [25](../studies/25-clean-slate/) [28](../studies/28-carousel/) [31](../studies/31-trade-winds/) [40](../studies/40-paper-tiger/) | 6 | 0 | 5 | 0 |
| Carry, curves & commodities — [27](../studies/27-steamroller/) [29](../studies/29-hedgers-toll/) [35](../studies/35-contango/) [36](../studies/36-greenback/) [59](../studies/59-downhill/) [60](../studies/60-long-shot/) | 6 | 1 | 4 | 0 |
| Calendar & seasonal — [01](../studies/01-overnight-anomaly/) [41](../studies/41-hangover/) [42](../studies/42-last-call/) [48](../studies/48-groundhog/) [55](../studies/55-summer-lull/) [67](../studies/67-fed-drift/) [79](../studies/79-sleigh-ride/) [80](../studies/80-cold-open/) [81](../studies/81-four-year-itch/) [82](../studies/82-witching-hour/) [89](../studies/89-turn-of-the-month/) [90](../studies/90-weekend/) [95](../studies/95-holiday-cheer/) [96](../studies/96-new-year-pop/) | 14 | 4 | 3 | 0 |
| Mean reversion & stat-arb — [05](../studies/05-twin-spread/) [23](../studies/23-broken-tether/) [26](../studies/26-sand-castle/) [32](../studies/32-rip-tide/) [33](../studies/33-slingshot/) [71](../studies/71-ambush/) | 6 | 3 | 1 | 0 |
| Macro & valuation timing — [37](../studies/37-barometer/) [47](../studies/47-paper-moon/) [49](../studies/49-black-gold/) [56](../studies/56-tide-table/) [66](../studies/66-inverted/) [85](../studies/85-dr-copper/) | 6 | 2 | 3 | 0 |
| ML & model forecasting — [10](../studies/10-markov-mint/) [12](../studies/12-paper-prophet/) [39](../studies/39-black-box/) [84](../studies/84-moon-math/) | 4 | 0 | 0 | 0 |
| Microstructure & crowds — [04](../studies/04-social-oracle/) [09](../studies/09-phantom-kernel/) [11](../studies/11-vanishing-penny/) | 3 | 1 | 1 | 0 |
| Pre-registered — [14](../studies/14-gamma-gospel/) | 1 | — | — | — |

\* *"Survived costs" = stamped Investable or Fragile (alive on paper, even if
thin). The complement is Mirage.*

Three patterns jump out:

- **ML & forecasting is the deadest corner of the bench: 0 for 4.** Every
  model-driven forecaster — Markov pipeline [10], ARIMA+GARCH [12], neural net
  [39], the Stock-to-Flow model [84] — produced an in-sample story and an out-of-sample coin flip.
- **Calendar effects are the opposite failure mode: among the most *real* per
  capita (4 of 10) and almost none tradable.** The pattern is genuinely in the
  data; the trade built on it forfeits more than it captures
  ([42](../studies/42-last-call/), [55](../studies/55-summer-lull/)) — or dies
  the moment it's published ([67](../studies/67-fed-drift/)).
- **Momentum, trend and carry don't die — they limp.** These families
  collect Fragile stamps, not Mirage ones (momentum 5/6 alive-but-thin, carry
  4/6): premia with a century of literature that one tape can't certify and
  costs nearly erase.

---

## Five lessons the bench keeps teaching

These aren't opinions — each one fell out of multiple studies independently.

**1 · The edge dies at the costs line, not the signal line.**
Of the 22 statistically real signals, 19 failed or barely survived
tradability. The overnight drift is real and untradable
[[01](../studies/01-overnight-anomaly/)]; intraday reversal is real with a
3.31 bp break-even that lives in the least-liquid names
[[33](../studies/33-slingshot/)]; the turn-of-the-month premium is real at
*t* = 5.1 and a window-only book — even with its cash leg paid the T-bill —
still compounds half of buy-and-hold
[[42](../studies/42-last-call/)]; IBS snap-back is real and gone at the spread
[[19](../studies/19-rubber-band/)]. Beat 6 — *could you trade it?* — is where
almost everything dies.

**2 · Survivorship doesn't just flatter results — it manufactures and even
inverts them.**
On a survivor panel of large caps, the lottery effect ran *backwards*
(−10.4%/yr, *t* = −2.5) [[53](../studies/53-jackpot/)], the idiosyncratic-vol
puzzle inverted decisively [[54](../studies/54-static/)], the 52-week-high
premium came out negative [[50](../studies/50-high-water/)], the net-issuance
hedge flipped because the decade's diluters were the growth winners
[[64](../studies/64-share-shuffle/)], and asset-growth showed nothing where
the literature's premium hides in micro-caps
[[44](../studies/44-growth-spurt/)]. When we found a positive result on a
survivor panel, we capped it as an upper bound
[[48](../studies/48-groundhog/)] — the bias cuts both ways and we say which.

**3 · Post-publication decay is the norm, not the exception.**
The pre-FOMC drift is the most spectacular case on the bench: 3% of sessions
carried 11.5% of SPY's entire cumulative return — until Lucca-Moench published
it in 2011 and the drift collapsed from +0.24%/day to +0.09%
[[67](../studies/67-fed-drift/)]. The size premium never showed at all on a
39-year tradable proxy [[45](../studies/45-vanishing-act/)]; turn-of-the-month
faded from 13.8 to 4.8 bp/day after 2008 [[42](../studies/42-last-call/)];
betting-against-beta decayed 0.70 → 0.28 [[43](../studies/43-free-lunch/)];
textbook pairs stopped paying once everyone copied them
[[05](../studies/05-twin-spread/)]; the vendor's dual-momentum edge thinned
right after the publication that sells it [[40](../studies/40-paper-tiger/)];
oil-predicts-stocks reads exactly zero out of sample
[[49](../studies/49-black-gold/)]. An anomaly's discovery date is the start of
its obituary.

**4 · Leverage is never free — the "free lunch" is usually the financing
bill, in disguise.**
Betting against beta needs 2.78× leverage to be market-neutral, and realistic
financing drags its Sharpe from 0.47 to 0.02
[[43](../studies/43-free-lunch/)]. The retail CFD markup — charged on the
whole notional, not the borrowed slice — costs a levered dip-buyer 2.65
pts/yr [[30](../studies/30-house-edge/)]. The 3× ETF "free amplifier" tripled
the drawdown, not the Sharpe (0.90 vs 0.98, −82% trough)
[[61](../studies/61-slow-burn/)]; extending duration for term premium
*lowers* the Sharpe [[59](../studies/59-downhill/)]; and vol-targeting the
carry trade makes its crash *worse* [[27](../studies/27-steamroller/)]. When
a strategy's appeal is "same return, just levered," the lender has already
priced your idea.

**5 · The only things still green on the bench manage risk — nothing that
predicts returns pays.**
All three Investable stamps are risk machinery: one scales exposure *down* when
markets get loud [[16](../studies/16-storm-shy/)], one balances risk
across assets and wins on Sharpe, not return [[68](../studies/68-all-weather/)],
and one is the plain 60/40 that lowers risk-adjusted-by-the-bond-bull, not
return [[97](../studies/97-balancing-act/)]. The honest fix for carry's crash is
diversification, not a smarter signal [[36](../studies/36-greenback/)]; a
thin trend sleeve earns its keep as crisis alpha inside a 60/40, not
standalone [[31](../studies/31-trade-winds/)]; gold and bitcoin flunk their
hedge stories yet survive as diversifiers
[[69](../studies/69-safe-haven/), [70](../studies/70-digital-gold/)]; and the
min-vol ETF really does cut risk, just not beat the market
[[58](../studies/58-bunker/)]. Nothing on this bench forecasts returns and
pays. Several things manage risk and do.

---

## The podium

**🟩 The three that made it.**
[**16 · Storm-Shy**](../studies/16-storm-shy/) — Real × Investable. Scale
exposure down when realized vol spikes. It survives robust inference, real
costs, capacity, a parameter sweep and a third tape — and note what it is: not
an alpha, a *risk overlay*.
[**68 · All-Weather**](../studies/68-all-weather/) — Real × Investable. Risk
parity earns the best Sharpe of anything we tested (0.92) with a third of
equities' drawdown — by predicting nothing and balancing everything. Half the
*return* of stocks, though: the green is risk-adjusted, not absolute.
[**97 · Balancing-Act**](../studies/97-balancing-act/) — Real × Investable. The plain
60/40 lifts the excess-of-cash Sharpe over 100% stocks (HAC *t* = 2.3, a bootstrap CI
clear of zero) and halves the drawdown — but it forfeits ~2.6 pts/yr of return, leans on
the historic bond bull, and the bonds did *not* cushion 2022. Risk-adjusted, not absolute.

**🟨 The honest fragiles** — Real signals that survive on paper but are thin,
decaying, or capacity-starved. Worth knowing; not worth quitting your job for:

| | Study | What's real | Why only fragile |
|:--:|---|---|---|
| [48](../studies/48-groundhog/) | Groundhog | Month-of-year seasonality, *t* = 4.1, undecayed | Survivor-panel upper bound; breaks even near ~19 bp |
| [52](../studies/52-smoke-screen/) | Smoke-Screen | Accruals: cash-backed earnings win, Sharpe 0.64 | Short-side costs; documented post-2000 fade |
| [56](../studies/56-tide-table/) | Tide-Table | CAPE forecasts 10-year returns (R² 0.28) | A tide table, not a stopwatch — useless at 1 year |
| [59](../studies/59-downhill/) | Downhill | Term premium, +2.2%/yr over cash | Sharpe 0.32 vs cash's 1.82; 2022 took −23% |
| [63](../studies/63-free-fall/) | Free-Fall | Short-vol carry, +12%/yr (SVXY) | Skew −4.8, one −83% day; five crash days wiped 95% |
| [66](../studies/66-inverted/) | Inverted | Curve inversion → +1% next 18m vs +16% normal | ~5% of months, a year of melt-up first — no sell button |
| [67](../studies/67-fed-drift/) | Fed-Drift | Pre-FOMC drift carried 11.5% of SPY's return | Publication killed it: +0.24%/day → +0.09% after 2011 |
| [71](../studies/71-ambush/) | Ambush | Confluence of four dead-net edges: +19.6 bp/day at K≥3 (HAC *t* = 3.1), undecayed, costs defeated by rarity | ~15 trades/yr → +1.2%/yr excess; OOS Sharpe +0.28 under the frozen 0.30 bar |
| [75](../studies/75-knee-jerk/) | Knee-Jerk | Connors RSI(2) oversold bounce: pooled HAC *t* = 10.7, beats a coin by +57 bp/trade | Decayed 35% since the 2008 book; long-only beta in a bull market; the 200-SMA filter *hurts* |

---

## Challenge the bench

This page will be wrong eventually — that's the design. One hundred and two verdicts is
one hundred and two falsifiable claims, each with reproducible code, pinned data
fingerprints, and the exact line where we think the dream dies.

- **Think a Mirage is tradable?** Fork the study, change the cost model or
  the venue, and show the break-even. Beat 7 of every notebook says what we'd
  consider convincing.
- **Think a None is real?** The inference stack (HAC, Lo, bootstrap, Reality
  Check) is in [`quantlab/`](../quantlab/) — run it on your variant.
- **Got a candidate for the queue?** Open an issue. The ideas that look most
  embarrassing to test are usually the best ones.

The map gets a new chip every time. `python tools/make_bench_figures.py`
redraws it.

---

*Part of [Open-Alpha-Lab](../README.md). Counts generated from the README
table by [`tools/make_bench_figures.py`](../tools/make_bench_figures.py).
Not investment advice — research and education. See [LICENSE](../LICENSE).*
