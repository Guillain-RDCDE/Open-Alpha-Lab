# What 60 teardowns taught us

*Sixty famous trading ideas — anomalies, folk strategies, vendor backtests, things
people swear by — each put through the [same protocol](../METHODOLOGY.md) and
stamped twice: **is the signal real?** and **does it survive real execution and
scale?** This page is the view from above. It aggregates; it doesn't re-judge —
every verdict below links back to the study that earned it.*

![The bench map — 60 studies on a Signal × Tradability grid](bench_map.png)

*(Regenerate with `python tools/make_bench_figures.py` — it parses the
[README table](../README.md), so it's always in sync.)*

---

## The score

Of the 60 studies on the bench, 59 carry final stamps (study
[14](../studies/14-gamma-gospel/) is pre-registered, verdict pending):

| | Investable | Fragile | Mirage | |
|---|:--:|:--:|:--:|:--:|
| **Real** | **1** | 5 | 7 | 13 |
| **Weak** | 0 | 11 | 16 | 27 |
| **None** | 0 | 1 | 18 | 19 |
| | **1** | **17** | **41** | **59** |

Read it the way the colours tell you to:

- **13 / 59 signals are statistically real.** Roughly one famous idea in five
  survives autocorrelation-robust inference. The other four-fifths are weak
  (27) or plain noise (19).
- **41 / 59 are mirages once you try to trade them.** Costs, capacity,
  decay, or the discovery that the "edge" was beta all along.
- **Exactly 1 / 59 is investable.** [Storm-Shy](../studies/16-storm-shy/) —
  and tellingly, it isn't a return predictor at all. It's a risk-management
  overlay.

The single most important cell isn't the green one — it's **Real × Mirage
(7 studies)**. Seven effects that are *genuinely there in the data* and still
can't pay you. That gap between "true" and "tradable" is the bench's whole
thesis, measured.

---

## Where ideas go to die — mortality by family

We sorted the 60 into rough families. The boundaries are judgement calls (is
the 52-week high a chart pattern or a momentum factor? we said factor) — the
totals below are honest, the taxonomy is approximate.

| Family | Studies | Real | Survived costs* | Investable |
|---|:--:|:--:|:--:|:--:|
| Equity factors & fundamentals — [18](../studies/18-dull-roar/) [34](../studies/34-aftershock/) [38](../studies/38-chorus/) [43](../studies/43-free-lunch/) [44](../studies/44-growth-spurt/) [45](../studies/45-vanishing-act/) [46](../studies/46-bargain-bin/) [50](../studies/50-high-water/) [51](../studies/51-blue-chip/) [52](../studies/52-smoke-screen/) [53](../studies/53-jackpot/) [54](../studies/54-static/) [57](../studies/57-yield-trap/) [58](../studies/58-bunker/) | 14 | 1 | 3 | 0 |
| Technical & chart patterns — [02](../studies/02-falling-knife/) [07](../studies/07-coiled-spring/) [08](../studies/08-true-strength/) [13](../studies/13-crimson-hour/) [15](../studies/15-sigma-sleight/) [17](../studies/17-glass-ceiling/) [19](../studies/19-rubber-band/) [21](../studies/21-fools-gold/) [22](../studies/22-crystal-ball/) | 9 | 1 | 1 | 0 |
| Momentum & trend — [20](../studies/20-freight-train/) [24](../studies/24-stampede/) [25](../studies/25-clean-slate/) [28](../studies/28-carousel/) [31](../studies/31-trade-winds/) [40](../studies/40-paper-tiger/) | 6 | 0 | 5 | 0 |
| Carry, curves & commodities — [27](../studies/27-steamroller/) [29](../studies/29-hedgers-toll/) [35](../studies/35-contango/) [36](../studies/36-greenback/) [59](../studies/59-downhill/) [60](../studies/60-long-shot/) | 6 | 1 | 4 | 0 |
| Calendar & seasonal — [01](../studies/01-overnight-anomaly/) [41](../studies/41-hangover/) [42](../studies/42-last-call/) [48](../studies/48-groundhog/) [55](../studies/55-summer-lull/) | 5 | 3 | 1 | 0 |
| Mean reversion & stat-arb — [05](../studies/05-twin-spread/) [23](../studies/23-broken-tether/) [26](../studies/26-sand-castle/) [32](../studies/32-rip-tide/) [33](../studies/33-slingshot/) | 5 | 2 | 0 | 0 |
| Vol & risk overlays — [03](../studies/03-fear-gauge/) [06](../studies/06-clockwork-vol/) [16](../studies/16-storm-shy/) [30](../studies/30-house-edge/) | 4 | 2 | 1 | **1** |
| Macro & valuation timing — [37](../studies/37-barometer/) [47](../studies/47-paper-moon/) [49](../studies/49-black-gold/) [56](../studies/56-tide-table/) | 4 | 1 | 2 | 0 |
| ML & model forecasting — [10](../studies/10-markov-mint/) [12](../studies/12-paper-prophet/) [39](../studies/39-black-box/) | 3 | 0 | 0 | 0 |
| Microstructure & crowds — [04](../studies/04-social-oracle/) [09](../studies/09-phantom-kernel/) [11](../studies/11-vanishing-penny/) | 3 | 1 | 1 | 0 |
| Pre-registered — [14](../studies/14-gamma-gospel/) | 1 | — | — | — |

\* *"Survived costs" = stamped Investable or Fragile (alive on paper, even if
thin). The complement is Mirage.*

Three patterns jump out:

- **ML & forecasting is the deadest corner of the bench: 0 for 3.** Every
  model-driven forecaster — Markov pipeline [10], ARIMA+GARCH [12], neural net
  [39] — produced an in-sample story and an out-of-sample coin flip.
- **Calendar effects are the opposite failure mode: the most *real* per
  capita (3 of 5) and almost none tradable.** The pattern is genuinely in the
  data; the trade built on it forfeits more than it captures
  ([42](../studies/42-last-call/), [55](../studies/55-summer-lull/)).
- **Momentum, trend and carry don't die — they limp.** These families
  collect Fragile stamps, not Mirage ones (momentum 5/6 alive-but-thin, carry
  4/6): premia with a century of literature that one tape can't certify and
  costs nearly erase.

---

## Five lessons the bench keeps teaching

These aren't opinions — each one fell out of multiple studies independently.

**1 · The edge dies at the costs line, not the signal line.**
Of the 12 statistically real signals, 11 failed or barely survived
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
premium came out negative [[50](../studies/50-high-water/)], and asset-growth
showed nothing where the literature's premium hides in micro-caps
[[44](../studies/44-growth-spurt/)]. When we found a positive result on a
survivor panel, we capped it as an upper bound
[[48](../studies/48-groundhog/)] — the bias cuts both ways and we say which.

**3 · Post-publication decay is the norm, not the exception.**
The size premium is the cleanest case: +0.1%/yr over 39 years, sign-flipped
since 2010 [[45](../studies/45-vanishing-act/)]. Turn-of-the-month faded from
13.8 to 4.8 bp/day after 2008 [[42](../studies/42-last-call/)];
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
pts/yr [[30](../studies/30-house-edge/)]. Extending duration for term premium
*lowers* the Sharpe [[59](../studies/59-downhill/)], and vol-targeting the
carry trade makes its crash *worse* [[27](../studies/27-steamroller/)]. When
a strategy's appeal is "same return, just levered," the lender has already
priced your idea.

**5 · The only thing still green on the bench is risk management — not
prediction.**
The lone Investable stamp scales exposure *down* when markets get loud
[[16](../studies/16-storm-shy/)]. The honest fix for carry's crash is
diversification, not a smarter signal [[36](../studies/36-greenback/)]; a
thin trend sleeve earns its keep as crisis alpha inside a 60/40, not
standalone [[31](../studies/31-trade-winds/)]; blending decorrelated signals
genuinely raises Sharpe — until a losing leg dilutes it
[[38](../studies/38-chorus/)]; and the min-vol ETF really does cut risk, just
not beat the market [[58](../studies/58-bunker/)]. Nothing on this bench
forecasts returns and pays. Several things manage risk and do.

---

## The podium

**🟩 The one that made it.**
[**16 · Storm-Shy**](../studies/16-storm-shy/) — Real × Investable. Scale
exposure down when realized vol spikes. It survives robust inference, real
costs, and capacity — and note what it is: not an alpha, a *risk overlay*. The
bench's only green chip doesn't predict anything.

**🟨 The honest fragiles** — Real signals that survive on paper but are thin,
decaying, or capacity-starved. Worth knowing; not worth quitting your job for:

| | Study | What's real | Why only fragile |
|:--:|---|---|---|
| [48](../studies/48-groundhog/) | Groundhog | Month-of-year seasonality, *t* = 4.1, undecayed | Survivor-panel upper bound; breaks even near ~19 bp |
| [52](../studies/52-smoke-screen/) | Smoke-Screen | Accruals: cash-backed earnings win, Sharpe 0.64 | Short-side costs; documented post-2000 fade |
| [56](../studies/56-tide-table/) | Tide-Table | CAPE forecasts 10-year returns (R² 0.28) | A tide table, not a stopwatch — useless at 1 year |
| [59](../studies/59-downhill/) | Downhill | Term premium, +2.2%/yr over cash | Sharpe 0.32 vs cash's 1.82; 2022 took −23% |

---

## Challenge the bench

This page will be wrong eventually — that's the design. Sixty verdicts is
sixty falsifiable claims, each with reproducible code, pinned data
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
