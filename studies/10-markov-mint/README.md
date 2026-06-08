# Study 10 — Markov-Mint 🎲 — can a Markov chain on price history "win every single trade" on a prediction market?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Studies 02–07 test a trigger in **price/vol/flow** on the **stock** tape, this one
> leaves equities for **prediction markets** and tests a different animal: not a chart pattern
> but a whole **quant pipeline** — Markov chain → Monte-Carlo → calibration → Kelly — sold on
> a viral thread as a way to print money on Polymarket.*

## Verdict — read this first

*Measured on a **reproducible**, fully offline experiment: **2,000** synthetic binary markets
whose price is the exact Bayesian posterior — a **martingale**, so the price is provably the
best estimate of the outcome and **no edge exists by construction** — plus a second basket
with a **planted favorite-longshot wedge** so we can size what a real edge would be worth. The
article's five-step pipeline is ported **verbatim** ([`markov_mint/markov.py`](markov_mint/markov.py)),
then scored against the **true** resolutions. Bid/ask charged: **2¢** round-trip. As-of
2026-06-01, sample fingerprint `585b80af7d53`; every number in [`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — does the chain find an edge the price misses? | `NONE` | On a provably-fair market the machine's realized directional edge is **−0.68 pp** (HAC *t* = **−0.77**) — a coin flip; the *most* any method could capture given the true odds is **exactly 0**. The raw Monte-Carlo "edge" is zero-mean noise whose spread **collapses from ~20 pp to ~2 pp as history grows** — the fingerprint of estimation error, not information. |
| **Tradability** — does it survive Kelly + a normal bid/ask? | `MIRAGE` | Kelly-sized and scored against truth, the bankroll is **incinerated** — **0.0003×** after a 2¢ spread, and still **0.002× at *zero* cost**. The article's calibration table tops out at **0.958**, so every contract trading richer is handed a probability *below its own price* — a mechanical **BUY NO** on **568 / 2,000** markets. Shorting a fair favorite at 98¢ loses ~98% of the time. |
| **"Win every single trade"?** | `BUSTED` | Realized win rate **51.6%** — a coin flip. And the only real effect in the whole pipeline, the favorite-longshot bias, nets **−13.6% per trade *even for an oracle with the true probabilities*** once a 2¢ spread is charged. |

> **In one sentence:** run the five-step "Markov chain that wins every trade" on markets whose
> price is *provably* fair and it finds nothing but Monte-Carlo noise — worse, a hard-coded
> calibration ceiling makes it reflexively short every strong favorite, so Kelly-sizing the
> "edge" doesn't break even, it **destroys the bankroll**; and the one genuine effect it leans
> on, the longshot bias, is too thin to beat a two-cent spread even with perfect information.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

> *"Model the price as a Markov chain of states, Monte-Carlo the resolution, calibrate against
> the longshot bias, size with Kelly, execute as a maker — and harvest a repeatable edge on
> Polymarket. Win every single trade."*

The claim comes from a viral X/Twitter thread by Alex (@de1lymoon), *How To Use Markov Chains
To Win Every Single Trade + [Quant Framework]* (26 May 2026; ~1.2 M views — see
[`docs/references.md`](docs/references.md)). It is unusually concrete — a runnable five-step
pipeline, not vibes:

1. **Build a transition matrix.** Discretise a contract's price history into 10 states
   (0–10¢, …, 90–100¢) and count how often it moves from each state to every other.
2. **Monte-Carlo the resolution.** Random-walk 10,000 paths through the matrix to expiry; the
   fraction landing in "YES territory" is your probability estimate.
3. **Calibrate** that raw probability against an empirical favorite-longshot table (attributed
   to a "72.1 M-trade" study) so you "stop overpaying for longshots".
4. **Size with quarter-Kelly.**
5. **Execute with limit orders** (maker, not taker).

The evidence offered is the framework itself and an empirical citation we could **not locate a
primary source for** (the "Becker, 72.1 M trades, $18.26 bn" figures; we treat them as folklore
and test the *method*, not the unverifiable tape — see references). Crucially, the post never
runs the pipeline against a *null*: it never asks what the machine prints on a market that is
already correctly priced. That is the entire study.

> 🔬 **For the quants** — H₁: the pipeline's signed bet, ``sign(calibrate(MC) − price)``,
> captures realized edge over the price, ``E[sign(edge)·(outcome − price)] > 0`` with HAC
> *t* > 2. The sharp null is an **efficient prediction market**: the price is a martingale
> equal to ``P(resolve YES | info)``, so by Samuelson (1965) there is no exploitable side and
> H₁ must fail. We build exactly that market (a Bayesian posterior — martingale by the tower
> property) and feed the author's own code.

## 2 · So What?

If a Markov chain on nothing but past prices could front-run resolutions, it would be a money
printer *and* a deep statement about markets: it would mean a prediction market's price — the
crowd's probability, the thing the whole venue exists to discover — systematically misses
information sitting in its own recent history. Forty years of efficient-market theory says the
opposite: a correctly priced contract is a **martingale**, its next move unforecastable from its
past by construction. So the stakes are a clean fork. Either the thread has found a crack in one
of the most-studied results in finance using a dozen lines of `numpy` — or the pipeline is an
elaborate way of trading noise, and the "edge" is what an estimator produces when there is
nothing to estimate. Prediction markets make the test unusually clean: every contract *resolves*,
so unlike a stock there is a ground-truth outcome to score every bet against.

> 🔬 **For the quants** — a martingale's price already minimises mean-squared error to the
> outcome; any function of the past price path has zero conditional covariance with
> ``outcome − price``. A reported edge can therefore only be (a) finite-sample noise in the
> transition-matrix estimate, or (b) a deterministic artefact of the pipeline's own choices
> (the "ends above 50¢" YES-definition; the calibration table's clipping). We measure both.

## 3 · How We'd Know

There is no cached Polymarket tape, and the claim is a *method* claim, so the honest test is to
feed the method data whose truth we control and pre-register what would kill it:

- **Signal `NONE`** if, on the efficient (martingale) null, the machine's realized directional
  edge is statistically indistinguishable from zero (HAC *t* < 2) — i.e. it does no better than
  the price it was handed. (The oracle benchmark — betting with the *true* probability — is
  identically zero here, because there is no exploitable side on a fair market.)
- **Tradability `MIRAGE`** if, Kelly-sized and scored against the true outcome, the strategy
  fails to grow the bankroll once a realistic bid/ask is charged — or if it loses *before* costs.
- **"Win every trade" `BUSTED`** if the realized win rate is a coin flip (any probabilistic edge
  wins < 100% of the time; the headline is mathematically impossible).

The traps we watch for, and how we defang them: **look-ahead** (the matrix and MC see only the
pre-resolution path); **a rigged null** (we don't assume efficiency — we *construct* a genuine
Bayesian posterior, then verify it is calibrated: ``E[outcome] = E[price]``); and **straw-manning**
(the pipeline is the author's code, verbatim, including the bits we think are bugs — we even add a
switch to *delete* the Markov stage and show the decisions barely care).

> 🔬 **For the quants** — protocol (shared desk rubric):
> 1. **Measure** the realized directional edge per trade and HAC-*t* it (Newey-West).
> 2. **Robust inference / is-it-noise** — the raw MC edge's mean and its *scaling with history
>    length* (noise shrinks ∝ 1/√transitions; signal does not).
> 3. **Critique the mechanism** — ablate the Markov stage (``raw_prob := price``); decompose the
>    P&L by price bucket to locate the calibration-ceiling forced-NO bug.
> 4. **Alpha vs artefact** — an oracle benchmark bounds the *real* recoverable edge on a planted
>    wedge; the gap to the machine is what the chain leaves on the table.
> 5. **Execution & capacity** — a spread sweep; the net to even a perfect-information trader.
> 6. **Verdict** — the three stamps.
>
> Engine: `quantlab.analytics.mean_tstat_hac`, `quantlab.stats.sharpe_ci_bootstrap`,
> `quantlab.repro`. Study code in [`markov_mint/`](markov_mint/).

## 4 · The Teardown

> *We run it. Here's what the data actually says.*

- **It finds nothing — the directional edge is a coin flip.** Across 2,000 fair markets the
  machine takes a side on 1,142 of them and captures **−0.68 pp** of edge per trade (HAC
  *t* = **−0.77**): statistically zero, and if anything negative. The oracle that knows the true
  probability captures **exactly 0** — because on a fair market there is no side that helps.
- **The chain's only output is noise that shrinks with data.** The raw Monte-Carlo "edge"
  (``raw_prob − price``) is zero-mean, and its standard deviation **collapses from ~19.8 pp at
  20 days of history to ~2.1 pp at 250** — exactly how estimation error behaves as a sparse
  10×10 matrix fills in. A real signal would not melt as you give it *more* data.
- **The Markov stage isn't even load-bearing — it's a noise *generator*.** Delete it entirely
  (``raw_prob := price``) and the bets collapse from **57% of markets to 18%**; the full
  pipeline's "edge" correlates just **0.38** with the price-only version, so **~85% of what it
  acts on is Monte-Carlo noise**. The chain roughly **triples the number of bets**, all of them
  coin flips.
- **It loses *before* costs — the calibration ceiling forces it to short favorites.** The
  article's table maxes out at **0.958**, so any contract richer than that is mechanically
  assigned a probability *below its own price* → a forced **BUY NO**. Here **568 / 2,000**
  contracts price above the ceiling, and the machine shorts **64%** of them. On the null that is
  a short of a fair favorite: the **(90¢–100¢] bucket holds the most trades (426) and the worst
  return (−39.7%), winning just 12.7%** of the time.
- **So Kelly incinerates the bankroll.** Quarter-Kelly, scored against truth: terminal bankroll
  **0.0003×** after a 2¢ spread — and still **0.0017× at zero cost**. Per-trade Sharpe **−0.12**
  (bootstrap CI **[−0.54, −0.03]**, 99.8% of resamples negative). Win rate **51.6%**.

> 🔬 **For the quants** — the realized directional edge is ``dir_sign·(outcome − price)``, pooled
> over active trades and HAC-tested; on the martingale null its expectation is zero by
> construction, and the sample confirms it (*t* = −0.77). The *capital* return is materially
> negative (−16.5% per trade at zero spread) because the forced-NO trades cluster at extreme
> prices where the binary payoff is most asymmetric — the calibration clipping turns "no signal"
> into "negative signal". Reproduce via [`examples/verify.py`](examples/verify.py) →
> [`docs/results.md`](docs/results.md).

<details>
<summary>🔬 Why a martingale defeats the whole pipeline — and why the MC looks like it "works"</summary>

A prediction-market price that equals ``P(resolve YES | information)`` is a martingale:
``E[p_{t+1} | ℱ_t] = p_t``. Two consequences sink the method. **(1)** A transition matrix
estimated from a martingale path encodes a driftless random walk; Monte-Carloing it returns,
in expectation, the current price — so the "edge" is sampling error, and we see its std fall as
1/√(transitions). **(2)** The author defines a path as YES when the *simulated price* ends in the
upper half of the state grid (``state ≥ n_states//2``), which is **not** the event resolving YES.
For a contract near an extreme this pushes ``raw_prob`` *more* extreme than the price, and then
``calibrate`` — clipped to [0.0043, 0.958] — drags any rich favorite below its price, manufacturing
a confident short. The pipeline *feels* like it is computing something (it prints crisp
probabilities and "edges"); it is laundering noise through a lookup table whose only economic
content is the longshot bias.

</details>

## 5 · The Verdict

- **Signal · `NONE`.** On a provably-fair market the machine captures no positive directional
  edge (−0.68 pp, HAC *t* = −0.77), and the most any method *could* capture is zero. Its raw
  output is estimation noise that shrinks with history. There is nothing there.
- **Tradability · `MIRAGE`.** Scored against truth and Kelly-sized, the bankroll is destroyed —
  0.0003× after a 2¢ spread, 0.0017× even at zero cost — because the calibration ceiling forces a
  BUY NO on every strong favorite, a guaranteed loser on fair odds. It doesn't *decay* to a
  mirage at some cost level; it is underwater before the first cent of spread.
- **"Win every single trade" · `BUSTED`.** 51.6% win rate — a coin flip — and even an oracle with
  the true probabilities nets −13.6%/trade once a 2¢ spread is charged. The headline is
  mathematically impossible and empirically a loss.

> 🔬 **For the quants** — decisive numbers in one place: machine directional edge −0.68 pp
> (HAC *t* = −0.77) vs oracle 0; raw-edge std 19.8 → 2.1 pp over history 20 → 250; ablation drops
> active-trade share 0.571 → 0.183 with edge-corr 0.38; null terminal bankroll 0.0003× @2¢ /
> 0.0017× @0; per-trade Sharpe −0.12 (CI [−0.54, −0.03]); top-bucket return −39.7% @ 12.7% win.
> Fingerprint `585b80af7d53`.

## 6 · Could You Trade It?

The honest money question has two layers, and the pipeline fails both. *First*, the method as
written is **worse than not trading**: Kelly-sizing its noisy "edge" hands back your stake fastest
exactly where it is most confident (shorting favorites it cannot price). *Second*, suppose you
threw the Markov machine away and traded the one real effect it gestures at — the
favorite-longshot bias. We plant a genuine wedge and let an **oracle that knows the true
probability** trade it: the gross edge is **+1.38 pp per trade**, real and statistically
present — but **net of a 2¢ round-trip it is −13.6%**. The recoverable edge lives in the
moderate-longshot band (~10–50¢, where it's worth ~7–8 pp) — precisely the low-liquidity
contracts where a real Polymarket spread is *widest*, not the 2¢ we charged. And the maker
rebate the thread leans on ("+1.12% per trade") ignores adverse selection and fill probability:
you are filled when an informed taker wants the other side. The candid bottom line: there is no
desk here. The pipeline is a negative-edge bet on a fair market, and the genuine bias underneath
it is a few cents that the bid/ask eats before you do.

> 🔬 **For the quants** — oracle gross +1.38 pp vs net −13.6% @2¢ means break-even spread for a
> *perfect-information* trader is well inside a single cent across most of the price range; on
> the moderate-longshot band where the edge concentrates, realistic spreads (often 3–10¢ on thin
> contracts) dominate it many times over. The maker "+1.12%" is an unconditional average that
> survivorship-ignores unfilled limit orders and prices in no adverse selection (Glosten-Milgrom);
> it is not a harvestable constant.

## 7 · Going Further

- **A non-Markov (path-dependent) market.** Our null is memoryless by construction, which is the
  *steelman* for a Markov model. The interesting variant plants genuine one-step memory the chain
  *could* in principle catch — but the calibration step would still overwrite it, which is itself
  worth showing.
- **A real Polymarket tape.** The offline core proves the method is empty on a fair market;
  pointing it at resolved Polymarket histories (with real spreads and the actual favorite-longshot
  curve) would put a number on how much the bias is worth *net* on the venue the thread targets.
- **Fix the pipeline and re-test.** Uncap the calibration table, redefine "YES" as the true
  resolution rather than "price > 50¢", and re-run — to show that even a *debugged* version
  reduces to "bet the longshot bias" and dies to the spread (i.e. the bugs are not why it fails).
- **The maker microstructure.** Model fills and adverse selection to price the "+1.12% maker
  rebate" honestly — the thread's strongest-sounding claim and its least examined.
- **What to PR:** a path-dependent market generator, a real-tape loader, the debugged-pipeline
  control, or an adverse-selection maker model.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`markov_mint/data.py`](markov_mint/data.py) | the synthetic markets — an efficient (martingale) null + a planted favorite-longshot wedge, with realized outcomes to score against |
| [`markov_mint/markov.py`](markov_mint/markov.py) | the article's five-step pipeline, ported verbatim (transition matrix, Monte-Carlo, calibration table, Kelly), with a switch to ablate the Markov stage |
| [`markov_mint/robustness.py`](markov_mint/robustness.py) | the falsification battery: headline HAC test, noise-vs-history, inertness, costed P&L, the calibration-ceiling decomposition, planted-edge recovery |
| [`examples/verify.py`](examples/verify.py) | the headline run → [`docs/results.md`](docs/results.md) (as-of + fingerprint) |
| [`notebooks/`](notebooks/) | `01_for_the_curious` (the story) and `02_for_the_quants` (the teardown), same seven beats |
| [`docs/references.md`](docs/references.md) | the thread + the efficiency / longshot-bias / Kelly literature it walks into |

The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
