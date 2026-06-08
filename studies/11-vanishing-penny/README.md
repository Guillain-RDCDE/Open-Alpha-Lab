# Study 11 — Vanishing-Penny 🪙 — how fast does a *guaranteed* Polymarket arbitrage close?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Studies 02–07 hunt an edge in a **price**, a **vol**, or an **information flow**,
> this one chases a different animal: a **risk-free** arbitrage that everyone agrees is
> real — and asks the only question that's left, which is **how long it lives**.*

## Verdict — read this first

*Measured on a **reproducible** run over **13** real Polymarket binary markets with
minute `prices-history` (2026-05-08 → 2026-05-31, as-of **2026-06-01**, gap fingerprint
`7baea17d9b7b`; every number in [`docs/results.md`](docs/results.md)). The offline core
([`examples/run_synthetic_demo.py`](examples/run_synthetic_demo.py)) validates the
half-life estimator against a **baked-in 6-minute** ground truth before it ever touches
the real tape. The standing caveat, named up front: the public minute tape **cannot see
the 2-second block** the real race runs on, so our half-life is a coarse* ***upper bound*** —
*which only deepens the verdict.*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is the arbitrage actually real? | `REAL` | It is risk-free by construction and well-documented: the paper (Saguillo et al. 2025) prices ~**\$40M** extracted in a year, and on our own real tape **161** genuine ≥3¢ `YES+NO` gaps open across 13 markets (median penny **4¢**). The free money exists. |
| **Tradability** — can a non-co-located trader catch it? | `MIRAGE` | Every one of those 161 episodes closes **inside our 1-minute measurement floor** (`frac_below_floor = 100%`, median episode duration **1 min**). Even against a *generous* 1-minute upper bound, a human reacting in 5 min keeps **3%** of the penny, in 30 min **~0%**. The edge belongs to the block, not the browser. |
| **Execution moat?** — is the edge structurally reserved? | `CONFIRMED` | The gap's true half-life is **below every resolution we can sample**: the median episode is exactly *one tick* long at 1, 2, 5, 15, 30 *and* 60-minute fidelity. A timescale that hides under any tape you bring is the definition of a moat — won in the ~30 ms before the next block, where retail is the exit liquidity. |

> **In one sentence:** the guaranteed Polymarket penny the viral thread sells you is
> **real and already gone** — it closes faster than the public tape can even sample, so
> the \$40M is a same-block infrastructure prize, and the retail "roadmap" is a slower way
> to provide the fast wallets their exit liquidity (and, at the bottom of the thread, an
> airdrop funnel).

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

A [viral thread](https://x.com/robrtcode) (2.9M views) lays out "the exact maths that pulled
**\$40,000,000** out of Polymarket — complete roadmap." Stated at full strength, the way its
believers do:

> *"When `YES` is \$0.62 and `NO` is \$0.33, that's \$0.95 — a guaranteed \$0.05. Quant
> systems scan 17,218 conditions across 2⁶³ outcomes in milliseconds, size with Kelly,
> execute parallel legs, and rotate capital. From April 2024 to April 2025 they extracted
> **\$39,688,585** in guaranteed arbitrage; the top wallet made **\$2,009,632** over 4,049
> trades — **\$496 of guaranteed profit per trade**, all year. The maths work. The
> infrastructure exists. The only question is whether you can build it before the next
> \$40M is extracted."*

It is a strong steelman precisely because the core is **true and checkable**: the arbitrage
is risk-free (a \$1-paying outcome bought for less than \$1), it is documented in a real
2025 paper — *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets*
([arXiv:2508.03474](https://arxiv.org/abs/2508.03474)) — and the wallet P&Ls are public
on-chain. There is no "is it real" to argue about. The thread then pivots to a sign-up link
and an "airdrop," which is the tell we keep in view.

> 🔬 **For the quants** — the claim is not a return forecast; it is a statement about a
> **first-passage time**. H₀ (the thread's implicit hope): the mispricing
> `g = 1 − (p_yes + p_no)` persists long enough — seconds-to-minutes — for a human to place
> two CLOB legs and bank `g`. H₁ (the desk's prior): `g` is closed by competing bots on a
> timescale far below human reaction, so the **half-life of `g`** is the whole game, and the
> realised retail capture `½^(latency / H)` is ≈ 0. We test H₀ by *measuring H*.

## 2 · So What?

If the penny lived even a minute or two, this would be the cleanest free lunch in modern
markets: a parameter-free, risk-free trade, documented to have paid \$40M, open to anyone
with a wallet. That is exactly the dream the thread monetises — and if it were reachable,
"markets are efficient" would have a \$40M hole anyone could climb through.

If instead the penny is gone in **seconds**, the same \$40M is the opposite lesson: a real
inefficiency that is *entirely* an **execution** prize — won by whoever submits inside the
2-second Polygon block, with everyone slower providing the exit liquidity that makes the fast
wallets' profit possible. The arbitrage being *real* is what makes the retail pitch
dangerous: the maths check out, so the mark never suspects the part that doesn't.

> 🔬 **For the quants** — the stakes live in the ratio of two latencies. The fast wallets
> act in ~**30 ms**; the block is ~**2 s**; a human loop (spot → log in → size both legs →
> submit two orders that confirm a block apart) is **minutes**. Guaranteed profit per
> attempt is the Bregman divergence between the quoted state and the no-arb manifold — but
> *captured* profit is that divergence times `½^(latency/H)`, and with `H` in seconds and
> `latency` in minutes the second factor is rounding error. The paper's own **45%** fill
> rate on *combinatorial* legs (vs 87% single-condition) is the same moat seen from the
> execution side.

## 3 · How We'd Know

The trap in this study is **resolution**, not selection. "Arbitrage exists" is trivially
true and proves nothing about tradability; the honest question is *how long a given gap
stays open*, and to answer it you need to actually watch gaps **open and close**. So the
falsifiable test is a **half-life**, announced before we run it:

- **Measure `H`, the half-life of the gap `g = 1 − (p_yes + p_no)`** — minutes until an
  opened mispricing has shrunk by half — across many real markets.
- **The line that decides the verdict:** if `H` is comfortably longer than a human reaction
  (say, minutes), the penny is **retail-reachable**; if `H` is at or below the **finest tape
  we can sample** (1 minute), it is a `MIRAGE` for anyone not racing the block, and we say so.
- **The trap we name out loud:** the CLOB `prices-history` tape is sampled at ~1 minute and
  its order-book snapshots froze ~2026-02-20, so we can only ever measure an **upper bound**
  on `H`. We therefore treat "we can't resolve it" as evidence *for* the mirage, never against
  — and we prove the estimator works on synthetic data where the answer is known, so a null
  on real data is a fact about the market, not the code.

> 🔬 **For the quants** — the protocol, powered by this study's
> [`prediction_arb/`](prediction_arb/): (1) **decompose/measure** — carve each gap series
> into episodes (a run above a 3¢ threshold), take the per-episode time-to-half and a pooled
> log-linear decay fit, two estimators that must agree if the decay is exponential;
> (2) **robust inference** — a percentile bootstrap CI on the median half-life;
> (3) **critique magnitude** — the `frac_below_floor` (episodes too short to even *observe*
> halving) and a **resolution sweep** that re-detects on a deliberately coarsened tape;
> (4) **alpha vs beta** has no analogue here (the trade is risk-free), so its slot is the
> **execution moat** — `retail_capture(H, latency)`; (5) **capacity** is the order-book depth
> (frozen, hence a beat-7 lead); (6) **verdict**. Engine: `data`, `arbitrage`, `robustness`.

## 4 · The Teardown

> *Synthetic first (the estimator must recover a half-life it was* ***given***\*), then the real
> 13-market tape. Reproduce: [`examples/run_synthetic_demo.py`](examples/run_synthetic_demo.py)
> (offline) and [`examples/verify_real.py`](examples/verify_real.py); full tables in
> [`docs/results.md`](docs/results.md).*

- **The estimator works — that's not the problem.** On the offline synthetic book (48
  markets, arbitrage gaps that decay with a *baked-in* 6-minute half-life), both estimators
  recover it: median time-to-half **6.0 min**, log-linear fit **6.19 min**, bootstrap CI
  **[6.0, 7.0]** over 1,647 episodes. Feed it a known half-life and it hands it back — so a
  null on real data is a measurement, not a bug.
- **On real markets, the penny is real.** Across 13 liquid binary markets, **161** genuine
  episodes clear the 3¢ bar, median peak **4¢** — the free money the thread describes does
  open, repeatedly.
- **And it is gone by the next sample.** **100%** of those 161 episodes are *below the
  1-minute floor*: not one lasts long enough to be **seen** halving (median duration **1
  minute** — the gap is sub-threshold again at the very next minute print). The empirical
  half-life is therefore **unmeasurable** — it is shorter than the finest tape we have.
- **The resolution sweep is the signature.** Re-detect on a coarsened tape and the median
  episode is *exactly one tick* at **every** fidelity — 1, 2, 5, 15, 30 and 60 minutes — while
  the episode count collapses **161 → 2**. A duration that equals your sampling interval no
  matter how you sample is the fingerprint of a process whose true timescale is **below all of
  them**.
- **So retail capture rounds to zero.** Even granting a *generous* 1-minute upper-bound
  half-life, `½^(latency/H)` leaves a 1-minute reaction **50%**, a 5-minute reaction
  **3.1%**, a 10-minute reaction **0.1%**, a 30-minute reaction **0.003%**. The true sub-minute
  half-life makes all of these optimistic.

> 🔬 **For the quants** — episodes are maximal runs of `|g| ≥ 0.03`; `time_to_half` is the
> assumption-light median (drops episodes that close before halving), `fit_half_life` the
> pooled through-origin slope of `log(|g|/|g_peak|)` on Δt. The real fit returns **7.72 min**,
> but that estimator survives only on the handful of multi-minute episodes — it is **selection
> -biased toward the slow tail** that the sweep shows collapsing, which is why we headline the
> empirical floor (`frac_below_floor = 1.0`), not the fit. Bootstrap CI is `nan` by
> construction: with zero finite per-episode half-lives there is nothing to resample. All from
> [`docs/results.md`](docs/results.md), as-of 2026-06-01, fingerprint `7baea17d9b7b`.

<details>
<summary>🔬 The maths, in full</summary>

A binary market's two outcomes trade in *separate* CLOB books, so `p_yes` and `p_no` are
independent quotes; the **arb gap** is `g(t) = 1 − (p_yes(t) + p_no(t))`. When `g > 0`,
buying one share of each side costs `1 − g` and pays exactly \$1 at resolution — a locked
`g` per pair, the "market-rebalancing" arbitrage of the paper (its other class,
*combinatorial*, exploits logical dependencies *across* markets and is beat-7). An episode
opens when a shock — a news print, a fat market order, a lagging leg — pushes `|g|` above
threshold; competing arbitrageurs then close it, and if that closing is proportional to the
remaining gap the decay is exponential, `g(t) = g₀ · ½^{Δt/H}`, with **half-life** `H`. The
synthetic generator builds exactly this (`g₀ ~ θ + Exp(scale)`, decay `½^{1/H}` per step,
plus observation noise) so `H` is known; the two estimators recover it. The realised retail
edge on a peak penny `g₀`, reacting `latency` after the open, is `g₀ · ½^{latency/H}` minus
two half-spreads and gas — and with `H` below a minute and `latency` in minutes, the first
factor alone kills it before costs are even charged.

</details>

## 5 · The Verdict

> *The stamps, now earned.*

- **Signal — `REAL`.** The arbitrage is risk-free by construction, documented at ~\$40M
  (Saguillo et al. 2025), and visible on our own tape: 161 genuine ≥3¢ gaps, median 4¢. There
  is nothing to debunk about *existence* — and saying so plainly is half the point, because it
  is what makes the retail pitch land.
- **Tradability — `MIRAGE`.** Every gap closes inside our 1-minute floor (100% below floor,
  median duration 1 min). Against even a generous upper bound the captured fraction is 3% at a
  5-minute reaction and ~0 at 30; the real, sub-minute half-life makes it worse. No venue,
  sizing, or "bot service" a retail thread can sell changes a first-passage time measured in
  seconds. My going-in prior was `FRAGILE` — a thin, fast-but-catchable edge; the data was
  less kind, and cleaner for it.
- **Execution moat — `CONFIRMED`.** The median episode is one tick long at every fidelity from
  1 to 60 minutes — a timescale that hides under any tape. The \$40M is therefore an
  execution prize reserved for sub-block latency (the paper's fast wallets act in ~30 ms),
  with the slow providing exit liquidity. This is the rare desk study where the signal is
  *unimpeachable* and the mirage is **entirely** in the reachability.

> 🔬 **For the quants** — decisive numbers in one place: real `n_episodes 161`,
> `median_peak_penny 0.04`, `frac_below_floor 1.00`, `median_duration_min 1`,
> `half_life_median nan` (sub-floor), `half_life_fit 7.72` (slow-tail biased); resolution sweep
> `n_episodes 161→2` and `median_duration_min = fidelity` for k ∈ {1,2,5,15,30,60}; capture
> `{1min: 0.50, 5min: 0.031, 30min: 3e-5}` against a generous 1-min bound. Synthetic control:
> baked `H = 6.0`, recovered `6.0 / 6.19`, CI `[6.0, 7.0]`. As-of 2026-06-01, fingerprint
> `7baea17d9b7b`.

## 6 · Could You Trade It?

> *The honest money question — the beat that separates this desk from a thread.*

No — and uniquely for this desk, *not* because the signal is fake. It is the realest signal
we've ever stamped. You can't trade it because the order of operations is fixed against you.
Polymarket runs a **Central Limit Order Book**, so the two legs are **sequential, not atomic**:
you buy `YES` at \$0.30 ✓, the book updates, and your `NO` fills at \$0.78 ✗ — the guaranteed
\$0.40 became a loss. The fast wallets avoid this by submitting both legs inside one ~2-second
block from a ~30 ms detection loop; the thread's "copy a fast wallet" pitch fails for the same
reason — you see the fill at block *N+1* and buy from the very wallet you meant to copy, at a
worse price. The paper's **45%** fill rate on combinatorial legs is the professionals
*themselves* missing nearly half the time; a browser refresher's rate is ~0. And the thread's
actual ask isn't a trade at all — it's a wallet connect and an "airdrop," i.e. the product being
sold is *you*.

The honest "what would it take" is therefore not better discipline but a different game
entirely: a colocated WebSocket feed, an integer-programming dependency detector, parallel
same-block submission, Kelly sizing against live book depth. That's the paper's stack, not a
retail one — and even it leaves half the combinatorial money on the table.

> 🔬 **For the quants** — break-even is moot the usual way (the gross is risk-free), so the
> binding constraint is the **capture fraction**, and `½^{latency/H} → 0` for any human
> `latency` against a sub-minute `H`. Capacity (order-book depth, Kelly-capped at ~50% of the
> book) is the *professional's* constraint and needs the snapshot feed that froze 2026-02-20 —
> a beat-7 lead, not a retail one. The lived series is the episode-duration distribution in
> [`docs/results.md`](docs/results.md): a spike at one minute and essentially nothing above it.

## 7 · Going Further

> *Open threads — and how you, the reader, could push this.*

- **Sub-second reconstruction.** The whole verdict is bounded by a 1-minute tape. Rebuild the
  gap from on-chain Polygon `OrderFilled` events (the paper analysed 86M transactions) and you
  could measure `H` *directly* in seconds instead of bounding it — almost certainly confirming
  it's sub-block, but quantifying the moat instead of inferring it.
- **The combinatorial class.** This study prices only the single-market `YES+NO` rebalancing
  arb (\$10.6M of the paper's \$40M). The larger \$29M is **combinatorial** — logical
  dependencies *across* markets (if "R wins PA by 5+" then "R wins PA"). Detecting those needs
  the dependency graph (`load_manifest` is the hook); their half-lives and fill rates are the
  obvious next measurement.
- **Capacity from depth.** Order-book snapshots before 2026-02-20 are still reconstructable —
  enough to ask the *professional's* question this study brackets: at what size does even a
  fast wallet move the book against itself, and where does the \$496/trade median come from?
- **A contributor could PR:** point `verify_real.py --discover` at a wider, event-driven
  universe (election nights, earnings, sports finals — where shocks are biggest), and test
  whether `H` lengthens at all when the whole market is dislocating at once.

The deep version — the synthetic validation, the bootstrap, the resolution sweep, and the
capture math — is in [`notebooks/02_for_the_quants.ipynb`](notebooks/).

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full method: episodes, two half-life estimators, sweep, capture |
| [`docs/results.md`](docs/results.md) | **the real run** — every headline table, fingerprinted and as-of'd |
| [`docs/markets_manifest.json`](docs/markets_manifest.json) | the 13-market input of record (slug + CLOB token ids + active window) |
| [`docs/references.md`](docs/) | sources + literature map (the two arXiv papers, the thread, the Polymarket API) |
| [`prediction_arb/`](prediction_arb/) | the study package: `data` · `arbitrage` · `robustness` |
| [`examples/`](examples/) | [`run_synthetic_demo.py`](examples/run_synthetic_demo.py) (offline) · [`verify_real.py`](examples/verify_real.py) (the real run) |

Every number is produced by [`prediction_arb/`](prediction_arb/), in the house style of the
shared [`../../quantlab/`](../../quantlab/) engine; `pytest` covers it in CI.
