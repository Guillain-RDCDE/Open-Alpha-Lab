# Study 05 — Twin-Spread 👯 — does pairs trading still pay after the world copied it?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Studies 02–04 hunt a trigger in the **price**, the **vol**, or the **information
> flow**, this one lives in the **cross-section**: the bet isn't on one name moving, but
> on two names moving back **together**.*

## Verdict — read this first

*Measured on a **reproducible** run of the textbook GGR (1999) rule over a cached, liquid
**174-name** US universe, 1962–2026 (split-only prices). The honest test is the **modern
era** (2005–2026): the universe is a *liquid basket*, **not** the thousands-of-names
CRSP cross-section GGR formed from, and its breadth grows from **7** eligible names in the
early 1960s to **171** today — only post-2004 are there enough names (≥60) to build the
genuinely tight pairs the rule needs. As-of 2026-06-01, price fingerprint `ac25cb7061be`;
every number in [`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — does the spread actually revert into a profit? | `NONE` | The minimum-distance pairs **don't reconverge enough to pay**: modern-era gross is **−0.37%/mo** (Sharpe **−0.44**, bootstrap CI **[−0.81, −0.04]**, 98% of resamples negative) — *negative even before a cent of cost*. The full sample is statistically zero (CI [−0.35, +0.13]). |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` | There's no edge for costs to kill — and they deepen the loss anyway (monotone in the spread), to **−0.43%/mo net**, **Sharpe −0.44**, a **−77% max drawdown**, with **β≈0** so there isn't even market beta to fall back on. Liquidity is *not* the binding constraint (capacity ~\$59k/leg); the **missing edge** is. |
| **Decay since GGR?** — has the famous edge faded? | `CONFIRMED` | The strategy's positive years cluster in **1983–2003**; the well-populated modern era is **mostly red** (worst: 2020 −1.2%, 2022 −2.7%, 2023 −1.9% monthly). The only modern green is in **dislocations** (2008 +0.9%/mo, Sharpe 1.30; 2019; 2025) — pairs trading as crisis insurance, not an everyday edge. |

> **In one sentence:** run honestly on a tradeable liquid basket, the parameter-free
> pairs-trading rule the tweet celebrates has **no convergence edge left** in the modern
> era — it's significantly negative *before* costs, market-neutral so there's nowhere to
> hide, and saddled with a −77% drawdown: a textbook the world arbitraged past, leaving
> the naive follower holding the tail of pairs that diverge and never come back.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

A [viral thread](https://x.com/MatiasScalbi/status/2063042609816252666) resurfaces the
most respectable anomaly in the book: **pairs trading**, the relative-value rule Nunzio
Tartaglia's quant desk ran at Morgan Stanley in the 1980s and Gatev, Goetzmann &
Rouwenhorst (GGR) published in 1999. Stated at full strength, the way its believers do:

> *"Find two stocks whose prices have moved together. When they diverge, short the winner
> and buy the loser — and wait. They've always snapped back. It made **~1.4% a month**,
> a **Sharpe near 0.6** against the market's 0.09, with **near-zero beta** — and it kept
> paying even **after** the paper told the world exactly how to do it."*

It is the steelman to beat, for a reason most folklore can't match: the rule is
**parameter-free**. No regression, no fitted threshold, no cherry-picked lookback — you
rank pairs by a single distance, trade a fixed 2-sigma trigger, and close on the cross.
That's precisely why "it still worked after publication" was such a striking claim: there
were no knobs to have overfit.

> 🔬 **For the quants** — H₁: the top-N minimum-distance pairs earn a **positive
> convergence return**, E[r] > 0, on **committed capital**, with an annualised Sharpe
> whose bootstrap CI clears zero, and a market **β ≈ 0**. The sharp version, H₁′: that
> this survives realistic bid-ask costs *and* the post-2002 decimalised, crowded regime.
> Null H₀: the spread is a non-stationary random walk dressed as mean reversion — the
> "convergence" is a coin flip, and the negative skew (occasional pairs that never
> reconverge) drags the mean to zero or below.

## 2 · So What?

Pairs trading is the **origin story of statistical arbitrage** — the strategy that built
the first quant desks and still anchors how market-neutral books are taught. If a rule
this simple, this public, and this *old* still printed ~1.4% a month at near-zero beta,
it would be the cleanest standing counterexample to "markets are efficient and edges die
on publication" — free money that survived being written down.

If instead it's been **arbitraged flat** — or worse — then it's the canonical case study
in **alpha decay**: a real inefficiency, found, published, crowded, and competed to death,
leaving a rule that still *looks* like it should work and quietly bleeds. That's the more
useful lesson, and the one the desk keeps meeting: *the cleaner and more famous the edge,
the more thoroughly the market has already eaten it.*

> 🔬 **For the quants** — the stakes are in the **skew**, not the mean. A convergence
> trade is short a straddle on the spread: many small wins when pairs snap back, rare
> large losses when one breaks (M&A, a guidance cut, an index reconstitution). The
> question is whether the win frequency × size still dominates the tail once the easy
> convergence has been competed away — i.e. whether what's left is a positive carry or
> just the **un-hedged short-gamma tail** nobody else wanted.

## 3 · How We'd Know

The trap is that a backtest of pairs trading *looks* alive even when it's dead: the win
rate stays above 50% (lots of little convergences), so the eye sees "it works" while the
mean quietly goes negative on the blow-ups. So we don't ask "did most trades win?" — we
ask the desk's usual sharper questions:

- **Does committed capital actually compound up?** Not per-trade hit-rate — the honest
  portfolio return on *all* the capital you set aside, winners, losers, and idle pairs
  alike. A rule can win 56% of trades and still lose money.
- **Is it positive *before* costs?** If the gross convergence return is already ≤ 0, no
  cost discussion is needed — there was never an edge. We charge costs only to size a
  real one.
- **Does it hold in the era we can actually test?** This universe is a *liquid basket*,
  not CRSP — and it only has enough names for tight pairs **after ~2004**. We headline
  the modern era and flag the thin early sample loudly, rather than quote a 1962 number
  formed from seven stocks.
- **Has it decayed?** The same rule, run year by year across 64 years, so the trend —
  not one lucky window — carries the verdict.
- **Is it even alpha?** A dollar-neutral book *should* have β≈0; we confirm it, because a
  β≈0 means the (negative) return isn't disguised market exposure — the verdict rests
  entirely on the convergence itself.

And the honesty rail this study leans on hardest: **the universe is the result.** Pair
quality depends entirely on how many names you form from, and that count is not constant —
so we report the eligible-name count and the selected pairs' tightness over time, and let
the reader see exactly when the test becomes fair.

> 🔬 **For the quants** — the shared desk protocol, powered by [`quantlab/`](../../quantlab/)
> and this study's [`pairs_trading/`](pairs_trading/): (1) form pairs by minimum SSD of
> normalized prices over 252 sessions, trade the next 126 at a 2σ open / zero-crossing
> close, **wait=1** (act on the close that triggers, earn from the next session); (2)
> bootstrap the Sharpe CI, run the year-by-year decay; (3) critique magnitude — committed
> vs employed capital, win-rate-vs-mean skew, the eligible-universe confound; (4) α-vs-β
> via a regression on the equal-weight tape; (5) cost sweep + square-root-impact capacity;
> (6) verdict. Engine: `data`, `pairs`, `backtest`, `robustness`.

## 4 · The Teardown

> *We ran the textbook rule over the cached 174-name universe (1962–2026, split-only).
> Headline = the modern era (2005–2026, eligible ≥ 60). Reproduce:
> [`examples/verify_real.py`](examples/verify_real.py); full tables in
> [`docs/results.md`](docs/results.md).*

- **The selector works — that's not the problem.** On the offline synthetic universe (true
  cointegrated twins hidden among noise), the minimum-SSD rule recovers **~85–100%** of the
  real twins and harvests their reversion for a **+0.95%/mo, Sharpe 1.7** — the machinery
  finds and trades genuine mean reversion when it exists. So a flat result on real data is
  a statement about the *market*, not a bug in the code.
- **On real pairs, the spread doesn't pay.** Modern-era committed capital earns
  **−0.37%/mo gross, −0.43%/mo net**, Sharpe **−0.44**. The win rate is **55.7%** — *more
  winners than losers* — and the mean is still negative: the textbook negative skew, big
  divergence losses swamping many small convergence gains.
- **It's negative even at zero cost.** The cost sweep starts at a 0 bp spread and the
  modern monthly net is already **−0.39%** (Sharpe −0.40). Costs make it monotonically
  worse (−0.74%/mo at a 40 bp half-spread), but they aren't the cause — *there is no edge
  for them to kill.*
- **The bounce isn't hiding anything either.** GGR's bid-ask-bounce control — wait extra
  days before executing — barely moves it (−0.43% → −0.56%/mo from wait 1→5). When a rule
  has real bounce profit, waiting bleeds it away; here there's no profit to bleed.
- **It's cleanly market-neutral — which is the bad news.** β = **0.00**, R² = **0.00**
  against the tape. The −0.43%/mo isn't disguised short-beta you could explain away; it's
  the convergence rule itself, losing on its own terms.
- **And it has decayed.** Run year by year, the positive prints cluster in **1983–2003**;
  the modern era is mostly red, worst in **2020 (−1.2%), 2022 (−2.7%), 2023 (−1.9%)**
  monthly. The only modern green years are **dislocations** — **2008 (+0.9%/mo, Sharpe
  1.30)**, 2019, 2025 — when everything mean-reverts at once. Pairs trading survives as
  *crisis insurance*, not as a standing edge.

> 🔬 **For the quants** — committed-capital daily series stitched across non-overlapping
> 126-session windows; Sharpe CI by 2,000-sample bootstrap (`quantlab.stats`); decay by
> calendar-year grouping of the daily P&L; neutrality by OLS on the equal-weight
> cross-section return; capacity by `impact_bps(N)=c·10⁴·√(N/ADV$)` solved at a nominal
> 20 bp edge (moot here — the realised edge is negative). Modern bootstrap CI **[−0.81,
> −0.04]** excludes zero on the *negative* side; full-sample **[−0.35, +0.13]** straddles
> it. Reproduce: [`examples/verify_real.py`](examples/verify_real.py).

<details>
<summary>🔬 The maths, in full</summary>

Each name's **normalized price** is the total-return index re-based to 1 at the formation
start, `p̃ᵢ(t) = ∏_{s≤t}(1+rᵢ,s)`. Formation distance is the sum of squared deviations
`SSD(i,j) = Σ_form (p̃ᵢ − p̃ⱼ)²`; the top-N smallest are selected, each carrying the
formation spread σ = std(p̃ᵢ − p̃ⱼ). In the trading window the normalized indices keep
compounding continuously, so the live spread is comparable to σ. A position opens the day
the spread first exceeds `k·σ` (k=2): short the rich leg, long the cheap leg, \$1 each
(dollar-neutral). It closes when the spread crosses zero, or at the window's end. With
execution lag `wait`, the position held on day t is the one implied by the close at
`t−wait`, so day-t P&L `= posₜ·(rₐ,ₜ − r_b,ₜ)` uses no look-ahead. A round trip pays
**four half-spreads** (two legs in, two out). Committed-capital portfolio return averages
all N pairs each day (idle pairs contribute 0) — the conservative GGR convention; employed
capital divides by deployed pair-days instead (reported alongside). The convergence trade
is structurally **short gamma on the spread**: bounded gains to the cross, unbounded loss
if the pair breaks — hence win-rate > 50% with a negative mean.

</details>

## 5 · The Verdict

> *The stamps, now earned.*

- **Signal — `NONE`.** The minimum-distance pairs do not reconverge into a profit. Modern
  gross is **−0.37%/mo** (Sharpe −0.44, bootstrap CI **[−0.81, −0.04]** — 98% of resamples
  negative), and it stays negative at a literal zero spread, so it isn't a cost artefact.
  The full sample is statistically indistinguishable from zero (CI [−0.35, +0.13]). My
  going-in prior was `WEAK` — a small, decayed-but-positive carry; the data was less kind.
- **Tradability — `MIRAGE`.** There's no edge to charge costs against, and costs deepen the
  loss anyway, monotonically, to **−0.43%/mo net** with a **−77% max drawdown**. The book
  is genuinely market-neutral (β≈0), so there isn't even a beta to bank. Capacity is large
  (~\$59k/leg before square-root impact bites a *hypothetical* 20 bp edge) — i.e. liquidity
  was never the constraint here, unlike Study 04's micro-caps. The constraint is that the
  thing doesn't work.
- **Decay since GGR — `CONFIRMED`.** Best years 1983–2003; modern era mostly red; green only
  in dislocations. Consistent with the literature (Do & Faff 2010): pairs-trading
  profitability fell sharply after 2002 as decimalisation narrowed spreads and stat-arb
  desks multiplied.

> 🔬 **For the quants** — decisive numbers in one place: modern committed monthly net
> −0.0043, gross −0.0037, Sharpe −0.44, CI [−0.81, −0.04], frac_neg 0.983, win-rate 0.557,
> median hold 56 sessions, max DD −0.77, β −0.00, R² 0.00; cost-sweep net at 0 bp = −0.0039;
> wait-rule flat −0.0043→−0.0047. All from [`docs/results.md`](docs/results.md), as-of
> 2026-06-01, fingerprint `ac25cb7061be`.

## 6 · Could You Trade It?

> *The honest money question — the beat that separates this desk from a backtest blog.*

You wouldn't, and the reason is unusually clean: **there is nothing to execute well.** Most
of this desk's mirages are real signals that costs or capacity erase — here the signal is
gone *before* the first cost, so there's no entry skill, venue, or sizing that rescues it.
The names are liquid mega-caps (median ADV ~\$148M), so unlike the micro-cap feed of Study
04, capacity is ample — which only sharpens the point: you *could* put real size on this,
and real size on a −0.4%/mo, −77%-drawdown, market-neutral bleed is just a slower way to
lose. The one place the rule earns its keep is the **2008-style dislocation**, where
everything reverts at once and it prints +0.9%/mo — but a strategy you can only trade in a
crisis is a hedge with a 60-day fuse, not a standing book.

The honest "what would it take" is therefore a *different strategy*, not better execution:
a **stop-loss** to cap the short-gamma tail (the naive rule holds losers to window-end —
that's most of the −77%), a **cointegration filter** so you trade pairs with an economic
anchor instead of whatever hugged tightest by luck, and a **far broader universe** so the
selected pairs are genuinely close. Those are the forks in beat 7 — and each is a concession
that the *textbook* rule, the one the tweet sells, doesn't clear the bar.

> 🔬 **For the quants** — break-even is moot (gross < 0). The relevant capacity line,
> `N* = ADV$·(edge_bps/(c·10⁴))²`, gives ~\$59k/leg only against an assumed 20 bp edge; at
> the realised negative edge it's undefined. The lived series is the committed-capital
> equity curve in [`docs/results.md`](docs/results.md), not the win-rate. Turnover is low
> (median hold 56 sessions), so it isn't costs-by-churn — it's direction.

## 7 · Going Further

> **The door this leaves ajar.** We killed the *naive, parameter-free* rule on a liquid
> basket — which is exactly the rule the viral thread sells. That does **not** prove modern
> stat-arb is dead; it proves the 1999 textbook recipe, unmodified, is. The interesting
> question is which single modern fix, if any, drags it back over the line — and on this
> universe none of the obvious ones is free.

- **A stop-loss.** The −77% drawdown is the un-capped short-gamma tail (losers held to
  window-end). Does a hard stop turn the skew survivable — or just lock in the losses faster?
- **A cointegration gate.** Replace "tightest SSD" with an actual Engle–Granger / Johansen
  test, so a pair needs an economic reason to revert, not just a lucky formation year.
- **A broader, sector-aware universe.** GGR formed from *thousands* of names; we had ~170.
  The PR that matters most: point `data.load_universe` at a real S&P 1500 cache and re-run.
- **Total-return prices.** We ran split-only (the cached mode); dividends fold a small,
  named bias *against* the rule — a total-return rerun is the clean robustness check.
- **The crisis-only book.** The only green is in dislocations — is "pairs trading as a
  conditional 2008/2020 hedge" a real, sizeable thing, or just three lucky years?

The deep version — the synthetic validation, the bootstrap, the decay curve, the cost and
neutrality teardown — is in [`notebooks/02_for_the_quants.ipynb`](notebooks/).

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full method: formation, the trade, decay, neutrality, costs |
| [`docs/results.md`](docs/results.md) | **the real run** — every headline table, fingerprinted and as-of'd |
| [`docs/references.md`](docs/) | sources + literature map (GGR 1999, Do–Faff 2010, the decay literature) |
| [`pairs_trading/`](pairs_trading/) | the study package: `data` · `pairs` · `backtest` · `robustness` |
| [`examples/`](examples/) | [`run_synthetic_demo.py`](examples/run_synthetic_demo.py) (offline) · [`verify_real.py`](examples/verify_real.py) (the real run) |

Every number is produced by [`pairs_trading/`](pairs_trading/), in the house style of the
shared [`../../quantlab/`](../../quantlab/) engine; `pytest` covers it in CI.
