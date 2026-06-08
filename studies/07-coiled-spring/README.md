# Study 07 — Coiled-Spring 🌀 — does a stock resting on its 20-EMA really spring into an explosive breakout?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Studies 02–04 hunt a trigger in the **price**, the **vol**, or the **information
> flow**, and Study 05 lives in the **cross-section**, this one tests a **chart-pattern
> rule** straight out of a retail trading book — the kind of thing sold on a handful of
> beautiful winners.*

## Verdict — read this first

*Measured on a **reproducible**, costed backtest over a cached, liquid **174-name** US
universe, 1962–2026 (split-only prices; daily returns winsorized at ±100% to kill bad
prints). The rule is mechanised verbatim from the book — 20-EMA pivot, an EMA-holding
pullback, a **2× volume** breakout — with **no fitted parameter**. The headline test is
exit-agnostic: the forward return after a breakout vs a **random entry in the same stock**
over the same horizon. As-of 2026-06-01, price fingerprint `42590aa02dc9`; every number in
[`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — does the breakout beat just holding the same name? | `WEAK` | Buying the breakout and holding 10 days beats a random same-stock entry by **+1.2%** (20 days: **+1.5%**), but the HAC *t* is only **2.0–2.3** and **0** at 5 days — a *whisper* of real short-term momentum, not the book's fireworks, and not corrected for the universe of TA rules one could have searched. |
| **Tradability** — does it survive costs, capacity, scale? | `FRAGILE` | The tradable rule nets **+0.58%/trade** after 15 bps — but the **median trade *loses* (−0.25%)** and the win rate is **41%**; the positive average is a thin right tail (per-trade Sharpe **0.05**, CI [0.002, 0.090]). It's really **long-momentum-regime beta** (best years 2000 +4.3%, 2020 +4.5%, 2024 +2.6%; worst 2008 −3.6%, 2013 −3.2%) and **dies by ~75 bps round-trip** — a cost level the *small-caps the book actually trades* would blow straight through. |
| **Explosive as advertised?** — the +30-50%-in-days promise | `BUSTED` | Across all **1,674** breakouts, the share that gross **≥ +30%** is **1.7%** — a **1-in-60 lottery**, not the base rate. Median hold is **3 sessions**, median outcome a small loss. The book's pitch is the cherry-picked tail. |

> **In one sentence:** the 20-EMA pivot breakout carries a faint, real pulse of short-term
> momentum — about **+1% over ten days** beyond simply holding the stock — but the book's
> "explosive +30-50% in days" is a **1-in-60 tail event sold on survivors**: the median
> trade loses, the positive average is bull-market beta, and on the small names it targets,
> ordinary costs finish it.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

> *"A stock that pulls back to its rising 20-EMA and then breaks its pivot high on big
> volume is about to explode — +30 to +50% in 6 to 10 days."*

The claim comes from Jayesh Shah's self-published *Trade the 20 EMA* (see
[`docs/references.md`](docs/references.md)). It is admirably specific — three hard steps,
no indicator soup:

1. **Form a 20-EMA pivot.** A stock trading *below* its 20-period EMA breaks above it and
   carves a high, then turns down. That high is the "pivot".
2. **The pullback holds the EMA.** Price drifts back toward the EMA but **must not close
   below it** — "if it does, drop the trade and look elsewhere" (the book's hard rule).
3. **Buy the pivot breakout** — the moment price closes back above the pivot high, **on
   volume at least twice the prior month's average**. Stop one tick under the breakout
   bar's low; trail out, or scale half off at +10%.

The evidence in the book is a gallery of gorgeous winners — UCOBANK, SKIPPER, FACT — each
running 40–70% in a fortnight. No losers. No costs. No out-of-sample test. That gallery is
exactly what this study replaces with the *whole* distribution.

> 🔬 **For the quants** — H₁: the forward *k*-day return following a qualifying breakout,
> bought at the next open, exceeds the same stock's ambient drift over *k* days (a random
> same-stock entry) by a margin with HAC *t* > 2; and H₁′ (the book's strong form): a large
> mass of breakouts deliver ≥ +30% within ~10 sessions. We test both, exit-agnostic first.

## 2 · So What?

If true, it would be a gift: a mechanical, screenable rule that turns a few days of
patience into 30%+ — repeatable across any liquid market, no Bloomberg terminal required.
It would also be a genuine dent in weak-form market efficiency: a *pure chart pattern*,
visible to anyone, that reliably front-runs an explosive move. Either retail technical
analysis has a real edge hiding in plain sight, or — far more likely, given fifty years of
literature — the rule's allure is survivorship: the winners get screenshotted into a book,
the losers and the stop-outs quietly don't.

> 🔬 **For the quants** — a real +1.2%/10-day excess at scale would annualise to a fat
> Sharpe *if it were consistent*; the entire question is whether it's a stable conditional
> mean or a fat-tailed lottery whose average is a handful of jackpots. The verdict hinges
> on the **median**, the **win rate**, and the **break-even cost**, not the mean.

## 3 · How We'd Know

We mechanise every word of the rule (no eyeballing "looks coiled"), run it over a 174-name
liquid US universe back to 1962, and **pre-register what would make us call mirage** — the
desk doesn't move goalposts:

- **Signal `NONE`** if the forward 10-day return after a breakout is statistically
  indistinguishable (bootstrap CI straddles zero, HAC *t* < 2) from a **random entry in the
  same stock** over the same horizon — i.e. the breakout adds nothing beyond "this name
  drifts up."
- **Tradability `MIRAGE`** if, after realistic round-trip costs, the mean net per trade
  ≤ 0, or the equity is just disguised market/momentum beta.
- **Explosive `BUSTED`** if the +30% outcome is a tail event rather than the base rate.

The traps we watch for: **look-ahead** (a pivot is a local max — knowable only bars later,
so the breakout is sought strictly after confirmation); **survivorship** (we count every
breakout, winners *and* losers, including stop-outs); and **selection/data-snooping** (the
parameters are the book's, not fitted — but a single TA rule still rides an implicit search
the literature warns about, so a marginal *t* is *generous*, not conservative).

> 🔬 **For the quants** — protocol (shared desk rubric):
> 1. **Measure** the raw effect exactly: per-signal forward returns at *k* = 5/10/20, no
>    stop, no fitting.
> 2. **Robust inference** — Newey-West (HAC) *t* on the per-name *excess* over a 20×
>    random-entry baseline; bootstrap on the per-trade Sharpe.
> 3. **Critique the magnitude** — the full net-return distribution, the +30% hit rate, the
>    median vs the mean.
> 4. **Alpha vs beta** — decay-by-year exposes the bull-regime clustering.
> 5. **Execution & capacity** — round-trip cost sweep to break-even; per-name $-volume cap.
> 6. **Verdict** — the three stamps.
>
> Engine used: `quantlab.analytics.mean_tstat_hac`, `quantlab.stats.sharpe_ci_bootstrap`,
> `quantlab.repro` (as-of + fingerprint). Study code in [`coiled_spring/`](coiled_spring/).

## 4 · The Teardown

> *We run it. Here's what the data actually says.*

- **The breakout day carries a whisper — not a bang.** Buying at the next open and holding,
  the average breakout returns **+0.97% / +2.09% / +3.53%** at 5 / 10 / 20 sessions. Netted
  against a random entry *in the same stock*, the **excess** is **+0.53% / +1.20% / +1.54%**
  — real-direction, but small, and only marginally significant (HAC *t* = **1.79 / 2.05 /
  2.26**; at 5 days it's indistinguishable from drift).
- **The "explosive move" is a 1-in-60 lottery.** Of 1,674 breakouts, **1.7%** gross ≥ +30%.
  The **median net trade is −0.25%** and the average hold is **3 sessions**. The book's
  gallery is the right tail, presented as the centre.
- **The typical trade loses; a few jackpots carry the mean.** Win rate **41%**, average win
  **+7.1%**, average loss **−3.9%** (payoff 1.82), best **+194%**, worst **−52%**. Mean net
  **+0.58%/trade**, per-trade Sharpe **0.05** (bootstrap CI [0.002, 0.090], 2.3% of
  resamples negative) — barely off zero.
- **Costs reach break-even around 75 bps round-trip.** Mean net falls from **+0.73%** at
  zero cost to **−0.07%** at 80 bps. On large caps 15 bps is cheap; on the *small-caps the
  book actually trades* (tiny float, the very names that pop 40%), realistic spread +
  impact lives near or past that break-even.
- **The "edge" is bull-regime beta.** The best years are momentum blow-offs — **2000
  +4.3%, 2020 +4.5%, 2024 +2.6%** monthly-equivalent per trade — and the worst are
  chop/bear — **2008 −3.6%, 2013 −3.2%, 2022 −1.6%, 2026-YTD −2.8%**. No era where it's a
  stable standalone edge; it makes money when *everything* is breaking out.

> 🔬 **For the quants** — the headline excess is computed per name (breakout forward return
> minus that name's mean random-entry forward return over the same horizon), pooled, then
> HAC-tested — so a name that simply trends can't manufacture the excess. The bootstrap CI
> on the per-trade *Sharpe* of the net ledger is [0.002, 0.090]: positive, but the lower
> bound is a rounding error from zero, and no multiple-testing penalty has been applied.
> Reproduce via [`examples/verify_real.py`](examples/verify_real.py) → [`docs/results.md`](docs/results.md).

<details>
<summary>🔬 Why the median matters more than the mean here</summary>

The net-return distribution is sharply right-skewed: percentiles
[5%, 25%, 50%, 75%, 95%] = [−11.3%, −2.4%, −0.25%, +2.0%, +13.5%]. The mean (+0.58%) sits
above the 60th percentile — it is *made* by the thin tail beyond +13%, not by a typical
outcome. A trader doesn't get the mean; they get a string of small losses punctuated by
rare jackpots, which is precisely the equity curve that *feels* like the book until the
jackpots don't arrive. This is why the verdict leans on the median, the win rate, and the
break-even cost rather than the flattering average.

</details>

## 5 · The Verdict

- **Signal · `WEAK`.** There *is* a small, real directional pulse — a 20-EMA pivot breakout
  beats a random same-stock entry by ~+1.2% over ten days (HAC *t* ≈ 2). But it is an order
  of magnitude short of the claim, marginal once you remember the implicit rule-search, and
  vanishes at the 5-day horizon. A whisper of short-term momentum, dressed up as a system.
- **Tradability · `FRAGILE`.** It survives 15 bps on large caps with a hair-thin
  +0.58%/trade — but the median trade loses, the average is a fat-tailed artefact (Sharpe
  0.05), the returns cluster in momentum-bull years, and break-even cost is ~75 bps, right
  where the small, illiquid names the book targets actually trade.
- **Explosive as advertised? · `BUSTED`.** 1.7% of trades do the advertised +30%. The
  promise is the survivor's tail.

> 🔬 **For the quants** — decisive numbers in one place: excess forward return +1.20%/10d
> (HAC *t* = 2.05); per-trade net Sharpe 0.05 (CI [0.002, 0.090]); median net −0.25%, win
> rate 40.7%; +30% hit rate 1.7%; break-even round-trip ≈ 75 bps; bull-regime decay
> (2000/2020/2024 best, 2008/2013 worst). Fingerprint `42590aa02dc9`.

## 6 · Could You Trade It?

Walk it from "interesting print" to live P&L. You'd screen for names just above a rising
20-EMA, wait for the volume-gated pivot break, buy the next open, and trail a stop. On
**large caps** the capacity is fine (median traded name does **$98M/day**; a 1% participation
cap is ~$1M per name) and 15 bps costs leave the thin +0.58%/trade *technically* alive — but
a per-trade Sharpe of 0.05 on a 41%-win, fat-tail rule is not a book a desk would run; one
dry spell of jackpots and you're underwater on a coin flip. On the **small caps where the
book's 40% pops actually happen**, the math inverts: spreads and market impact on a
low-float name spiking 3× volume push round-trip costs toward — and past — the ~75 bps
break-even, so the *only* trades with the upside are the ones costs eat. The candid bottom
line: there's a faint momentum continuation you could *almost* monetise on liquid names
where the payoff isn't there, and a juicy payoff on illiquid names where the costs aren't
survivable. That gap is where the book lives.

> 🔬 **For the quants** — break-even ≈ 75 bps vs a +73 bps gross edge means the strategy is
> a *cost-toggle* away from zero on large caps; square-root impact on a small-cap breakout
> (participation forced high by the volume spike that defines the signal) plausibly doubles
> effective cost. Decay-adjusted, the net Sharpe is indistinguishable from a long-momentum
> tilt you could harvest more cheaply with a plain momentum factor.

## 7 · Going Further

- **Total-return rerun.** We run split-only (the cached mode); dividends are second-order on
  short holds but a `total_return` pass is the clean robustness PR.
- **A genuinely small-cap universe.** The honest test of the book is the float-starved names
  its examples come from — where the upside *and* the costs both live. Our liquid cache
  understates both; a Russell-microcap parquet set would sharpen the tradability verdict.
- **Let winners run.** Our trailing stop exits in ~3 sessions; the no-stop forward-return
  test (beat 4) is the steelman "let it run" version and is still weak — but a half-off-at-
  +10%-and-trail variant (the book's favourite) is worth a `backtest.ExitRules` sweep.
- **Reality Check.** Apply White (2000) across a grid of EMA spans / volume gates to price
  the implicit rule-search; we expect the marginal *t* ≈ 2 to fall below significance.
- **What to PR:** a small-cap universe, the total-return rerun, the exit-rule sweep, or a
  momentum-factor control regressing the ledger on UMD to see how much "edge" is just UMD.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`coiled_spring/signals.py`](coiled_spring/signals.py) | the mechanised rule — EMA, causal pivots, the EMA-holding pullback, the volume-gated breakout |
| [`coiled_spring/backtest.py`](coiled_spring/backtest.py) | enter-next-open, stop, trailing exit; the per-trade ledger + the exit-agnostic forward-return test |
| [`coiled_spring/robustness.py`](coiled_spring/robustness.py) | breakout-vs-random-entry (the headline), bootstrap, cost sweep, decay-by-year, capacity |
| [`coiled_spring/data.py`](coiled_spring/data.py) | the cached real universe + a synthetic one with **planted springboards** to recover offline |
| [`examples/verify_real.py`](examples/verify_real.py) | the headline run → [`docs/results.md`](docs/results.md) (as-of + fingerprint) |
| [`notebooks/`](notebooks/) | `01_for_the_curious` (the story) and `02_for_the_quants` (the teardown), same seven beats |
| [`docs/references.md`](docs/references.md) | the book + the TA literature it walks into |

The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
