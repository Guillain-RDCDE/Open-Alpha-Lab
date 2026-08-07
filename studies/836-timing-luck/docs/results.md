# Results — Study 836 (Rebalance Timing Luck): the same strategy, a different Sharpe

*Generated from [`timing_luck/`](../timing_luck/) on a **deterministic, offline synthetic
panel** (seed 836) by [`examples/verify.py`](../examples/verify.py). This is a research-method
demo, so the tape is built on purpose: the **null** (`mom_edge = 0`) is a cross-section where a
momentum sort has **zero genuine edge**, and the **positive control** (`mom_edge = 1`) plants a
real momentum premium. Real free data can never certify "zero edge", so there is no real-tape
stamp — the study is capped at `NONE` on the SIGNAL axis. Null panel: 30 assets × 2,600 daily
rows (2012-01-02 → 2021-12-17). Stamped as-of 2026-06-30. Fingerprint `abbc37b5f962`
(null panel, seed 836, momentum lookback 126d, period 21d, top/bottom 30%).*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Does rebalance timing break inference?" `CONFIRMED`

We take **one** monthly cross-sectional momentum book — 6-month lookback, long the top 30% /
short the bottom 30%, dollar-neutral, rebalanced every 21 trading days — and run it **21 times**,
once for each rebalance **offset** (day 0, day 1, …, day 20 of the cycle). Every run trades the
*identical* rule on the *identical* tape. The only difference is *which day of the month* it
rebalances.

- The **luckiest** offset (#14) prints an annualised Sharpe of **+0.168**; the **unluckiest**
  (#1) prints **−0.242**. Same strategy, same data — a **phantom gap of 0.410 Sharpe units**
  (sd across offsets **0.111**), the offsets averaging **−0.017**. An analyst who happened to
  rebalance mid-month would call this book a modest winner; one who rebalanced early would fire
  it. Both are looking at the same rule.
- The gap is **luck, not skill**: rank the offsets by Sharpe in the first half of the sample and
  the second half, and the Spearman rank correlation is **+0.044** — essentially zero. The
  offset that was lucky yesterday is a coin-flip tomorrow; across 25 seeded worlds the *identity*
  of the best offset is scattered uniformly across 0…20. There is nothing to forecast.
- The **fix works**: tranch / overlap all 21 offsets into a single book (rebalance a slice every
  day) and the dispersion is **gone by construction** — one curve, Sharpe **−0.019** (NW *t* =
  **−0.06**). On the null there was never any edge; tranching just removes the phantom noise the
  offset choice was adding.

So `NONE` on the signal axis (a synthetic-only demo — the offset dispersion is luck, and even the
dispersion-free tranched book finds no edge), `MIRAGE` on tradability (the lucky offset does not
persist, so there is nothing to harvest, and the book loses net of costs), and `CONFIRMED` on the
myth-check (yes — an arbitrary rebalance-date choice genuinely swings the reported Sharpe by ~0.41
units on the identical strategy, and tranching collapses it).

## Data stamp

- **Null panel** (`mom_edge = 0`, no momentum edge): 30 assets × 2,600 daily rows,
  2012-01-02 → 2021-12-17, fingerprint `abbc37b5f962`, seed 836. Common market factor (beta = 1
  for all names, so a dollar-neutral book cancels it exactly) + idiosyncratic noise.
- **Positive-control panel** (`mom_edge = 1`): same generator with a persistent latent trend that
  genuinely predicts forward returns — a real momentum premium.
- **Strategy:** trailing-126-day momentum, long top 30% / short bottom 30%, equal-weight,
  dollar-neutral, rebalanced every **21 days**; signal known at the close of `d-1`, held from `d`.

## The headline — one book, 21 rebalance offsets (null tape, 2,453 common days)

Per-offset annualised Sharpe of the **identical** momentum long-short:

| | offset 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| **Sharpe** | −0.15 | −0.24 | −0.04 | −0.09 | 0.06 | −0.13 | 0.01 | −0.03 | 0.04 | 0.13 | 0.03 | −0.04 | −0.00 | 0.08 | **0.17** | 0.03 | 0.06 | 0.14 | −0.02 | −0.13 | −0.21 |

| quantity | value |
|---|--:|
| **luckiest offset** (#14) | Sharpe **+0.168** |
| **unluckiest offset** (#1) | Sharpe **−0.242** |
| **phantom gap** (luckiest − unluckiest) | **0.410 Sharpe units** |
| dispersion (sd across offsets) | 0.111 |
| average across offsets | −0.017 |

The same rule, the same tape — and a **0.41-Sharpe** swing that is nothing but the calendar
accident of the rebalance day.

## Luck, not skill — the lucky offset does not persist

| test | value | reading |
|---|--:|---|
| Spearman rank corr of offset Sharpes (first half vs second half) | **+0.044** | ≈ 0 ⇒ the lucky offset is unforecastable ⇒ **pure luck** |

If the offset dispersion carried real information, the ranking would persist. It does not — the
best offset in-sample is a coin-flip out-of-sample.

## The fix — tranching / overlapping portfolios collapses the dispersion

| book | Sharpe | mean bps/day | NW(10) *t* | dispersion |
|---|--:|--:|--:|--:|
| pick one offset (the game) | −0.24 … **+0.17** | — | — | **0.410** |
| **tranched / overlapping** (all 21) | **−0.019** | −0.079 | **−0.06** | **0.000** |

Averaging the 21 offset books into one overlapping portfolio leaves **a single curve** — there is
nothing left to be lucky about, so the dispersion is zero. On the null there was no edge to begin
with; the tranched book confirms it (NW *t* = −0.06).

## The timer — can you get paid for the tranched book? (null tape)

Charge one-way × NAV on the slice rotated each day (≈ 2/period of NAV), plus 50 bps/yr borrow on
the short leg:

| one-way cost | gross/day | cost/day | net/day | Sharpe (net) | *t* (net) |
|---|--:|--:|--:|--:|--:|
| **1 bp** | −0.079 bps | 0.327 bps | **−0.406 bps** | −0.097 | −0.30 |
| **5 bps** | −0.079 bps | 1.089 bps | **−1.168 bps** | −0.278 | −0.87 |

There is nothing to harvest: the null book is flat gross and loses net, and the *only* way to
"beat" it — picking the lucky offset — does not persist. **Mirage.**

## Synthetic positive control — the machinery detects a PLANTED premium (25 seeds each)

The same protocol on the null and on a tape with a genuinely planted momentum premium, averaged
over 25 seeds (the house rule):

| world | phantom Sharpe spread | offset rank corr | tranched Sharpe | tranched NW *t* | \|t\| ≥ 2 |
|---|--:|--:|--:|--:|--:|
| **null** (`mom_edge = 0`) | **0.442** | −0.088 | **−0.047** | −0.147 | **0/25** |
| **planted** (`mom_edge = 1`) | **0.447** | +0.003 | **+1.387** | **+4.244** | **24/25** |

Two things hold. (a) The **phantom dispersion is present in both worlds** (≈ 0.44 Sharpe units)
— it is an artefact of *when* you rebalance, entirely independent of whether there is real edge
underneath. (b) The **tranched book is silent on the null** (fires 0/25) and **robustly positive
when a real premium is planted** (Sharpe 1.39, NW *t* 4.24, fires 24/25) — the machinery detects
genuine edge and is not itself the artefact. The best-offset identity across the 25 null seeds is
scattered uniformly over 0…20: the winner is a coin-flip. *(A faithful-engine / power check only —
never cited in support of a real-tape stamp.)*

## Why the verdict is what it is

1. **The offset dispersion is luck, and there is no real edge to find.** On a tape built to have
   zero momentum edge, the 21 offsets swing the Sharpe by 0.41 units, the lucky offset does not
   persist (rank corr +0.04), and the dispersion-free tranched book finds nothing (NW *t* −0.06).
   A synthetic-only demo — no real tape, so never `REAL`. **Signal `NONE`.**
2. **Nothing to trade.** You cannot harvest a dispersion whose winner is unforecastable; the
   tranched null book is flat gross and loses net of costs. **Tradability `MIRAGE`.**
3. **The pitfall is real.** An arbitrary rebalance-date choice genuinely swings the reported
   Sharpe by ~0.41 units on the *identical* strategy — enough to hire or fire it — and tranching
   / overlapping portfolios collapses the dispersion to a single curve. **`CONFIRMED`.**

## The honest takeaway

A Sharpe ratio is only as honest as the arbitrary construction choices behind it. The rebalance
*date* — supposedly an innocuous detail — swings the reported Sharpe of the identical rule by
~0.4 units, enough to flip a verdict from "hire" to "fire", and the swing is pure luck: the lucky
offset is a coin-flip out-of-sample. The fix is free and public — tranch the rebalance across
every offset (overlapping portfolios) and the phantom dispersion vanishes while any genuine
premium survives. `NONE` × `MIRAGE`, myth `CONFIRMED`. This is a method demo on a synthetic world
by design — it can never earn `REAL`, which requires a robust *t* ≥ 2 on a real tape.
