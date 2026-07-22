# References & literature map — Study 794 (Commodity Carry)

## The claim under test

- **The folklore / factor.** *Backwardated commodities (positive roll yield / carry)
  out-earn contangoed ones, cross-sectionally.* Carry — "the return you'd earn if the
  spot price never moved" — is, for a commodity future, essentially the roll yield: the
  slope of the term structure. A downward-sloping (backwardated) curve means the long
  rolls from an expensive expiring contract into a cheaper deferred one and banks the
  difference; an upward-sloping (contango) curve is a roll *drag*. The cross-sectional
  premium says: sort commodities by carry, go long the backwardated and short the
  contangoed, and collect.
- **The academic anchor.** The commodity carry / roll-yield premium is one of the most
  documented in the space:
  - Gorton & Rouwenhorst (2006, *Facts and Fantasies about Commodity Futures*, FAJ) —
    the average-return properties of a broad commodity-futures cross-section.
  - Erb & Harvey (2006, *The Strategic and Tactical Value of Commodity Futures*, FAJ) —
    show the roll return, not the spot return, dominates long-run commodity-futures
    performance and that term-structure sorts (backwardation vs. contango) separate
    winners from losers.
  - Koijen, Moskowitz, Pedersen & Vrugt (2018, *Carry*, JFE) — unify carry across asset
    classes; commodity carry (the futures basis) earns a significant cross-sectional
    premium in their broad sample.
- **The open question we test.** Does that cross-sectional carry signal survive on the
  narrow slice a *free* researcher can actually see — a two-name energy proxy (WTI,
  Henry Hub gas), with the carry read from the real EIA futures term structure and the
  returns taken from investable ETFs? A two-name cross-section is the thinnest possible;
  the honest expectation is a badly under-powered test, and the verdict must reflect the
  number that comes out, not the strength of the literature.

## What we measure, and the honesty rails

- **Carry is measured ex-ante.** For each commodity, the monthly annualized log roll
  yield `ln(C1/C2)·12` from the EIA near-month curve settle on the last session of the
  month `t` — a public, point-in-time number. `C1 > C2` (backwardation) => positive
  carry. (The estimator also accepts a wider `C1`-vs-`C_k` slope where deeper contracts
  are cached; the headline uses the front-pair `C1/C2`.)
- **One documented execution lag.** The curve settle is known after the close of month
  `t`; the position is entered at the first session of month `t+1` and held to that
  month's last session. The outcome return is therefore strictly *forward* of the
  signal — zero look-ahead.
- **The pooled panel test is the workhorse.** With only two names, a cross-sectional
  sort is degenerate, so the primary estimator cross-sectionally demeans *both* carry
  and forward return within each month (removing the common energy move) and regresses
  demeaned return on demeaned carry with a **Newey-West (6-lag)** HAC `t`. That slope is
  the pure cross-sectional carry→return relationship and is the decisive real-tape
  number.
- **The roll mechanism is validated directly on the investable tape.** The
  front-minus-laddered ETF gap on the *same* crude (USO − USL) is the realized roll; we
  regress it on WTI carry — in backwardation the front should beat the ladder — as a
  mechanism check independent of the sort.
- **Costs, borrow, and worst case are charged.** The timer's costs are one-way bps ×
  turnover × NAV per leg, the short leg pays an explicit borrow, and the worst single
  month is reported. A deterministic seeded **synthetic control** with a tunable planted
  carry premium proves the machinery is unbiased (the null must not fire across 20
  seeds) — it is never cited in support of the real-tape stamp.

## The proxy limitation (stated plainly)

Clean, broad historical commodity term structure (a full cross-section of agriculture,
metals, softs and energy futures curves) is **not freely available**. This study is a
**two-name energy proxy** — WTI crude and Henry Hub natural gas — for the general
cross-sectional commodity carry premium. Two names is the thinnest cross-section
imaginable; the cross-sectional numbers here are under-powered by construction and can
neither confirm nor refute the broad-universe factor documented in the literature. They
test one honest question: does the carry signal separate winners from losers on the
liquid energy slice a free researcher can actually see?

## Data sources

- **Futures term structure (carry signal)** — EIA v2 daily near-month energy futures:
  WTI `RCLC1..4` ($/bbl), Henry Hub `RNGC1..4` ($/MMBtu). Free, no key beyond
  `DEMO_KEY`. Cached under `_cache/cc_energy_curve.csv`.
- **Investable returns (outcome + roll proxy)** — yfinance total-return closes: USO
  (WTI front), USL (WTI 12-month ladder), UNG (gas front), UNL (gas 12-month ladder).
  Cached under `_cache/cc_etfs.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [35-contango](../35-contango/) — measures the **realized roll drag** as the
  laddered-minus-front ETF return gap (USO/USL, UNG/UNL) and uses a *synthetic* bucket
  for the cross-section. It grades the drag itself (Weak/Mirage). Study 794 instead uses
  the **real EIA futures curve slope as the ex-ante carry SIGNAL** and asks whether that
  signal *predicts* which investable commodity out-earns — the forward, cross-sectional
  carry-premium test, not a measurement of the drag.
- [380-curve-roll-down](../380-curve-roll-down/) — the roll-down / carry premium in a
  **single** asset (a Treasury duration sleeve), on the yield curve. Single-asset, fixed
  income; not a cross-sectional commodity sort.
- [660-carry-everywhere](../660-carry-everywhere/) — a **multi-asset blend** (FX + bond
  + equity + commodity carry, equal-weighted). Its commodity leg is one of four; Study
  794 isolates the commodity leg alone and reads its carry from the actual futures curve
  rather than an ETF-gap proxy.
- [661-uso-roll-decay](../661-uso-roll-decay/) — the **single-fund** USO-vs-spot roll
  decay (does USO structurally lose to headline oil?). It is a one-name time-series
  decay story; Study 794 is a cross-sectional carry sort across two commodities.

None of the siblings run a forward, cross-sectional, ex-ante-carry→return test on the
real energy term structure — that is this study's own axis.
