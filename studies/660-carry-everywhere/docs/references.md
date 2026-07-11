# References & literature map — Study 660 (Carry-Everywhere)

## The claim under test

- **The academic anchor.** Koijen, Moskowitz, Pedersen & Vrugt (2018, *Carry*, Journal
  of Financial Economics) define carry, for any asset, as "the return you would earn
  if prices stay the same" and show that a carry signal built this way predicts
  returns in **equities, government bonds, currencies, commodities, US Treasury bond
  futures, credit and options** — not just the textbook FX carry trade. They document
  positive, low-pairwise-correlation carry premia across asset classes and a
  diversified "carry everywhere" factor with a materially higher Sharpe than any
  single sleeve.
- **The building blocks, one per sleeve.** FX: Meese & Rogoff (1983) on uncovered
  interest parity's empirical failure; the classic AUD/JPY, NZD/JPY "carry pairs"
  literature (Menkhoff, Sarno, Schmeling & Schrimpf 2012, and see sibling
  [364-fx-carry-trade](../364-fx-carry-trade/)). Bonds: the term-spread /
  expectations-hypothesis-failure literature (Fama & Bliss 1987; Campbell & Shiller
  1991) — being long duration, funded short, is a term-premium harvest. Equities: the
  dividend-yield / value literature (Fama & French 1992/1993) — KMPV use dividend
  yield as their equity carry proxy, which mechanically overlaps with the classic
  value factor. Commodities: the theory of storage and roll yield (Keynes 1930's
  "normal backwardation"; Gorton & Rouwenhorst 2006; Erb & Harvey 2006) — the return
  from the futures curve's slope, independent of the spot price.
- **The honest caveat, stated up front.** KMPV's headline sample is long (decades),
  global (dozens of instruments per asset class) and uses live deposit rates and
  futures curves. Ours is one free-data proxy per asset class, over one 19-year
  window (2007-07 → 2026-06) — this is a **replication attempt on a coarser, shorter,
  free tape**, not a critique of the original paper's own (much richer) sample.

## What we measure, and the honesty rails

- **Four static, ex-ante-fixed sleeves**, not a monthly-re-ranked cross-section —
  the free tape has no daily deposit-rate panel or futures-curve history, so, like
  sibling studies 364/612/638, we use a **transparent fixed proxy**: FX long
  AUD+NZD / short JPY+CHF (the textbook high-yielder/funder pairing), bond long
  IEF / short SHY (term-spread trade), equity long VYM / short VUG (dividend-yield
  tilt), commodity long DBC / short GSG (roll-yield isolation via Invesco's
  "Optimum Yield" methodology against a naive front-month roll). Because the
  composition never changes, there is **zero look-ahead** to document (the one
  convention worth stating: monthly rebalance to par weights at the month-end
  close).
- **Newey-West (1987) HAC** *t* on each sleeve's and the combo's one-sample mean
  (monthly returns are autocorrelated; a naive *t* would overstate significance).
  A **circular block-bootstrap** (block = 6 months, 2,000 draws) gives the combo
  Sharpe a confidence interval rather than a point estimate treated as gospel.
- **The equity-carry confound, named plainly.** Dividend yield is KMPV's own equity
  carry proxy, but on any *single* long/short pair it is also, mechanically, a value
  tilt — and 2007-2026 contains the largest secular growth-over-value divergence in
  decades. We do not disguise this: the results explicitly flag that the EQ leg's
  weakness likely reflects that megacap-growth regime more than a "carry" failure
  per se.
- **The crisis-window test is the brief's own ask, run honestly both ways.** We
  hardcode the two textbook "carry unwind" windows (2008 GFC, 2020 COVID) as facts
  (no fitting) and report the combo's cumulative return **and every individual leg's**
  — so a reader can see exactly why the combo did *not* crash the way a synchronized
  "carry factor" story predicts, even though individual legs moved violently.

## Data sources

- **FX spot** (`AUDUSD=X`, `NZDUSD=X`, `JPY=X`, `CHF=X`), **Treasury ETFs** (`IEF`,
  `SHY`), **dividend/growth equity ETFs** (`VYM`, `VUG`), **commodity ETFs** (`DBC`,
  `GSG`) and the **cash reference** (`BIL`) — yfinance total-return (dividend-adjusted)
  daily closes, no key, cached under `_cache/` (`cev_closes.csv`), 2007-06-01 →
  2026-06-30.
- Invesco DBC methodology ("Optimum Yield," designed to reduce contango drag /
  harvest backwardation relative to a naive front-month roll):
  https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=DBC
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [364-fx-carry-trade](../364-fx-carry-trade/) — the **FX carry leg alone**, done
  properly: a full G10 cross-section, its own HAC *t* (1.14), skew (−0.75) and
  crash-decile analysis. This study's FX sleeve is a coarser 2-vs-2 static basket;
  we cite 364's own verdict rather than re-deriving FX carry from scratch. 364 is
  **one ingredient** here, not the subject.
- [147-fx-momentum](../147-fx-momentum/) — a *different* FX signal entirely (12-1
  trailing return, i.e. trend/momentum, not the rate differential). No overlap in
  signal construction; both happen to trade G10 spot.
- [612-em-debt-carry](../612-em-debt-carry/) — a **single packaged-carry sleeve**
  (EM sovereign hard-currency coupon pickup vs Treasuries), with its own
  promised-vs-collected decomposition and equity-beta regression. It is one
  asset-class carry story done in depth; this study is four asset classes done
  more shallowly, combined.
- [638-value-momentum-everywhere](../638-value-momentum-everywhere/) — the sibling
  **"everywhere" combo study**, same multi-sleeve architecture (four asset classes,
  equal-weight combo, HAC *t*'s, a synthetic control) but for a **different signal
  pair** (value + momentum, not carry). Interestingly, it also lands on a
  statistically-zero combo on this free tape — two independent "everywhere" claims,
  two independent statistical zeros, no shared code path between the studies beyond
  the shared inference conventions.

None of the siblings test **carry, across all four asset classes, combined into one
basket** — that combination is this study's own axis.
