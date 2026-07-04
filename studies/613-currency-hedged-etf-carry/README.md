# Study 613 — Currency-Hedged-ETF-Carry 💴

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the hedge really pocket the rate differential? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On the **same-basket** pair (HEWJ = EWJ + JPY forwards) the wrapper is a full currency short (**β = 0.97**, R² = 0.87) and the residual carry is **+2.39 %/yr at HAC *t* = +4.54** full-sample, **+4.83 %/yr at *t* = +6.01** in the 2022+ Fed-vs-BOJ era — tracking the observable policy differential at **corr 0.85, pass-through slope 0.98**. DXJ/EWJ corroborates (*t* = +2.24/+4.75, β = 1.05 from its Apr-2010 hedged-mandate start). Honest split: the **EUR pair** passes the differential through (β = 0.98, slope 1.07) but its thin ~1.1 %/yr differential can't clear *t* ≥ 2 through basket-mismatch noise. |
| **Tradability** — can you actually pocket it? | ![Investable](https://img.shields.io/badge/Tradability-Investable-2ea44f?style=flat-square) | The Japan wrapper is **free** (HEWJ ER 0.50 % = EWJ 0.50 %), one click, multi-billion capacity; the fx-stripped isolation trade nets **+4.13 %/yr (*t* = +5.14)** after 50 bps borrow + costs in 2022+; the observable-differential switch rule needed **3–5 trades in a decade+**. Caveats: you pocket whatever the (moving) differential is — BOJ hikes shrank it ~5.3 → ~3.3 %/yr by mid-2026 — and Europe's wrapper ER gap (0.58 vs 0.09 %) eats half its differential. |
| **"Unhedged was better when carry was ~zero"?** | ![Busted](https://img.shields.io/badge/Unhedged_better_at_zero_carry%3F-Busted-8b949e?style=flat-square) | 2010–2015 (differential +0.01 %/yr): the carry component was **−0.66 %/yr (t = −0.49)** — nothing to pocket, exactly as the mechanics demand — yet the hedged class *still* won **+28 bps/mo**, entirely on the falling yen. At a thin differential the share-class choice is a **pure FX bet**, not a carry decision. |

> **In one sentence:** the "free carry hidden in a share class" story is *mechanically true and
> measurable* — HEWJ minus EWJ is a β ≈ 1 short of the yen plus a carry residual that tracked the
> US–Japan policy differential one-for-one (corr 0.85) and paid **+4.83 %/yr at HAC *t* = +6.01**
> while the Fed sat ~5 % above the BOJ — pocketable at zero extra expense ratio; just know that
> the tap is the *observable* differential (it closes as rates converge) and that at zero
> differential the hedged-vs-unhedged choice is nothing but a currency bet.

## What we tested

A currency-hedged equity ETF sells the foreign currency forward; covered interest parity prices
that forward at the short-rate differential, so `hedged − unhedged ≈ (r_US − r_foreign) − fx`.
We rearrange to **carry_hat = (hedged − unhedged) + fx** and measure it on three live pairs —
**DXJ/EWJ** (Japan 2010+; DXJ's pre-Apr-2010 *unhedged*-mandate era is excluded — its hedge β
was −0.15 there), **HEWJ/EWJ** (Japan 2014+, the decisive *same-basket* pair) and
**HEDJ/VGK** (Europe 2012+, sliced at its hedged-mandate start too) — on yfinance total-return
closes, with the US rate from `^IRX` and
BOJ/ECB policy rates hardcoded as step tables (sources cited). Inference is **Newey-West HAC**
(6 lags, 3/12 sensitivity) on the carry mean and on the hedge regression `diff = α + β·(−fx)`;
the high-differential era (2022-07→2026-06) is the stress test, the 2010–2015 zero-differential
era is the myth-check, and tradability charges borrow + one-way costs on the isolation spread
plus a one-month-lag share-class switch rule. A deterministic synthetic world with a **planted,
tunable carry** proves the estimator recovers the knob and that a zero-carry null cannot fire.
Distinct from [364-fx-carry-trade](../364-fx-carry-trade/) (direct currency carry as a risk
premium): here no currency position is chosen — we test the **ETF wrapper's mechanical
transfer**. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a hedged share class actually does, why selling yen forward pays you the rate gap, where the "free" money comes from (and when the tap closes) — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the carry decomposition, HAC t's on the same-basket pair, β ≈ 1 hedge regressions, rolling pass-through vs the policy differential, era splits, borrow-and-cost math, and the planted-carry synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`currency_hedged_etf_carry/`](currency_hedged_etf_carry/). The carry estimate is
`(hedged − unhedged) + fx` per month; foreign short rates are coarse hardcoded policy-rate step
tables (BOJ, ECB). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
