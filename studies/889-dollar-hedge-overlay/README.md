# Study 889 — Broad Dollar-Hedge Overlay 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is `hedged − unhedged` the rate differential, broadly? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | [Study 613](../613-currency-hedged-etf-carry/)'s Japan carry identity **generalises to broad EAFE**. On the clean same-basket pair (**HEFA = EFA + FX forwards**) the hedge is a near-full short of the foreign currency basket (**β = 0.93**, R² = 0.91) and the residual carry is **+1.68 %/yr at HAC *t* = +4.74** — sitting right on the observable **+1.35 %/yr** US−EAFE policy differential — with a bootstrap CI clear of zero (**[+0.90, +2.45]**) and *t* ≥ 2 in **both** eras (+2.89 pre-2022 / +4.88 in 2022+), *growing* with the gap. The longer DBEF/EFA pair corroborates (*t* = +2.56). *One-regime caveat: HEFA's tape starts 2014-03 and sits in a single US-out-yields / rising-dollar era — the carry identity is the regime-independent part, the raw Sharpe win is not.* |
| **Tradability** — can a dollar-regime hedge overlay bank it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The carry is real and cheap to *hold* (the hedged wrapper's expense gap over the unhedged twin is ~2 bp), but it is **not bankable as an overlay**: the "hedge when the US out-yields" **switch adds nothing** over just always-hedging — the US out-yielded EAFE ~**93 % of months** (1 switch in 12 yr), so there is no timing edge (overlay Sharpe 0.67 vs always-hedged **0.75**). The excess-Sharpe advantage (0.75 vs 0.40) has **overlapping CIs** and rests on one dollar regime; isolating the pure carry is a dollar-long spread that nets +2.53 %/yr at only *t* = +1.41 (fx vol swamps it). Real but thin & un-timeable. |

> **In one sentence:** the "free carry hidden in a hedged share class" that [613](../613-currency-hedged-etf-carry/)
> found for Japan is **mechanically true for broad developed international too** — HEFA minus EFA is
> a β ≈ 1 short of the foreign-currency basket plus a carry that tracks the (now positive, +1.35 %/yr)
> US−EAFE policy differential one-for-one and pays **+1.68 %/yr at HAC *t* = +4.74** — but the
> "hedge when the US out-yields" *switch* adds nothing over simply hedging (the US has out-yielded
> EAFE ~93 % of the time), so it is a real premium you *hold*, not an overlay you *time*.

## What we tested

A currency-hedged EAFE ETF sells the foreign-currency basket forward; covered interest parity
prices that forward at the short-rate differential, so `hedged − unhedged ≈ (r_US − r_foreign) − fx`.
We rearrange to **`carry_hat = (hedged − unhedged) + fx_foreign`** and measure it on two live pairs —
**HEFA/EFA** (EAFE 2014+, the *same-basket* clean pair: HEFA holds EFA + one-month forwards) and
**DBEF/EFA** (EAFE 2011+, same index / different provider) — on yfinance monthly total-return closes,
with the US rate from `^IRX`, an EAFE-weighted foreign policy-rate blend (ECB/BOJ/BoE/SNB), and the
`fx_foreign` **spot** basket from EUR/JPY/GBP/CHF (UUP is *not* used for the carry — its total return
embeds the collateral yield and cancels it, a documented trap). Inference is **Newey-West HAC** on
the carry mean and the hedge regression `diff = α + β·(−fx)`, a **block-bootstrap** Sharpe/mean CI, a
**2022 era split**, a costed **"hedge when the US out-yields" overlay** (one-month lag) and a
long-hedged/short-unhedged isolation spread, plus a 20-seed planted-carry **synthetic control**.
**Dedup:** [613-currency-hedged-etf-carry](../613-currency-hedged-etf-carry/) is the *single Japan
pair* identity we generalise here; [114-dollar-smile](../114-dollar-smile/) is the dollar's *macro*
behaviour, not a wrapper carry; [828-fx-dollar-factor](../828-fx-dollar-factor/) is a *cross-currency*
dollar risk factor, not an equity-wrapper hedge P&L; [145-home-bias](../145-home-bias/) is the
*allocation* question, not the currency-hedge decision. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a hedged EAFE fund does, why the hedge now *pays* a US holder the rate gap, why the switch adds nothing when the US always out-yields — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the carry decomposition, HAC *t*'s on the same-basket pair, the β ≈ 1 hedge regression, the era split, the excess-of-cash Sharpe race, the costed overlay, the UUP collateral-yield trap, and the planted-carry synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dollar_hedge/`](dollar_hedge/). The carry estimate is `(hedged − unhedged) + fx_foreign`
per month; foreign short rates are coarse EAFE-weighted policy-rate step tables (ECB/BOJ/BoE/SNB).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
