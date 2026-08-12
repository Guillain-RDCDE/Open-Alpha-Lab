# Study 909 — Preferred Reset Premium 🔧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do variable-rate preferreds out-carry fixed on a rate-adjusted basis in the high-rate regime? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | **Real but entirely regime-contingent.** Over the full 2014–2026 tape the variable-minus-fixed (VRP − PFF) spread is a thin **+1.28 %/yr** with Newey-West *t* = **+1.44** and a bootstrap CI **across zero** — not robust. Split at the 2022 hiking cycle it *flips on*: **−0.27 %/yr** (*t* = −0.28) when rates were floored, **+3.90 %/yr** (*t* = **+2.41**) in the high-rate regime, where the variable sleeve lost **8 points less** than fixed in 2022 (−11.4% vs −19.5%). A genuine reset mechanism in the high-rate regime, but it does **not** hold across sub-eras → Mixed, not Real. *Short history: VRP 2014, PFFV 2020 — one hiking cycle of evidence, named on the Signal axis.* |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The market-neutral long-variable / short-fixed isolation nets only **+0.72 %/yr** (Sharpe +0.16, *t* < 1) after costs — too thin to bank as a spread — and *timing* the regime (switch excess Sharpe +0.26) **underperforms** simply holding variable (+0.40). The honest, bankable form is a **structural tilt to variable-rate preferreds**, whose positive carry survives its milder −16.7% drawdowns net of buy-and-hold costs; but the edge over fixed is thin, regime-dependent and rests on one cycle. Real-but-thin → **Fragile**. |

> **In one sentence:** variable-rate preferreds really did out-carry fixed ones when rates
> rose — but the premium lives **entirely in the 2022+ high-rate regime** (vanishing when
> rates were floored), is thin and insignificant over the full sample, and can't be timed —
> so the honest read is a **regime-contingent signal you tilt toward, not a bankable trade**.

## What we tested

The reset story: fixed-rate preferreds are long-duration bonds wearing an equity coupon and
got crushed in 2022; **variable / fixed-to-floating** preferreds reset their coupon off short
rates, so they held up and should out-carry fixed on a rate-adjusted basis in a high-rate
regime. We race a **variable sleeve (VRP, PFFV)** against a **fixed sleeve (PFF, PGX, PGF)**,
all **excess-of-cash** (minus BIL T-bills), on yfinance total-return closes (2014-06 → 2026-06
for the flagship VRP-vs-PFF pair), measuring the **(variable − fixed)** monthly spread (which
nets out the shared credit beta) with a Newey-West *t*, a block-bootstrap CI on the spread and
the Sharpe advantage, a **2022 era cut**, the equity-like drawdowns, a costed long-short
isolation spread, and a rising-rate regime-switch tilt (one execution lag) — plus a 20-seed
synthetic control. Short history (VRP 2014, PFFV 2020; one hiking cycle) is named on the
**Signal** axis. **Dedup:** [338-preferred-stocks](../338-preferred-stocks/) tests the
preferred asset class **as a whole** (the *level*), not the variable-vs-fixed spread *inside*
it; [339-convertible-bonds](../339-convertible-bonds/) is a different hybrid (embedded equity
option, no coupon reset); [340-bank-loans](../340-bank-loans/) is the floating-rate **loan**
analogue (BKLN), not preferreds; [885-ultra-short-credit-pickup](../885-ultra-short-credit-pickup/)
is a front-end spread pickup with no reset/duration contrast. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a variable coupon holds up when rates rise, the 2022 exhibit, and why it's a regime bet not a law |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race, the spread Newey-West *t*, the bootstrap CIs, the 2022 era cut, the equity-like drawdowns, the cost math, the regime-switch, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pref_reset/`](pref_reset/). yfinance total-return closes, cached under this study's own `_cache/`;
every Sharpe is excess-of-cash (minus BIL). Short history (VRP 2014, PFFV 2020) — magnitudes rest on one
hiking cycle. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
