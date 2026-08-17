# Study 956 — The Custody Fee 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — can the tape see what the depositary wrapper costs you? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | It sees *something*: the ADR hands its holder **+13.8 bp/yr** less income than the home line (median +7.5), positive on **9 of 10** names (sign test *p* = 0.011), name-bootstrap CI **[+5.0, +25.7] bp/yr**, positive in both eras. It is not robust. Pooled *t* = **+2.36** falls to **+1.92** on leave-one-out; collapse the six euro-area issuers into one observation (they share a tax regime and a dividend calendar, so they are not ten independent draws) and *t* = **+1.84**; only **3 of 10** names clear a block bootstrap; the mean is inflated by Novartis, whose +59 bp is a spin-off recording artefact (drop it and the mean halves to **+8.8**); the price placebo is contaminated on three names; **5 of 15 pairs are unusable** because the vendor's LSE adjusted close carries no dividends at all. The panel is a **survivor panel** — large issuers still listed on both venues in 2026, which excludes exactly the de-sponsored ADRs where fee disputes concentrate. |
| **Tradability** — is it worth crossing a border to escape? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No. Owning the home lines instead beats the ADR basket by **+14.6 bp/yr gross at HAC *t* = +0.11** (excess-of-cash on both legs, one-way FX conversion × NAV at *t*+1, long-only so no borrow); a **15 bp/yr** foreign safekeeping charge — under what any US brokerage bills for foreign settlement — flips it negative, and rebalancing turnover is charged on neither leg, so even the gross figure flatters the home basket. It is also not an implementable backtest: it inherits a centred (forward-peeking) bad-print filter that moves it by hundreds of bp. A real, small, unavoidable cost of the wrapper — not an arbitrage. |

> **What the tape cannot tell you:** it measures a **combined income shortfall**, not a fee. At treaty rates the withholding alone would cost **26–96 bp/yr** — several times the whole gap — so subtracting it drives every name negative. That proves the tax is *absent* from the vendor's total-return series (the per-ADS dividend recorded is the gross declared amount) — and the same argument bites the fee, which is often billed as a separate DTC line item and would then never touch this tape either. So **13.8 bp/yr is an upper bound on the depositary fee, not a measurement of it**; that its median (5.3 cents per ADS/yr) lands on the published 1–5 cent schedules is suggestive, not evidence. Meanwhile the *naive* version of this comparison reports a fee a hundred times too large, because London's adjusted close has no dividends in it.

## What we tested

Fifteen ADR / home-line pairs, 2000-01-03 → 2026-06-30, as-of 2026-06-30. Estimand: the **slope in time** of `log(ADR total return) − log(home line total return × FX)`, with a separate intercept per detected level-shift segment so an ADS-ratio change contributes nothing, plus a **price-only ratio placebo** that must stay flat. The headline is the *income* leg — `log(total-return) − log(price-only)` on each side, which is currency-free — with the treaty-withholding split swept 0 → 1.5× (it fails). Newey-West *t* at 252 lags, per-name block bootstrap, name bootstrap, sign test, leave-one-out, currency-block clustering, era cut, break-threshold sweep, and one traded leg (own the home line instead: excess-of-cash vs BIL on both legs, one execution lag, one-way cost × NAV, no short leg). The coverage screen reads the *home* leg's own realised yield only, so it cannot select on the answer.

**Dedup:** distinct from **955-adr-overnight-catchup** (the ratio's tradable *deviations*, not its drift), **916-withholding-drag-international** (the same tax leak inside a *fund* wrapper, benchmarked against a country blend rather than the security's own twin), **913-tracking-difference-persistence** (a tracker versus its index, not a receipt versus its underlying), **889-dollar-hedge-overlay** / **906-em-local-hedged** (FX as a risk to strip; here it cancels out of the estimator entirely), and **636-exchange-listing-pop** (the act of listing, not the standing cost of being listed).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a depositary actually charges you, why it never shows on a price chart, the London data trap, and how big the bill really is |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the trend estimator and its placebo, HAC vs block bootstrap, the sign test, leave-one-out, the era cut, the withholding sweep that proves non-identification, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`adr_drag/`](adr_drag/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
