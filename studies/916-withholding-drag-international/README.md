# Study 916 — Withholding Drag 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — can the tape see the foreign tax withheld before a fund's dividend reaches you? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No — the quantity is **not identified on the tape**. Measurement is exact (differenced yield reproduces declared cash to **under 1 bp** on all seven funds), but the *gross* dividend is never published and every US-listed benchmark suffers the same treaty withholding. The VEA-vs-blend gap is **−47.5 bp/yr with the wrong sign** (HAC *t* = −1.07) and collapses to **−0.5 bp** (*t* = −0.01, CI ±69) once the 47 bp fee difference is added back — insignificant in both eras, sign-flipping between them, swinging 72 bp across country weights, never exceeding \|*t*\| = 0.28 across a 35–55 bp fee-gap sweep, and below a resolution floor that already fails to recover a *known* 26 bp fee gap between EFA and IEFA. The fee add-back is itself the *most generous* assumption available (2026 fees on a 19-year window; fees fell). Universe is today's surviving large funds (mild ex-post tilt). |
| **Tradability** — is there a wrapper that dodges it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to act on. Granting the inferred 41 bp/yr central drag, no US-listed wrapper avoids it: the single-country "gross" benchmark is taxed identically and costs **~102 bp/yr more** excess-of-cash (HAC *t* = −1.05, unmoved across a 0–25 bp cost sweep). The only genuine lever — the foreign tax credit in a taxable account — is a tax-return mechanic, off-tape. |

> **In one sentence:** A fund's realised income yield can be measured from the tape to within a basis point (VEA pays **302 bp/yr**), but the tax withheld *before* the fund ever reported it cannot — so the honest answer is an inference spanning **16–101 bp/yr** depending on an assumed rate, and the "gross-yield" benchmark built to pin it down turns out to measure fees and country weights instead.

## What we tested

Reconstruct each fund's realised distribution yield as **total-return close minus
price-only close** (`auto_adjust=True` vs `auto_adjust=False`, both split-adjusted),
validate that ruler against the funds' declared cash dividends, then compare the broad
developed-ex-US funds (**VEA, IEFA, EFA, VXUS**) against a monthly-rebalanced
**EWJ/EWU/EWG 50/30/20** single-country blend of the same market — one execution lag on
the rebalance, one-way cost × NAV, no short leg, excess-of-cash race vs BIL, HAC *t*,
63-day block-bootstrap CIs, an era cut, a cost sweep, a blend-weight sweep and a
fee-gap sweep, over 2007-07-30 → 2026-06-30 (as-of 2026-06-30). Expense ratios (2026
fact sheets on a 19-year window — a **named anachronism**, and one that flatters the
hypothesis), blend weights and the effective withholding rate are labelled
**ASSUMPTIONS** and each is swept on its own grid.
**Dedup:** distinct from **913-tracking-difference-persistence** (races trackers' *total*
returns, not the income leg), **914-sec-lending-offset** (the other hidden fund-level
cash flow, revenue rather than tax), **915-k1-vs-1099-structure** (a wrapper tax question
that *is* identified, because the two wrappers sit on opposite sides of the tax),
**516-dividend-month-premium** / **143-dividend-capture** (trading *around* ex-dates, not
measuring them), and **568-effective-tax-rate** / **599-tax-loss-harvesting** (tax as a
corporate characteristic and as an investor action, not as a fund-level leak).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the ruler that works, why the "gross" benchmark points the wrong way, fees vs tax, the labelled inference, what you could actually do about it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the measurement identity and its validation, the EFA/IEFA resolution floor, HAC *t* and bootstrap CIs, era cut, weight and fee sweeps, the excess-of-cash race, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`withholding/`](withholding/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
