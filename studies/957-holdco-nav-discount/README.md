# Study 957 — Holdco Discount 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The gap does **not** mean-revert: a discount one standard deviation wider than its own two-year norm predicts nothing six months out (pooled Driscoll-Kraay *t* = **−0.09**, R² = 0.0000, sign negative in **5/7** names — a coin flip). A timed hedged rule does post a gross Sharpe of **+0.566** (*t* = **+3.04**), but that falls to *t* = **+1.87** once the three thin OTC ADR pairs, whose stale closes manufacture exactly this pattern, are removed; **none of the 17 cells** in the enter/exit threshold grid reaches a net \|*t*\| = 2; and the headline 504-day standardisation window is the **best of five** swept. No single name clears \|*t*\| = 2. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of 10 bps a leg and 1% borrow the Sharpe is +0.312 (*t* = +1.69) with a bootstrap CI of **[−0.05, +0.66]**; flat at 25 bps, −0.47 at 50; it needs a short leg in the very ADRs producing the signal, and it flips sign under the other defensible hedge. The patient version — own the discount and wait — returned **−1.68%/yr** for 22 years. |

> **In one sentence:** Seven holdcos whose NAV is markable straight off the tape say the same thing — the discount is a **price** (of control, of tax leakage, of holding-company overhead), not an **error**: it does not close on any timetable you can trade, and what looks like a hedged edge is friction-sized reversal concentrated in the panel's stalest quotes.

## What we tested

For seven listed holdcos whose value is dominated by one **listed** stake — Heineken Holding,
Christian Dior, Liberty Broadband, Naspers, Prosus, Bollore, SoftBank — we mark NAV from the
tape (`k × stake price + other`, **price-only** closes) and test (a) whether the discount
mean-reverts, via a Driscoll-Kraay pooled regression of the forward 126-day change on the
trailing z, and (b) whether a **dollar-neutral** buy-the-wide-discount pair pays after 10 bps
a leg and 100 bps borrow, raced against an always-on control. `k` and the "everything else"
term are **PROXIES** (three published share counts, four anchored to a reported discount) and
are swept 0.8–1.2× / 0.5–1.5×; costs, borrow, eras, leave-one-out, hedge construction, the
entry/exit thresholds and the standardisation window are swept too. Three names are cut at
**announced corporate events** — Bollore at the Vivendi split, SoftBank at the Alibaba
disposal, Liberty Broadband at the 2024 Charter merger agreement, after which its gap is a
merger spread and not a discount. Survivorship runs *in favour* of the thesis — the
take-privates of Liberty TripAdvisor and Cannae left no tape, and a discount ending in a
buyout is one that closed.
**Dedup:** distinct from **367-closed-end-fund-discount** and **910-managed-distribution-cef**
(a manager *publishes* the NAV), **378-etf-nav-premium** (a creation-redemption mechanism
*forces* convergence), **620-a-h-premium** (same company, two venues), **366-merger-arbitrage**
(a contractual convergence date) and **239-spinoffs** (after the split, not before).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why EUR 1 of LVMH costs 79 cents, why it stays that way, the stale-quote trap, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the NAV assumptions, Driscoll-Kraay pooled regression, dollar-neutral pair race, bootstrap CIs, cost/borrow/calibration sweeps, the stale-quote decomposition, live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`holdco_nav/`](holdco_nav/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
