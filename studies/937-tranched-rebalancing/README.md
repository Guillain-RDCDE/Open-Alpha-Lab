# Study 937 — Tranches 🍰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the timing-luck cone real, and does tranching close it? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | On the real SPY/IEF tape the **identical** monthly rule prints excess Sharpes from **+0.499 to +0.755** and 23-year terminal wealth **95.1% apart**, decided by nothing but which day of the month you rebalance. This is a *measurement, not a premium* — a dispersion has no null to reject, and we refuse the cheap "CI excludes zero" (a standard deviation never straddles zero). It earns Real by size and by reproducing on every cut: both eras (sd 0.096 / 0.056), a 12-1 momentum sleeve (0.039), the tradable BIL cash leg (0.051), a second day of execution delay (0.071), and an edge-free random rule that is **wider in 19 of 20 seeds** (mean 0.093 vs 0.066) — the signature of an artefact. Tranching closes it **exactly** at 21 tranches, two-thirds at four, gain **+0.044** (the one interval that could have gone negative: [+0.014, +0.073], \|*t*\| ≈ 2.7, era-robust). Not claimed: any single date being significantly better — best-minus-worst HAC *t* is **+1.98**, and selection-biased. Survivorship: SPY/IEF are hindsight-picked survivors, but the cone is measured *across dates on the same assets*, so survivorship cannot manufacture it. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-e67e22?style=flat-square) | Thin and account-dependent. The gain is **variance, not return**: **+0.07 pp/yr** of CAGR against −0.98 pp of vol, and the tranched book *is* the average of the 21 dates, so it cannot be anything else. Proportional cost is free (turnover **2.741x → 2.700x NAV/yr, −1.5%**, both legs counted, flat from 0 to 25 bps) but **broker tickets are an ASSUMPTION**, not tape: 2.6 → **57.6 tickets/yr**, so at 0.5 bp of NAV per ticket the full book pays **0.29%/yr** — four times the CAGR it adds — while inside a large fund it pays nothing. Tax is not modelled and runs the same way. The honest operating point is **four tranches**. |
| **Is the lucky rebalance date forecastable?** | ![Barely](https://img.shields.io/badge/Barely-8b949e?style=flat-square) | Half-to-half rank correlation **ρ = +0.309**; the first-half winner ranked 2/21 later, worth **+0.058** over the 21-date average — which, compared like-for-like on those same rows, actually *beat* tranching's **+0.042** there. One ex-post-selected draw, with a different date winning the second half: we report it against ourselves rather than the convenient way round. The two rules' offset rankings correlate **+0.556**, so some of the cone is probably a shared month-end footprint (study 604) rather than pure noise — chasing it just re-enters the lottery the fix removes for nothing. |

> **In one sentence:** the day of the month on which you happen to run a monthly rule is worth up to **0.26 of a Sharpe point and 95% of terminal wealth** over two decades — implementation luck, wider still for a rule with no edge — and splitting the book into overlapping tranches deletes that lottery **exactly**, for **−1.5% turnover** and **+0.07 pp/yr**, which makes tranching a governance fix worth doing and not a return you can spend.

## What we tested

A deliberately ordinary sleeve — hold **SPY** for the month if it closed above its
**200-day** average on the rebalance day, else **IEF** — run once for **every one of the 21
rebalance offsets** in the 21-day cycle, then as **N = 4, 12, 21 overlapping tranches** at
every rotation. State known at the close of *t* earns the return of *t+1* (one `shift`; a
second day of delay is re-run and changes nothing); cost = one-way bps x traded notional (a
switch trades both legs); no shorts, so no borrow; Sharpes excess-of-cash on both sides
against an **^IRX PROXY** accrual leg (BIL cross-checks it on 2007+).
SPY∩IEF 2002-07-30 → 2026-06-30, books live 2003-06-16 → 2026-06-30. Block-bootstrap bands,
an era cut, a cost sweep, a ticket-fee assumption sweep, a persistence test, a 20-seed
edge-free control and a planted/null synthetic one. **Dedup:** [836-timing-luck](../836-timing-luck/) measured
the same phantom dispersion on a **synthetic** panel with zero planted edge; 937 is the
sequel it asks for — the **real tape**, with the **fix** as the object of study.
[936-rebalance-bands](../936-rebalance-bands/) varies *when* to trade a fixed-weight book;
we vary only the schedule's **phase**. [102-free-rebalance](../102-free-rebalance/) is the
*economic* rebalancing premium; [604-month-end-rebalancing-flows](../604-month-end-rebalancing-flows/)
is a real month-end *flow* (and the honest suspect behind our +0.556 cross-rule
correlation); [110-faber-timing](../110-faber-timing/) evaluates the 200-day rule itself,
which here is only a carrier. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the lottery hiding in "rebalanced monthly", the 95% wealth gap, why four sleeves fix it, what it costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the cone by N against 1/√N, joint block-bootstrap CIs, the selection-biased best-minus-worst *t*, persistence, era cut, cost and ticket sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`tranching/`](tranching/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
