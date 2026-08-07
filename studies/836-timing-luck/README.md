# Study 836 — Rebalance Timing Luck 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the offset-to-offset Sharpe dispersion real edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a tape built so momentum has **zero** genuine edge, the **same** book rebalanced on 21 different days of the month prints Sharpes from **−0.24 to +0.17** — a **0.41-Sharpe phantom gap** on the *identical* strategy. It is **luck, not skill**: the offset ranking's first-half/second-half rank correlation is **+0.04** (unforecastable), and even the dispersion-free tranched book finds nothing (NW *t* = **−0.06**). A synthetic-only method demo — no real tape, so it can never earn `REAL`. |
| **Tradability** — can you harvest the dispersion? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | You cannot trade a spread whose winner is a coin-flip: across 25 seeded worlds the best offset is scattered uniformly over 0…20. The tranched null book is flat gross (−0.08 bps/day) and **loses net** of costs (−0.41 bps/day at 1 bp). Nothing to spend. |
| **Does rebalance timing break inference?** | ![Confirmed](https://img.shields.io/badge/Does_rebalance_timing_break_inference%3F-Confirmed-8b949e?style=flat-square) | **Yes.** An arbitrary rebalance-date choice swings the reported Sharpe by **~0.44 units** (25-seed mean) on the identical rule — enough to hire or fire it — and the dispersion is present *whether or not* real edge exists. **Tranching / overlapping portfolios** collapses it to a single curve (dispersion → 0), while a *planted* premium still lights the tranched book up (Sharpe **1.39**, NW *t* **4.24**, fires **24/25**). |

> **In one sentence:** the celebrated "monthly momentum" backtest hides a silent, arbitrary choice —
> *which day* you rebalance — and that choice alone swings the reported Sharpe by ~0.4 units of pure
> luck (the lucky offset is unforecastable), so a smooth-looking track record can be an accident of
> the calendar; the free fix is to tranch the rebalance across every offset, which collapses the
> phantom dispersion while preserving any genuine edge.

## What we tested

Hoffstein, Faber & Braun (Newfound Research, 2019) and Hoffstein–Sober–Vezeris (2020), **"Rebalance
Timing Luck: The (Dumb) Luck of Smart Beta"**: the same rules-based strategy rebalanced on a
different day of the period traces a materially different equity curve and Sharpe — a phantom
dispersion that is pure luck, not skill, and the fix is portfolio **tranching / overlapping
portfolios**. We build **one** monthly cross-sectional momentum long-short (trailing-126-day signal,
long top 30% / short bottom 30%, dollar-neutral, rebalanced every 21 days, signal known at the close
of `d−1`) and run it once for **every rebalance offset 0…20** on a deterministic synthetic panel
whose null (`mom_edge = 0`) has zero momentum edge by construction. We quantify the Sharpe
dispersion, test whether the lucky offset **persists** out-of-sample (it doesn't), collapse it with
the tranched/overlapping portfolio, cost the book, and prove on a seeded positive control
(`mom_edge > 0`) that the machinery still detects a genuinely planted premium. Synthetic-only, so
capped at `NONE` on the **Signal** axis. **Dedup:** [349-regime-dependence](../349-regime-dependence/)
varies the *sample/regime*, not the rebalance day; [102-free-rebalance](../102-free-rebalance/) is
the *economic* rebalancing premium, not this *statistical* date artefact; [604-month-end-rebalancing-flows](../604-month-end-rebalancing-flows/)
is a real predictable *flow* around reconstitution, whereas timing luck has nothing underneath — the
lucky offset is a coin-flip. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the *same* strategy on a different rebalance day can look like a winner or a loser, why that gap is luck, and how tranching makes it vanish — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the per-offset Sharpe fan, the out-of-sample offset-persistence rank correlation, the tranched Newey-West *t*, the cost math, and the 25-seed synthetic positive control |

The fingerprinted headline run (null panel fp `abbc37b5f962`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the whole machinery runs offline and deterministic on the
synthetic world in [`timing_luck/data.py`](timing_luck/data.py). Sources & literature map:
[docs/references.md](docs/references.md).

---

*Engine: [`timing_luck/`](timing_luck/). A synthetic-only research-method demo — no real tape, so
never `REAL` (which needs a robust *t* ≥ 2 on real data). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
