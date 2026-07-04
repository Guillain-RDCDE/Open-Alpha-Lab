# Study 606 — OPEC Announcement Effect 🛢️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do OPEC decision days move oil? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the vol · None on the drift.* Decision-day \|return\| runs **1.40–1.53×** the event-free baseline on all three tapes (Welch *t* = **+3.01 to +3.17**, intraday-range *t* up to **+4.80**, placebo *p* ≤ 0.018) — but the **signed** drift day 0..+5 is zero everywhere: all 15 horizon×tape HAC *t*'s sit **below \|1.2\|**. |
| **Tradability** — can you trade the announcement? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No drift to harvest at any cost (CL net +345 bps/event at 10 bps one-way — but *t* = 1.40, never significant). The day-0-sign continuation's one borderline cell (USO *t* = 2.20) **collapses to *t* = 1.19 without the five 2020 meetings**. The vol bump itself is exactly where option vol is already marked up. |
| **"Vol doubles on OPEC day"?** | ![Busted](https://img.shields.io/badge/Vol_doubles%3F-Busted-8b949e?style=flat-square) | Louder, yes — **~1.4–1.5×**, *t* ≥ 3. Doubled, no: the bootstrap 95% CI **excludes 2.0 on every tape** (uppers 1.67/1.89/1.85), and the planted-×2 synthetic control shows this harness would have read a true doubling at *t* ≈ 6. |

> **In one sentence:** across **107 hardcoded OPEC/OPEC+ ministerial decision days
> 2000-2026**, oil is genuinely ~**1.5× louder** on decision day (Welch *t* ≥ 3 on WTI,
> Brent and USO) — but the folklore's other half is empty: vol does **not** double, the
> post-decision drift is statistically zero at every horizon, and the "trade the
> announcement" rule is a five-meetings-of-2020 artifact — **Mixed, and a Mirage to trade**.

## What we tested

We froze the full calendar of OPEC Conference (2000-2016) and OPEC+ ONOMM (2016-2026)
decision days — **107 meetings**, hardcoded with per-date sources from the OPEC
press-release archive, JMMC/subgroup calls excluded by a pre-registered scope rule — and
mapped each to its first tradable session on three tapes (`CL=F`, `BZ=F`, `USO`,
yfinance OHLC, as-of 2026-06-30). **Vol axis:** day-0 |return| and intraday range vs an
event-free (±5-session halo) baseline — Welch *t*, Brown-Forsythe spread test, variance
ratio, a block-bootstrap CI on the vol multiple and a 2,000-draw random-calendar placebo.
**Drift axis:** cumulative close(-1)→close(+k) drift, k = 0..5, per-event *t* plus a
Newey-West dummy regression on the daily tape. **Tradability:** the day-0-sign
continuation rule entered at the day-0 settle (lagged variant shown), at 2/5/10 bps
one-way futures costs, with ex-2020 and sub-period robustness. A deterministic synthetic
world with a *planted* vol multiple and drift proves the machinery fires when it should
and stays quiet on the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what OPEC decision days actually do to oil — louder, yes; directional, no — and why "trade the announcement" quietly means "you needed March 2020 in the sample", in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the vol multiple with bootstrap CI + placebo, Brown-Forsythe and Welch tests, HAC drift regressions across 15 horizon×tape cells, the continuation rule's 2020 decomposition, costs, and the planted-effect machinery control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`opec_announcement_effect/`](opec_announcement_effect/). Event input = the frozen
107-meeting table in [`data.py`](opec_announcement_effect/data.py) (source-commented);
siblings: [313-geopolitical-shock](../313-geopolitical-shock/) covers unscheduled
wars/crises, [226-crude-seasonality](../226-crude-seasonality/) covers calendar-time
patterns — this study is the scheduled-decision corner. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
