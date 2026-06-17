# Study 287 — Easter-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Is there a Good-Friday / Easter seasonal?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The session **before** Good Friday (always Maundy Thursday) earns **+0.33%/day**, **+29.6 bps** over the everyday drift, **up 68%** of the time. HAC *t* = **+3.28**, perm p = **0.012**, and it **survives the day-of-week control** (HAC *t* = +3.29 vs Thursdays only). Present in both halves of the sample (*t* = 4.29 / 2.14). This is a real **pre-holiday effect**, not Easter magic. *Signal-axis caveat:* price-only index, vendor back-revises the level; the 1980–2002 third is weak (*t* = 1.19) and the pre-holiday effect is known to decay. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The edge survives costs *per event* (net +31 bps after a 2 bps round-trip), but it fires **once a year** — ~0.31% annual contribution, dominated by simply holding the index 252 days. Capacity is huge (it's the S&P) but the sliver can't be safely levered on one overnight gap, and the documented effect has decayed elsewhere. |
| **Busted?** | ![No](https://img.shields.io/badge/No-2ea44f?style=flat-square) | Not folklore: there **is** a robust seasonal at the Good-Friday gate. But it is the **generic pre-long-weekend drift**, present because Good Friday closes the NYSE — there is no Easter-*specific* magic, and Easter Monday adds nothing once the Monday effect is removed. |

> **In one sentence:** Good Friday is a market holiday, so the real action is the session *before* it — and that Maundy-Thursday session shows a genuine, control-surviving **pre-holiday drift** (+30 bps, HAC *t* > 3), but it fires once a year and is the well-known pre-holiday effect rather than anything Easter-specific.

## What we tested

Good Friday (= Easter Sunday − 2 days, by the Computus algorithm) has been an NYSE
holiday in every year 1950–2025, so there is no Good-Friday return to measure. We run a
clean **event study** on the two sessions that bracket the closed day: the **pre-holiday
session** (always Maundy Thursday) and the **post-holiday session** (always Easter
Monday). Each is tested against the **unconditional daily mean** — *and* against the
**same-weekday** baseline, since both events are weekday-locked. We report a one-sample
t-test (vs 0 and vs the baseline), a Newey-West HAC t on the excess return, a 10,000-draw
permutation test, the 1950–79 / 1980–2002 / 2003–25 split, and a gross/net tradability
column. A synthetic positive control confirms the engine fires when a pre-holiday premium
is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why the action is *before* Good Friday, the +30 bps drift, the day-of-week control, the answer in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | one-sample & HAC t, the permutation distribution, the same-weekday control, sub-periods, the n=76 power calc, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`easter_effect/`](easter_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
