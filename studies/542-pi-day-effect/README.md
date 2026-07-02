# Study 542 — Pi-Day-Effect 🥧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the market behave differently on Pi Day / constant dates? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On SPY (1993-2026, 24 Pi Days) the Pi-Day mean is **+2.55 bps** vs **+4.08 bps** on other days — contrast **−1.53 bps**, Welch *t* **−0.08** (*p* 0.94), HAC *t* +0.12. The pooled six-constant set: contrast **+4.45 bps**, *t* **+0.47**, and a **random-date-set placebo *p* = 0.66**. Bonferroni across the six constants → every corrected *p* = **1.00**. Sign flips across windows. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Each Pi Day is an isolated round trip: a "hold only on Pi Day" rule harvests +2.55 bps gross and pays ~10 bps of frictions → **net −7.45 bps/event**. ~1 event/year. Nothing to trade. |
| **Numerology vs data** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The market does not know its digits of π. On 98 years of ^GSPC, Pi Day is again *negative* (*t* −0.77); the one flicker (golden ratio, raw *p* 0.030) is a **best-of-six mirage** that collapses to Bonferroni *p* 0.18. |

> **In one sentence:** returns on Pi Day (3/14) and the other "mathematical constant" dates (*e*, τ, φ, √2, Feigenbaum) are indistinguishable from — and if anything slightly *below* — ordinary trading days: 66% of *random* six-date sets beat the constant set, Bonferroni erases everything, and the lone significant sub-window rests on four Pi Days and dies out of sample.

## What we tested

The tongue-in-cheek numerology claim: dates that encode a famous constant — **Pi Day** (3/14),
**Euler's day** (2/7), **Tau day** (6/28), **golden-ratio day** (1/6), **√2 day** (1/4),
**Feigenbaum day** (4/6) — see the market behave differently. We take **daily SPY log-returns
(1993-02-01 → 2026-06-12, fingerprint `3e185e607be5`)** with **^GSPC (1928-2026)** as a long-tape
robustness check, and run: a **Welch two-sample *t*** on Pi-Day (and pooled constant-day) returns
vs all other days, a HAC *t* on the event mean, a **random-date-set placebo** (how often does a
random set of six calendar slots beat the constant set — the multiple-testing-aware null), a
**per-constant Bonferroni sweep**, a naive Pi-Day timing rule with costs, a four-window robustness
sweep, and a deterministic **seed-robust synthetic positive control** that plants a constant-day
bump and proves the engine catches it while staying flat at the null. *Distinct from the weekday
superstition [163 Friday-13th](../../163-friday-13th/) and the single-holiday date studies
[285 St-Patrick's-Day](../../285-st-patricks-day/) / [286 Valentine's-Day](../../286-valentines-day/):
this pools six **numerology** dates and prices in the date choice with a random-date-set null.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "the market knows π" would even mean, the Pi-Day bar chart, the random-date placebo reveal, why picking six dates dooms you, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Welch + HAC *t*, the random-date-set null, the Bonferroni sweep, the four-window sign-flip, costs, the 98-year GSPC check, and the seed-robust synthetic positive control |

The fingerprinted real-data run is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in [`pi_day_effect/data.py`](pi_day_effect/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`pi_day_effect/`](pi_day_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
