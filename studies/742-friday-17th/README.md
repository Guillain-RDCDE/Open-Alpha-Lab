# Study 742 — Friday-17th 😱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the FTSE MIB trade weak on Venerdì 17? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Venerdì-17 mean **+26.3 bps** on the FTSE MIB (one-sample *t* = **+1.27**) — *positive*, so the fear has the sign backwards; down-day rate **46.9%** (below a coin flip); Welch vs other Fridays *p* = **0.17**; random-calendar placebo left-tail *p* = **0.93**. EWI concurs (+10.1 bps, *t* = +0.52). |
| **Tradability** — could you bet the curse? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | *Shorting* the unlucky day (the folklore trade) **loses**: net **−38 bps/event** on the MIB (*t* = −1.85), −22 bps on EWI. No edge to charge costs against. |
| **Unlucky day?** | ![Busted](https://img.shields.io/badge/Unlucky_day%3F-Busted-8b949e?style=flat-square) | The 17th is a slightly *above-average* Friday; the most extreme neighbouring slot is the (superstition-free) **10th**, and even it dies under Bonferroni. Same shape as [Friday-13th](../../163-friday-13th/), one country over. |

> **In one sentence:** across **49** Friday-the-17ths since 1998, Italy's FTSE MIB is, if anything, faintly *green* (+26.3 bps, *t* = +1.27, placebo *p* = 0.93), a red day less than half the time, and the one bettable version of the curse — shorting the 17th — loses money net of costs; *Venerdì 17* leaves no mark on the Italian tape, exactly like Friday the 13th on Wall Street.

## What we tested

Italians fear **Friday the 17th** (*Venerdì 17*), not the 13th — the Roman numeral `XVII`
is an anagram of the Latin `VIXI`, "I have lived", i.e. "I am dead". The folk-finance claim
is the Latin-market cousin of the Anglo Friday-13th trade: a whole country in a darker mood
should leave the **FTSE MIB** trading weak that day. We derive every Venerdì 17 by pure date
arithmetic (Friday **and** the 17th — known before the open, so zero look-ahead), run a
one-sample *t* across these independent, non-overlapping events on the FTSE MIB itself
(local, EUR, price-only), cross-check on the tradable EWI ETF (USD, total-return), and put
it through a Welch contrast vs other Fridays, a Wilson down-day rate, a multi-seed
random-calendar placebo, a day-of-month Bonferroni sweep, and a costed short — with a
deterministic synthetic tape (a *planted* Venerdì-17 fear) as the positive control. **As-of
2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an Italian superstition *should* dent the Italian market if mood prices at all, what the tape actually shows (a faint green lift), and why shorting the curse loses money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* / Welch battery, the Wilson down-day rate, the random-calendar placebo, the look-elsewhere Bonferroni sweep, the sub-period split, the costed short, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`friday_17th/`](friday_17th/). Venerdì-17 dates are pure calendar arithmetic;
`FTSEMIB.MI` (EUR, price-only) and `EWI` (USD, total-return) are fetched via yfinance and
labelled as such everywhere. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
