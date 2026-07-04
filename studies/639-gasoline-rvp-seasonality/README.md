# Study 639 — Gasoline-RVP-Seasonality ⛽

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the RVP-calendar seasonal on the gasoline-crude spread real? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Feb–Apr, gasoline out-runs crude by **+11.30%/window** (Welch *t* across 21 years = **+6.20**, positive 90.5% of years); the only two months clearing the Bonferroni-×12 bar are **exactly the two the statute names** — March (*t* = +5.34, clears it two-sided) and September (*t* = −3.12, clears the one-sided directional bar; the statute pre-registers the sign). Both halves clear *t* ≥ 2 alone. Measured on the spliced front-month chain, **labelled a spot-price proxy** (splice caveat named). |
| **Tradability** — can a holder capture it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The curve already carries the law: a real front-month holder (UGA) pays the seasonal back at the rolls — **−10.4%/window**, negative **18/18 years**. The investable crack (UGA−CL) earns **−0.18%/window at *t* = −0.07**; long-only UGA in the window is sub-2 (*t* = 1.4) unhedged energy beta. Nothing left for costs to eat. |
| **"Does the futures curve already price it?"** | ![Confirmed](https://img.shields.io/badge/Curve_prices_it%3F-Confirmed-8b949e?style=flat-square) | Paired holder-vs-splice gap: **−10.40%/window** in the run-up (*t* = **−8.06**, 18/18 years) and the mirror-image **+6.38%** giveback in September (*t* = +4.95). The seasonal lives in the curve, not in anyone's P&L. |

> **In one sentence:** the May-1 summer-blend deadline (40 CFR 1090.215) really does put an
> enormous, dated seasonal on the gasoline-minus-crude spread — **+11.3% every Feb–Apr window,
> Welch *t* = 6.2, with March and September the only Bonferroni survivors, exactly as the law is
> written** — but the futures curve has read the Federal Register too: a real holder pays the
> entire seasonal back at the rolls (−10.4%/window, negative 18 years out of 18), leaving the
> investable version at *t* = −0.07 — **Real, and a Mirage**.

## What we tested

The claim: US gasoline must switch to expensive low-RVP summer blend by **May 1** (refineries/
terminals; retail June 1) and may switch back after **September 15** — a seasonal with an actual
law behind it. We test the **calendar**, not the crack level ([306-crack-spread](../306-crack-spread/)
tests the level; [226-crude-seasonality](../226-crude-seasonality/) tests WTI outright): monthly
log-excess of RBOB (`RB=F`) over WTI (`CL=F`), 2005→2026, per-year panel, Welch *t* across ~21
years on the Feb–Apr run-up, one-sample *t* on the September switch-back, per-month table with a
Bonferroni-×12 bar, window variants and a halves split. The third axis pits **UGA** (a real
front-month holder who pays every roll) against the spliced chain, paired per year, to ask
whether the curve pre-prices the statute; tradability charges the long-only UGA overlay 5/10/20
bps one-way × NAV plus the window T-bill (excess-vs-cash) and measures the investable UGA−CL
crack. Entries use the prior month-end close — the calendar is a statute known years in advance,
so the one-lag rule is satisfied by construction. A 20-seed synthetic world with a planted
seasonal + planted roll drag proves the machinery (never cited as evidence). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why gasoline gets pricier every spring *by law*, why the chart shows it perfectly — and why you still can't buy it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the per-year Welch panel, the Bonferroni month table, window/halves robustness, the paired holder-vs-splice roll-gap test, costs, and the planted-seasonal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md) (fingerprint `ef905dc22f3c`).

---

*Engine: [`gasoline_rvp_seasonality/`](gasoline_rvp_seasonality/). The signal is the Feb–Apr /
Sep RVP statute calendar on the RB−CL monthly excess; the myth-check is the paired UGA-vs-splice
roll gap. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
