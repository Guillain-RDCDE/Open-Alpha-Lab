# Study 843 — Waffle House Index 🧇

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do insurers dip and rebuilders rally after a major storm? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Over [+0..+20] sessions after 16 major US hurricanes, insurers drift **+1.15%** (*t* = +1.48) and rebuilders **−0.28%** (*t* = −0.25) — **both the wrong sign** — and the paired (reb − ins) spread is **−1.43%** (*t* = −0.88), the opposite of the claim. Random-calendar placebo *p* = **0.38–0.66**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long-rebuilders / short-insurers trade **loses at every horizon** (5/10/20 days), gross and net (−44 to −177 bps), win rate ≤ 44% — short the leg that drifts up, long the leg that goes nowhere. |
| **Does the market read the Waffle House Index?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Hurricanes are forecast days ahead, so the market prices them before landfall; the post-landfall reaction in the obvious large-caps is flat-to-backwards, and 16 events is low power. |

> **In one sentence:** FEMA reads a storm's severity off whether the always-open Waffle House closes — but the market does not read it off Allstate/Travelers/Progressive or Home Depot/Lowe's: across 16 major US hurricanes since Katrina the insurers-dip / rebuilders-rally story comes back **wrong-signed and insignificant**, a long-rebuilders/short-insurers trade loses at every horizon, and the only nominally significant cut (a rebuilder rally in the 7 catastrophic storms, *t* = 2.02 on n = 7) fails the sub-era robustness bar.

## What we tested

The "Waffle House Index" is FEMA's informal severity gauge: if a storm shuts the
always-open chain, it's catastrophic. Turned into a market question, the obvious
tradable corollary is that a **major US natural disaster** should *dip* the property &
casualty insurers who pay the claims (ALL/TRV/PGR) and *rally* the home-improvement
names who sell the rebuild (HD/LOW). We steelman it with a **market-adjusted event
study** on a hand-curated table of **16 major US hurricane landfalls, 2005 → 2024**
(Katrina, Sandy, Harvey, Irma, Michael, Ian, Helene, Milton …, NHC/NOAA public dates):
per-event cumulative abnormal returns in a [−10..+20]-session window (wide because
storms are *forecast* ahead of landfall), a paired rebuilders-minus-insurers directional
test, a 20-seed random-calendar placebo, a catastrophic-only and two-era robustness cut,
and a costed long-short timer — with the honest caveat that ~16 events is low power. A
deterministic synthetic tape with a *planted* insurer-down / rebuilder-up drift is the
positive control. **Survivorship is named on the Signal axis** (these are the surviving
large-caps, not the full P&C universe). One documented execution lag (landfall snapped to
the next NYSE session). **As-of 2026-06-30.**

**Dedup.** Distinct from [283-hurricane-season](../283-hurricane-season/) (the *seasonal*
calendar, not the disaster event), [316-bank-failure](../316-bank-failure/) (same
event-study machinery on a *bank-failure* shock calendar),
[313-geopolitical-shock](../313-geopolitical-shock/) (war/terror shocks) and
[707-plane-crash-effect](../707-plane-crash-effect/) (the closest cousin — a disaster
calendar testing a market-wide dip and a sector extra-drop); this study's own axis is
what a hurricane does to listed **insurers vs. rebuilders** specifically.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why insurers *should* dip and rebuilders *should* rally, what the tape actually shows, and why "the market already knew the storm was coming" wins |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the market-adjusted event-study anatomy, the paired directional test, the random-calendar placebo, the catastrophic-only look-elsewhere caveat, the costed long-short timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`waffle_index/`](waffle_index/). The disaster calendar is hand-curated from
NHC/NOAA public landfall dates; SPY and ALL/TRV/PGR/HD/LOW are total-return closes
fetched via yfinance, survivorship named on the Signal axis. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
