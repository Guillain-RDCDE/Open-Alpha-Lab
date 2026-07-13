# Study 719 — Met-Gala-Luxury 👗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No cut clears *t* ≥ 2: luxury basket 1-week AR **+0.44%** (*t* = **+0.74**), 1-month **+0.66%** (*t* = **+0.52**), and the 1-month random-window placebo lands the gala month **dead centre** (*p* = **0.500**). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The best honest, zero-look-ahead trade is 1-week **gross** *t* = +1.07; net of costs the 1-month capture is **+0.68%** at placebo *p* = 0.45 — indistinguishable from noise. |
| **Just one name?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The concentration fallback fails too: split into the four houses and none clears the bar — strongest is Richemont/1-month *t* = **+1.00**, LVMH itself *t* = +0.60. |

> **In one sentence:** the Met Gala is a four-hour, hundred-million-dollar advertisement for LVMH, Kering, Hermès and Richemont — and because everyone knows it's coming, the luxury basket does nothing unusual around it, every cut sitting squarely inside the luck cloud.

## What we tested

Markets-meet-culture folklore holds that the Met Gala — fashion's biggest night, the
first Monday in May, wall-to-wall with LVMH/Kering/Hermès/Richemont brands — gives the
luxury complex a "spotlight bump" from all that free global attention. We hardcode all
26 gala years 2000→2025 (three had no gala: 2000 & 2002 cancelled, 2020 COVID; 2021 was
held in September), build an equal-weighted basket of the four listed houses (`MC.PA`,
`KER.PA`, `RMS.PA`, `CFR.SW`), and measure its abnormal return vs the `VGK` Europe
benchmark from the last close before the Monday-night gala through 1 week and 1 month
after — with a random-window placebo, a per-name concentration split, and a
zero-look-ahead tradable-capture test across the 20 galas inside the VGK window (2005→).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the dead-centre placebo, why no single brand pops either, the trade that means nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the random-window placebo, the per-name split, the event anatomy, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`met_gala_luxury/`](met_gala_luxury/). The Met Gala calendar is hardcoded from
Wikipedia; **selection named**: `VGK`'s 2005 inception is the binding floor, so the test
speaks only to the modern (2005→) first-Monday-in-May Met Gala — which is also the only
era in which the date convention reliably holds. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
