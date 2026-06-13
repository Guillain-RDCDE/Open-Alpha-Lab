# Study 96 — New-Year-Pop 🐤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is January's small-minus-large spread really special? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. January's small-cap-minus-large-cap spread is **+0.18 pts** above the rest of the year on ^RUT/^GSPC since 1990 (HAC *t* = **+0.31**) and **+0.07 pts** on IWM/SPY since 2000 (*t* = **+0.12**) — both nowhere near the bar. Small beats large in only **16/36** Januaries (44%, Wilson95 [30%, 60%]): at or below a coin. |
| **Tradability** — does tilting to small for January pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. A timer holding small in January and large otherwise earns **8.93%/yr** vs **9.02%** for just holding large caps (and 0.593 vs 0.619 Sharpe on the tradable pair), net of costs. The tiny edge over small-cap buy-and-hold is just *being in large caps 11 months a year*. |
| **Decayed since Keim/Reinganum (1983)?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Can't test it here. Yahoo's ^RUT history starts in 1990, so the strong 1970s era is **outside the data**; on the 9 vs 27 Januaries we have, the pre/post-2000 difference is **−0.22 pts** (Welch *t* = **−0.13**) — a powerless zero. The effect is simply *already gone* on the modern tape. |

> **In one sentence:** the January small-cap effect that Keim and Reinganum documented in 1983 is **statistically invisible** on every tape Yahoo gives us (post-1990) — January's small-minus-large spread is a coin-flip, a small-in-January timer can't even beat just holding large caps, and the famous decay can't be measured because the strong era predates the data — a textbook anomaly that was real once and isn't here anymore.

## What we tested

The **January small-cap effect**, stated at full strength: *"small-cap stocks systematically beat large caps in January — driven by December tax-loss selling rebounding and year-end window-dressing — so tilt to small caps for the turn of the year."* (Keim 1983, *Size-related anomalies and stock return seasonality*; Reinganum 1983.) We work on the **monthly small-minus-large return spread** and ask the sharp question: is **January's** spread significantly higher than the other eleven months? We test two real pairs — the long sample **^RUT vs ^GSPC** (price-only, since 1990 — stated honestly, Yahoo has no dividends for these indices) and the tradable **IWM vs SPY** (total-return, since 2000) — with a HAC *t* on the January-vs-rest contrast, a Wilson interval on the hit-rate, a pre/post-2000 test of the *difference* (the famous post-publication decay), and a seasonal timer vs buy-and-hold of each leg. A synthetic control plants a known January bump (the harness detects it) or none (it doesn't) as the positive/negative control. This is the *January seasonal in the size spread* — distinct from the unconditional size premium ([Study 44 — Growth-Spurt](../../44-growth-spurt/)) and from the as-January-goes barometer ([Study 80 — Cold-Open](../../80-cold-open/)).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the spread-by-month bar chart, January as a coin-flip, why a small-in-January tilt can't beat just holding large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the January dummy, Wilson hit-rate, the pre/post-2000 decay test (and why it has no power), the seasonal-timer tradability read |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`new_year_pop/`](new_year_pop/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
