# Study 724 — Pumpkin-Spice-Season 🎃

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does SBUX beat the market in pumpkin-spice season? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The Aug–Nov excess-over-SPY spread is **+0.56%/mo, Welch t = 0.65**, block-bootstrap 95% CI **[−0.84%, +1.92%]** — a coin flip. No month clears \|t\| ≥ 2 (largest is March, off-thesis), and the placebo ranks the pumpkin window only **#2 of 12** four-month windows. |
| **Tradability** — does the PSL calendar add value? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Holding the SBUX-vs-SPY pair **only** in the season earns **Sharpe 0.27** — *below* holding it all year (**0.35**). The season dilutes the edge; the long-only rotation only edges SPY (0.66 vs 0.61) by borrowing SBUX beta for four months. |
| **"Pumpkin premium"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The season spread is **−0.69%** in 1993–2009 and only **+1.84%** (t = 2.06) in a snooped 2010-on half. A sign-flipping premium is no premium — and the QSR basket (t = 0.71) doesn't rescue it. |

> **In one sentence:** the most reliable ritual in American retail leaves *no* fingerprint in Starbucks' market-relative returns — the Aug–Nov excess is t = 0.65, the launch month August is the weakest of the four, an off-thesis spring window wins the placebo, and the "season only" trade underperforms just holding the pair; what's real is survivorship, not a calendar.

## What we tested

The folklore: Starbucks launches the Pumpkin Spice Latte in late August, "pumpkin spice season" runs
Aug–Nov, and all that autumn demand makes **SBUX beat the market** into the fall — so rotate into
Starbucks for the season and collect a seasonal premium ([Starbucks' own annual PSL launch
announcements](https://stories.starbucks.com/); the recurring "pumpkin spice economy" business press).
We test the tradable version on **SBUX minus SPY** monthly total-return excess (1993–2026, 400 months,
Yahoo Finance): per-month HAC t-stats, a season-vs-off Welch spread with a block-bootstrap CI, a
12-window placebo, a rotation race gross and net, a sub-period split, and a coffee/QSR-basket
robustness leg. Survivorship (SBUX is one hand-picked survivor) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a great stock fools you into seeing a calendar — the story, the placebo, the survivorship trap in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month HAC t-stats, season Welch spread + block-bootstrap CI, the 12-window placebo, rotation vs market-neutral pair, sub-period split, QSR basket |

The fingerprinted real-data run (SBUX + SPY + ^IRX, 1993–2026, fp `1d50b4e1e153`) is in
[docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py)
(`--fetch` to download); the offline machinery proof runs on the synthetic world in
[pumpkin_spice_season/data.py](pumpkin_spice_season/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
