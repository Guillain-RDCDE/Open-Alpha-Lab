# Study 94 — Level-Pegging ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | It **splits by regime**: equal-weight beat cap-weight by **+2.60%/yr** in 2003-2015 (HAC *t* +1.97) and **lagged by −2.74%/yr** in 2015-2026 (HAC *t* −1.57). The split is real — block-bootstrap *p* = **0.019** on the difference-of-differences — but the **full-sample edge is the wrong sign**: alpha **−0.23%/yr** (HAC *t* −0.21), HAC *t* on the daily diff ≈ **0**. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | As *"just buy RSP and always win"*, no: RSP **trailed SPY** over its entire investable life — relative wealth peaked at **1.305** in 2011 and is now **0.936**. Net of RSP's higher fee (20 vs 9 bps/yr) it earns **−0.32 pts/yr**. The tilt only pays if you correctly **time the breadth regime** — which the claim tells you not to do. |
| **Always wins?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | "Always" is false in the data: EW *led* the broadening 2003-2011 tape and *lost* the mega-cap-led run that followed. It's an anti-concentration **bet**, not a free lunch. |

> **In one sentence:** equal-weighting is a **small-cap / anti-concentration bet that pays in broadening markets and loses in narrow ones** — it genuinely beat the cap-weight S&P from 2003-2015, then *lost* through the mega-cap run, so over RSP's full life it has earned **less** than SPY; "equal-weight always beats" is **busted**.

## What we tested

The marketing claim, at full strength: *"equal-weighting **always** beats cap-weighting — the equal-weight S&P (RSP) outperforms the cap-weight S&P (SPY) because it tilts to smaller names and harvests a mechanical rebalancing premium. Just buy RSP."* (Cf. S&P's Equal Weight Index fact sheets; **Plyakha, Uppal & Vilkov 2012/2014**, *Equal or Value Weighting? Implications for Asset-Pricing Tests*.) We run both arms as **total-return buy-and-hold** index curves on the real **RSP-bounded window (2003+)**, net each of its expense ratio, and ask the two questions the word "always" can't survive: is there a **full-sample edge** (no — it's thin and the wrong sign), and is it **stable across regimes** (no — it flips, *p* = 0.019). A deterministic synthetic panel is the positive control: when small names carry the planted premium EW must win, when a few mega-caps dominate EW must lose — the spine tests both.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two arms, the equity curves, the RSP/SPY relative-wealth line that peaks in 2011 and falls, why "always" is the broken word |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the return difference, the alpha-vs-beta read, the regime split with a block-bootstrap test of the difference-of-differences, the fee drag |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`level_pegging/`](level_pegging/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
