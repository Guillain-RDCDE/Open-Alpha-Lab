# Study 332 — Downside-Beta 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Raw β⁻ quintile spread **+12.2%/yr**, HAC *t* = **+2.56** — but the *relative* downside beta (β⁻ − β, the part beta doesn't explain) earns **−0.5%/yr**, *t* = **−0.14**. Real on the raw sort · None on the downside-specific part. Survivorship-biased basket. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of 10 bps one-way the spread slips below the bar (*t* = **+1.55**), before short borrow on a heavy monthly beta-tilted long-short; the tradeable part is plain market beta, cheaper to buy as an index. |
| **Distinct from plain beta?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A plain-β sort earns nearly the same (+10.5%/yr, *t* = +2.09); strip β out and the premium is gone (*t* = −0.14). The downside premium is the beta premium wearing a costume. |

> **In one sentence:** stocks that crash with the market *do* seem to pay more — but only because they're high-beta stocks, and once you remove plain beta the celebrated "downside-risk premium" is a flat line through zero that dies under costs anyway.

## What we tested

Ang, Chen & Xing (2006), *Downside Risk*, found that stocks with high **downside beta** (β⁻ — beta measured only on down-market days) earn about **6%/yr** more than low-β⁻ stocks, and claimed this is *not* subsumed by ordinary market beta — a genuinely new risk factor born of loss aversion. We take that literally: each month we estimate every stock's trailing β⁻, sort the cross-section into quintiles, and hold a one-month-lagged long-short — then run the same sort on **plain beta** and on **relative downside beta** (β⁻ − β, Ang-Chen-Xing's own decisive control). A deterministic synthetic firm × day panel with a tunable, plantable downside premium is the positive control; the real tape is a survivorship-biased large-cap S&P 500 basket of daily total-return-adjusted closes, 2005–2026, loaded behind the opt-in guard.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the loss-aversion story, why the raw number is real, and the beta control that busts it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | β⁻/β/β⁻−β sorts, HAC *t*, block-bootstrap CIs, the synthetic positive control, the cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`downside_beta/`](downside_beta/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
