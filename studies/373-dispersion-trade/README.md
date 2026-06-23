# Study 373 — Dispersion-Trade 🌐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is index vol reliably "cheap" vs its names? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The gap is **real and reliably positive** — index vol **15.6%** vs average single-name vol **24.2%**, a **+8.5 vol-pt** gap **99.6%** of days — but that is the **subadditivity identity** (σ_index ≈ √ρ · avg σ for ρ < 1), not an edge. The *carry* that would monetize it fails decisively: **HAC *t* = 0.19**, placebo *p* = 0.43, Sharpe 0.17, win-rate 44%. Real-as-identity, weak-as-edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs barely matter (net Sharpe **0.15**). The killer is the *shape*: the realized-proxy carry is **negative-median, +4.1-skew convexity** — a slow bleed whose rare payoffs cluster in the very crashes (worst day June 2009) that detonate a short-correlation book. A Sharpe-0.15 stream you must finance is **not** a NAV-scale strategy. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | "Index vol is cheap" confuses an **identity** with an **edge**. A synthetic basket with **zero** planted carry shows the gap positive **100%** of days and a carry of *exactly zero* (HAC *t* = 0.00). The cheapness is the **price of being short correlation** — paid, not free. |

> **In one sentence:** yes, index volatility sits reliably below the average of its single-name vols — by **8.5 points, 99.6% of the time** — but that gap is the subadditivity identity (you're short correlation), and on a transparent realized-vol proxy the carry that's supposed to harvest it is a thin, 44%-win-rate, +4.1-skew bleed with **HAC *t* = 0.19**, so it is real-as-arithmetic, weak-as-edge, and a mirage as a strategy.

## What we tested

The dispersion trade — **sell index volatility, buy single-name volatility** — is the vol desk's favourite "free lunch," justified by the claim that index vol trades reliably *cheap* versus its constituents. True dispersion is an **implied-vol / implied-correlation** position (sell index variance swaps, buy single-name variance swaps); implied vols aren't on yfinance, so we build a transparent **realized-vol proxy**: rolling 21-day realized vol of **SPY** vs. the equal-weight average realized vol of a fixed **40-name** large-cap basket, and the gap `avg_single_vol − index_vol` (labelled a proxy throughout). We confirm the gap is real but **mechanical** (subadditivity), then build the long-single / short-index variance **carry book** struck at a trailing reference, and put its mean through a HAC *t*, a block-sign-flip placebo, a cost sweep, and a fat-tail audit. A deterministic synthetic correlated basket with a *planted* carry knob confirms the engine recovers a real edge **and** that the mechanical gap manufactures **zero** carry when the true edge is zero.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why index vol *has* to be lower than the average stock's, why "cheap" is the cost of insurance not a gift, and why the trade bleeds most days — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the subadditivity identity, the realized dispersion gap & implied correlation, the carry book with HAC *t* + block-sign-flip placebo, costs, the skew/tail, and a synthetic faithful-engine / planted-carry control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dispersion_trade/`](dispersion_trade/). Vol here is an explicit **realized-vol proxy** for the implied-vol / implied-correlation trade; the basket is a fixed 40-name stand-in for SPY's constituents. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
