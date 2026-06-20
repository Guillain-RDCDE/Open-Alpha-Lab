# Study 349 — Regime-Dependence 🪟

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real, resolvable regime pathology on the tape? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | A methodology demo certifies no tradable effect; on the real tape the decade-to-decade Sharpe gaps straddle zero (only 3 regimes — under-powered), and the planted regime split is a *synthetic control*, never market evidence. |
| **Tradability** — is there anything to buy? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The product is a *measuring instrument*, not a strategy. The lesson it teaches kills backtests; it does not mint alpha. |
| **Did ruling one decade prove skill?** | ![It_depends](https://img.shields.io/badge/It_depends-8b949e?style=flat-square) | **Confirmed on ground truth:** a strategy can post a *top* full-sample Sharpe while owing >half its edge to one regime — and a single Sharpe never reveals it. Decade-conditional Sharpe + a drop-best-decade re-estimate + a cross-regime gap CI catch it. |

> **In one sentence:** a single full-sample Sharpe can crown a strategy that merely *ruled one decade* — so before you call it skill, split the tape by regime, drop the best decade, and see whether the edge is still there (a 12-month trend overlay passes with a Sharpe of ~0.9 in *every* decade; a synthetic strategy with a *better* headline Sharpe collapses the moment its one lucky regime is removed).

## What we tested

The strongest version of the trap: *"This strategy printed a Sharpe of 1+ over twenty years — that's skill, not luck."* The literature on backtest overfitting (Bailey & López de Prado 2014) and parameter instability (Pesaran & Timmermann 2002) says a full-sample number can hide its source — and the canonical example is a record fit to one favourable era (the 60/40's 2010s; trend-following's persistence is the counter-example). This is a **research-method demo**: we build a deterministic tape where the ground truth is *planted* — one strategy with a durable edge in every regime, one with a big edge in a single regime and *a higher headline Sharpe* — and ask which lens tells them apart. Then we run the same lens on real canonical strategies (a 12-month trend overlay and a static 60/40) split by decade.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why one great decade can disguise itself as skill, in plain language — and the one chart that gives it away |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | decade-conditional Sharpe, the drop-best-decade HAC *t*, the cross-regime gap block-bootstrap CI, the stability score, and the planted positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`regime_dependence/`](regime_dependence/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
