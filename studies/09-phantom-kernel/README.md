# Study 09 — Phantom-Kernel 👻

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the arrival kernel `λ(δ) = A·e^(−kδ)` real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Under heavy-tailed order reach — with the tail exponent drawn from the band this study measured on real books — the kernel is a **power law**, not an exponential: a **per-order likelihood test** (each order counted once; AIC + Vuong) prefers the power law by **~0.83 nats per order** (AIC gap **+664k** over 400k orders, Vuong **V = +220**); the `k` you'd estimate is **0.27**, a number with no stable meaning, and a `k` that drifts 4× intraday misprices the "optimal" spread by **41%** at the study's own trading horizon (**163%** at the T=1 worst-case bound). The heavy-tail premise is consistent with **real Binance order books** (Clauset/Vuong, one-day snapshot): order size power-law on **4/4** markets, price-distance on **3/4**. |
| **Tradability** — does *skipping* AS "leave money on the table"? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | A **brainless inventory clamp** beats full AS on risk-adjusted P&L whenever inventory isn't dangerous (World A 5-seed Sharpe **3.27 vs 1.59**). AS's genuine benefit shows only in the **hostile** world (Sharpe **1.50**, best of four) — and the article's recommended "rolling-vol" fix **collapses** there (Sharpe **0.15**). |
| **The famous "optimal spread"** — is it the source of the edge? | ![Misattributed](https://img.shields.io/badge/Misattributed-8b949e?style=flat-square) | **Proven by ablation:** swap the phantom `k` fitted on the friction world (**0.272**) for the textbook **0.6** — a ~2.2× move in the model's famous parameter — and AS's World-B Sharpe **doesn't move** (**1.516 vs 1.514**, 5 seeds). The value lives in the **inventory skew**, which is **algebraically free of `k`**; the phantom corrupts only the *spread width* — the half of the model the article crowns, and the half that doesn't carry the edge. |

> **In one sentence:** Avellaneda-Stoikov's celebrated optimal-spread formula is built on an order-arrival law (exponential decay with a constant `k`) that the documented heavy-tailed reality breaks — making `k` a phantom that misprices the spread by ~41% at the study's own horizon — yet the model still earns its keep in hostile markets, because the part that actually works (the inventory skew) never depended on `k`: swapping the fitted `k` for the textbook one leaves the P&L unchanged, and a four-line inventory clamp matches or beats the whole apparatus whenever inventory isn't dangerous.

## What we tested

Avellaneda & Stoikov (2008) derive two "mathematically optimal" quotes — a reservation price skewed by inventory, and a closed-form spread width — and a generation of market-making bots (Hummingbot, HFT desks, on-chain AMMs) quote from them; [the viral write-up that prompted this study](docs/references.md) says not implementing them is "leaving serious money on the table." That closed form exists only because of one assumption: market orders arrive at a rate that fades **exponentially** with quote distance, `λ(δ) = A·e^(−kδ)`, with a stable `k`. Because AS is a theorem about a model world, we test it on a **reproducible, seed-fixed order-flow simulator** running identical code in two worlds — **A (textbook)**, where every AS assumption holds, and **B (frictions)**, wired with the realities the paper omits: heavy-tailed order reach (tail exponent 1.7, drawn from the band measured on the study's own real books; a heavier α=1.2 is kept as a labelled stress case), price jumps, stochastic vol, and informed flow.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the teardown |

The simulator headline run is fingerprinted in [docs/results.md](docs/results.md); its one empirical premise — that real order reach is heavy-tailed — is checked directly on four Binance futures books (per-market fingerprints, one-day snapshot) in [docs/results_real.md](docs/results_real.md), reproducible from [examples/confirm_heavy_tail.py](examples/confirm_heavy_tail.py) (data via [examples/fetch_binance.py](examples/fetch_binance.py)). The **beat-7 worked complement** — *does a jump-robust (bipower) volatility estimate rescue the article's collapsing rolling-vol "production fix"?* — is in [docs/extension.md](docs/extension.md) (verdict: **not rescued** — it mitigates the collapse but adapting spread width stays the wrong lever, which sharpens the core finding), reproducible from [examples/extension.py](examples/extension.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
