# Study 370 — Zero-DTE-Options ⏱️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — did the 0DTE boom make the tape more mean-reverting? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The daily **intraday** leg's lag-1 autocorrelation *did* drift the predicted way post-2022 (**−0.077 → −0.100**), but the change is small, **fails t ≥ 2** (block-bootstrap Welch *t* = **−0.28**, placebo *p* = **0.40**), **reverses sign** in the close-to-close proxy (**+0.10**), and grows only as you fish the break date later. A daily feed also *cannot see* the intraday object the claim is about. Direction-consistent but unproven ⇒ **Weak**, never Real. |
| **Tradability** — can you trade the "pin"? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The naive fade-the-prior-move trade is **weaker** post-2022 and **loses money net of 1 bp** (post-boom net **−0.05 bp/day**, Sharpe ≈ **0**). No positive, cost-surviving, post-boom edge to allocate to on the daily tape. |
| **Tape pinned by 0DTE?** | ![Unproven](https://img.shields.io/badge/Tape_pinned%3F-Unproven-8b949e?style=flat-square) | The 0DTE *boom* is unmistakable — the nearest-expiry chain (a true 0DTE day) shows **≈90%** of **6.8 M** contracts of volume within **±1%** of spot, across **32** near-daily expiries — but on every daily proxy we can build, the claim it **pinned** the index is indistinguishable from a random split. Real product, unproven tape effect. |

> **In one sentence:** the 0DTE-options boom is real and enormous (≈90% of a same-day chain's volume sits at-the-money), but on the only data a free feed gives — daily SPY OHLC — the post-2022 intraday tape is *not measurably* more mean-reverting (block-bootstrap Welch *t* = −0.28, placebo *p* = 0.40, and the close-to-close proxy moves the *opposite* way), the naive "trade the pin" rule actually pays *less* after the boom and loses money net of a basis point, and the intraday phenomenon the story is about simply cannot be seen from daily bars at all.

## What we tested

There is no free intraday SPX option tape, so we test the believers' claim on **explicit daily proxies** (labelled throughout): from daily SPY OHLC we build the **open→close ("intraday") leg** and measure its **lag-1 autocorrelation** — a mean-reversion gauge, where *more negative* means *more pinned* — and we split the tape at **2022**, when same-day options went daily. We attach the **nearest-expiry option-chain snapshot** (as-of 2026-06-22, itself a 0DTE expiry) as the only *direct* 0DTE evidence yfinance gives. Inference is a **block-bootstrap Welch *t*** on the cross-2022 change plus a **placebo null** over random break dates, with a 1-day-lagged, cost-charged "trade the pin" check and a deterministic **synthetic control** that plants a known pinning regime switch (so we know the engine detects a real break and invents none).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what 0DTE options are, why the "pinning" story is plausible, what a daily feed can and can't see, and why the post-2022 tape doesn't look measurably more pinned — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | intraday-leg autocorrelation pre/post-2022, a block-bootstrap Welch *t* + placebo-break null, break-date robustness, the cost-charged pin trade, the 0DTE chain snapshot, and a synthetic pinning-regime control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`zero_dte_options/`](zero_dte_options/). The intraday tape here is an explicit **daily proxy** (open→close leg), not true intraday microstructure, and the option chain is a single **snapshot**. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
