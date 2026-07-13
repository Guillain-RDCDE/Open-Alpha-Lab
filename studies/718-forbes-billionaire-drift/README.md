# Study 718 — Forbes-Billionaire-Drift 💰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a tradable drift after the list? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The holdable post-list CAR is **+12.4%** vs the S&P but at Welch *t* = **1.70** (placebo *p* = 0.108) it **fails** the *t* ≥ 2 bar — and it **dissolves** to **+1.5%** (*t* = 0.22) as a plain excess over the market, **+7.7%** (*t* = 1.09) vs a tech benchmark. The only effects that clear \|*t*\| = 2 are the pre-list run-up (**+5.7%**, *t* = 3.02 — pure selection you can't trade) and a **negative** list-day dip (**−3.0%**, *t* = −2.81). **Survivorship** named here — it biases the survivor mean *upward*. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Per-name outcomes span **−64% to +85%**, what survives is tech beta you could buy in one QQQ ticket, and the two loudest fresh-billionaire vehicles of the era — **Nikola** and **Luminar** — cratered and **delisted** off the tape. Nothing benchmark-robust to harvest. |
| **"Billionaire glow?"** | ![Misattributed](https://img.shields.io/badge/Billionaire_glow%3F-Misattributed-8b949e?style=flat-square) | Reverse causality: the stock didn't rise *because* the founder made the list — the founder made the list *because* the stock rose. Strip that run-up and the tech beta, and there is no coronation glow left. |

> **In one sentence:** the trade "buy the newly-minted Forbes billionaire's company" looks tempting because these vehicles ran up **+5.7%** vs the market *before* the list — but that run-up is the very thing that put the founder on the list (selection, not signal), the coronation itself is a small **−3.0%** sell-the-news dip, and the holdable post-list "drift" of +12.4% fails significance (*t* = 1.70) and melts to +1.5% once you stop mistaking freshly-IPO'd tech beta for alpha — inside a portfolio that secretly owned two names (Nikola, Luminar) that went to zero.

## What we tested

We hardcode a **transparent, cited table of ~27 newly-minted-billionaire vehicles** — the public companies behind founders who first appeared on the annual [Forbes World's Billionaires list](https://www.forbes.com/billionaires/) (Airbnb, Snap, Robinhood, Rivian, Coinbase, Palantir, DoorDash…) — and run a textbook short-window **event study** around each list's publication date: the **cumulative abnormal return** (CAR), where "abnormal" means the stock's return minus a **market-model** fit (`stock = α + β·SPY`) estimated on a clean pre-list window. We measure three windows — the **run-up** [−63,−1], the **list days** [0,+2], and the **holdable post-list drift** [+1,+63] — add a placebo null sized to the event count, a one-day execution lag with costs, and the decisive **alpha-vs-beta** check: re-benchmark the drift against a tech ETF (QQQ) and against a plain β = 1 excess return. We name the two honesty problems loudly: **reverse causality** (the run-up *causes* the listing) and **survivorship** (Nikola and Luminar delisted, biasing the survivor mean upward) — both on the Signal axis. A deterministic synthetic control confirms the engine recovers a planted post-list drift **and** that ~25 ultra-volatile events can't reach significance unless the edge is enormous.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the run-up is a trap not a signal, what "abnormal return" means, how the +12% melts when you change the benchmark, and the two fresh billionaires that went bust — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model CARs in three windows, the reverse-causality selection trap, an alpha-vs-beta benchmark decomposition, a placebo null, survivorship folded back in, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`forbes_billionaire_drift/`](forbes_billionaire_drift/). The event table is an explicit **hardcoded, cited** stand-in (Forbes sells no new-entrants feed); the priced tape is **survivor-biased** (Nikola & Luminar delisted), named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
