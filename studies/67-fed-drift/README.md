# Study 67 — Fed-Drift 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks drift up before FOMC meetings? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, historically huge: **3.1% of sessions (the pre-FOMC days) earned 11.5% of SPY's entire cumulative return** since 1993; pre-2011 it was **19.8%** of all return. |
| **Tradability** — can you harvest it today? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It decayed on publication: pre-FOMC day **+0.243%→+0.094%/day**, diff t **1.7→0.4**. Post-2011 it's a normal day. |
| **"Still an edge after publication"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Textbook McLean-Pontiff decay — arbitraged away once everyone read the paper. |

> **In one sentence:** the pre-FOMC drift was one of the most striking calendar effects ever found — 3% of sessions carried a fifth of the market's entire return before 2011 — but it's a museum piece now: after Lucca-Moench published it, the pre-FOMC day decayed to statistically indistinguishable from any other (t 1.7 → 0.4).

## What we tested

The **pre-FOMC announcement drift** (Lucca & Moench 2015): equities drift up in the ~24 hours before a scheduled FOMC statement. We tag the trading session immediately before each of **264 scheduled FOMC announcements** (1993–2026) and compare SPY's return on those days to every other day — over the full sample, and **split at the 2011 publication era** to test whether the edge survived being written up. The offline control is a synthetic daily world with a known pre-FOMC drift (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how 3% of days carried a fifth of the market — and why that stopped |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the drift table, the Welch t, the pre/post-publication split, the decay |

The fingerprinted real-data run (SPY 1993–2026, fp `d824e220dbca`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads the shared SPY pull); the offline machinery proof runs on the synthetic world in [fed_drift/data.py](fed_drift/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
