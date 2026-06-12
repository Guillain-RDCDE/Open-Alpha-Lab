# Study 41 — Hangover 🥂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does January carry information? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Barely, but it's real: a *down* January precedes a softer rest-of-year (**+2.1%, up 60%** vs **+12.2%, up 87%** — Fisher exact **p = 0.012**, permutation **p = 0.002** on the mean gap). That's the only informative cell, the market still rises after it, and on its own the 60% (Wilson [42%, 75%]) is indistinguishable from the 76% base rate. |
| **Tradability** — can you trade the omen? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Sitting in T-bills after a down January cuts drawdown (−40% → −16%) but only by **reducing equity exposure** — a permanently smaller stock weight does the same. It acts once a year, and its CAGR edge is a **price-only artefact**: on a total-return tape, buy-and-hold's dividends (~3–4%/yr, every year) erase it. |
| **"Predicts the year"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The directional omen is **68% accurate — worse than the 76% you get by always predicting "up"**. As a forecaster it loses to a constant. |

> **In one sentence:** "as goes January, so goes the year" is a **base-rate illusion** — the rest of the year is up ~76% of the time *no matter what January did*, the omen's directional accuracy (68%) is *beaten by always saying "up"*, and its one faint real residue (a weak year after a down January — it does survive Fisher and permutation tests) "trades" only by holding less stock, while any post-1972 decay is too small to tell from noise (Fisher p = 0.29).

## What we tested

Wall Street's oldest calendar omen — the **January Barometer** (Yale Hirsch, 1972; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.365`): if the S&P rises in January, the year is bullish; if it falls, brace. We take the believers' headline at full strength ("up 87% of the time after an up January!") and ask the one question that decides it: **does conditioning on January beat the base rate of just predicting "stocks go up"?** We run it on the S&P 500 back to **1950** (price-only — ^GSPC carries no dividends, and we say so), score directional accuracy against the base rate with Wilson intervals and Fisher/permutation tests on every conditional cell (30 down-Januaries is a small sample — the uncertainty *is* the finding), build the tradable T-bill-after-down-January version against simply holding the index, and split pre/post-1972 for decay. The offline control is a synthetic year-world with a tunable predictive link (and a no-link null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "up 87% of the time" sounds amazing and means nothing, and how a forecaster can lose to a constant |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | base-rate vs conditional accuracy with Wilson intervals, the Fisher/permutation tests on the only informative cell, the exposure-reduction illusion, the (untestable) post-1972 decay |

The fingerprinted real-data run (S&P 500, 1950–2025, fp `fed7f700c91b`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [hangover/data.py](hangover/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
