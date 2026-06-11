# Study 34 — Aftershock 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price keep drifting after an earnings surprise? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The drift is real and correctly signed but *thin* on the real tape: the dollar-neutral long-high-SUE / short-low-SUE book earns a gross Sharpe of just **+0.30** over 2010–2026 (Newey-West *t* **+1.39**) — present, but not significant at conventional levels on a liquid large-cap universe. PEAD is a durable anomaly (Ball-Brown 1968; Bernard-Thomas 1989), much attenuated since publication. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The book rolls slowly (turnover **0.058/day**, paced by the earnings calendar), yet the gross edge is so small its **break-even cost is only 6.0 bp** — *inside* the realistic equity round-trip band (≈2–10 bp) — and that is **inflated by survivorship bias**. Net @5 bp the Sharpe is **+0.05** (≈ nothing); net @10 bp **−0.20**. |
| **Drift decays as predicted?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Yes — the surprise-signed cumulative abnormal return rises monotonically from **+0.14% (day 0) to +1.15% (day 69)** on real earnings: the Bernard-Thomas (1989) under-reaction shape, reproduced on the tape. The drift is genuinely there; it is just too small on liquid names to clear costs. |

> **In one sentence:** post-earnings drift is real and shows the textbook rise-then-flatten shape on the real tape, but on a liquid S&P 500 universe it is `WEAK` (gross Sharpe +0.30, *t* +1.39) and a tradability `MIRAGE` — its 6 bp break-even sits inside realistic costs and is survivorship-inflated.

> ✅ **Real-tape run, fingerprinted.** Run on cached EDGAR quarterly EPS → a seasonal-random-walk SUE (`surprise_q = eps_q − eps_{q−4}`, standardised by the stock's own trailing dispersion; announcement = earliest filing) traded against the cached S&P 500 split/dividend-adjusted-Close panel: **488 names, 23,000 events, 2010-01-04 → 2026-06-05, fingerprint `83140e2fef71`**. Reproduce offline via [examples/verify.py](examples/verify.py); the synthetic control that backs the tests is [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py). **Caveat:** the panel is *current* index membership ⇒ survivorship bias (delisted losers absent), which inflates the magnitudes; the qualitative verdict is robust.

## What we tested

The desk's idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.2**, earnings momentum / PEAD). The steelman: a stock under-reacts to its earnings *surprise*, so the price keeps drifting in the surprise's direction for weeks after the announcement (Ball-Brown 1968; Bernard-Thomas 1989) — and a book that goes long positive-surprise names and short negative-surprise names, rolled as earnings land, harvests that drift. We measure it on the **real tape** — cached EDGAR quarterly EPS turned into a seasonal-random-walk SUE (`surprise_q = eps_q − eps_{q−4}`, standardised by the stock's own trailing dispersion; announcement = earliest filing) traded against the cached S&P 500 split/dividend-adjusted-Close panel, entry lagged one day, strictly causal — and reproduce the Bernard-Thomas drift-decay curve. A seeded synthetic control (a known surprise→drift relationship, plus a null where the same surprises are noise) backs the test-suite.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why prices drift after earnings, the aftershock, and why on liquid names the drift is too small to trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: gross-vs-net Sharpe on the real tape, the Newey-West *t*, the cost wall & 6 bp break-even, the drift-decay curve, the holding-period sweep |

The real run — every fingerprinted, as-of'd number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the drift-decay curve + holding-period sweep) is in [docs/extension.md](docs/extension.md). Reproduce the real tape via [examples/verify.py](examples/verify.py); the synthetic control that backs the tests is [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
