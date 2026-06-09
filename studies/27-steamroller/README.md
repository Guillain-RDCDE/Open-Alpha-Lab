# Study 27 — Steamroller 🚧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-rate currencies out-earn? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — uncovered interest-rate parity fails, so the rate differential is a real, paid premium. Our synthetic G10 control recovers it (**+2.1%/yr**, Newey–West *t* = **+2.8**, Sharpe **+0.60**) and the full-UIRP null is flat (*t* **−0.2**); decades of academic evidence agree (Lustig–Verdelhan 2007; Menkhoff et al. 2012). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It's *cheap to run* (rates move slowly → low turnover, so it survives costs) — but it carries a fat negative tail: monthly skew **−1.54**, max drawdown **−28%** on the control, the global-risk-off crash that flattens carry every cycle. |
| **Crash risk?** | ![Severe](https://img.shields.io/badge/Severe-8b949e?style=flat-square) | The crash *resists* the desk's usual fix: vol-targeting lifts the Sharpe (**+0.68 → +0.95**) but does **not** shrink the drawdown (**−28% → −44%**) — it can lever you *into* the jump, because the crash is a sudden risk-off unwind, not a forecastable volatility build-up. |

> **In one sentence:** the carry trade is a real, durable, cheap-to-run premium — and it is rent paid for standing in front of a steamroller: a sharply negative-skewed crash that arrives all at once and shrugs off the vol-management that tamed the desk's other crashes.

> ⚠️ **Real run pending one fetch.** This study's real tape is **G10 rates + FX from FRED** (free, no API key) and has no pre-populated cache — run `python examples/verify.py --fetch` once to download it and write the fingerprinted [docs/results.md](docs/results.md). The verdict above is earned on the fully-validated synthetic control and the literature; the offline core reproduces via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

## What we tested

The desk's tenth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§8.2**, the FX carry trade). The steelman: borrow a low-interest currency, lend a high-interest one, and pocket the gap, because uncovered interest-rate parity fails (the high-rate currency doesn't depreciate enough to offset its yield). We prove the engine on a synthetic G10 with a *baked* carry premium punctuated by sticky risk-off crashes (and a full-UIRP null that earns nothing), run the dollar-neutral carry book (long high-rate, short low-rate), and show — uniquely on this desk — a crash that the vol-targeting overlay from [Study 16](../../16-storm-shy/) can't dodge.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: steady nickels from the rate gap, the steamroller crash, and why risk management can't dodge it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the carry premium by rate bucket, the Newey–West *t*, the negative-skew/downside-concentration crash, and the vol-managed comparison |

The real run — every fingerprinted, as-of'd G10 number — is in [docs/results.md](docs/results.md) (after one `--fetch`); the **beat-7 worked complement** (why vol-targeting lifts the Sharpe but not the drawdown) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` to download G10 rates + FX from FRED).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
