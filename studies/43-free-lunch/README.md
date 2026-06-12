# Study 43 — Free-Lunch 🍽️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-beta assets beat the market? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Barely. Self-financed and measured excess-of-cash like everything else, the beta-neutral BAB book's **gross** Sharpe is **0.32 — below SPY's 0.48**. The tilt exists but doesn't clear the simplest benchmark. |
| **Tradability** — does it survive the leverage it needs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. Beta-neutrality requires running the low-beta leg at **2.77×**; charge a realistic financing *spread* over the T-bill and the Sharpe falls **0.32 → 0.23 (1%) → 0.09 (2.5%)**, negative past ~3.5%. |
| **"Free lunch"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The premium **is** the rent on the leverage (cf. [Study 30 House-Edge](../30-house-edge/)) — and the book's good early years were half a **levered duration trade** (corr +0.55 with TLT); the intra-equity version is flat and small (0.23 → 0.20). |

> **In one sentence:** betting against beta is sold as a low-risk free lunch, but priced as the self-financed ~2.8×-levered book it actually is, it trails the market even gross, a realistic financing spread erases what's left, and its best years were mostly a bond-bull trade in disguise — the lunch was the leverage bill all along.

## What we tested

The "low-risk anomaly": low-beta assets earn more per unit of risk than high-beta ones, and the **BAB factor** (Frazzini & Pedersen 2014; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.594`) harvests it by going long low-beta (levered up to beta 1) and short high-beta (levered down). We build it on a liquid-ETF cross-section, 2000–2026, with an explicit **self-financing ledger** — borrowed slice at T-bill + spread, short rebate at T-bill − fee, every Sharpe excess-of-cash — and ask the question the headline quietly skips: **what happens when you pay for the leverage the low-beta leg requires?** We sweep the financing spread, compare to simply owning the market like-for-like, and check what the "low-beta half" of a cross-asset panel really is (mostly bonds and gold — the duration confound that masquerades as decay). The offline control is a synthetic factor world with known betas and a tunable low-beta premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "safe stocks beat risky ones" is true *and* not a free lunch, and where the borrowed money goes |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the self-financing ledger, the 2.77× leverage, the excess-of-cash race, the financing sweep, the TLT confound behind the "decay" |

The fingerprinted real-data run (13 ETFs + ^IRX, 2000–2026, fp `482c3b72db16`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic factor world in [free_lunch/data.py](free_lunch/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
