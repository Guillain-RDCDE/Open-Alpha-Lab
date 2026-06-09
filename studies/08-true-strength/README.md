# Study 08 — True-Strength ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The z-scored TSI is **84% spanned** by the MACD line and the RSI (pooled R² **0.835**); its zero-cross position agrees with the MACD's **99.4%** of days and its long/short **equity curve correlates 0.994** with the MACD's. Not a different signal — the same one, double-smoothed. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The TSI crossover nets a **0.61** Sharpe at 10 bps — but that's the **long-side equity beta** of a filter in the market ~50% of the time: strip the structural long bias (long/**short**) and the TSI's *timing* Sharpe collapses to **0.05** (MACD 0.05, RSI **−0.29**). You're paid for holding stocks, not the oscillator, and the thin remainder decays Sharpe 0.77→0.15 across a 0→40 bps cost sweep. |
| **"Truer" than MACD/RSI, as the name claims?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Three indicators, one trade. The "true strength" branding promises a distinction the data denies. |

> **In one sentence:** the True Strength Index is a **real but utterly generic** momentum oscillator — 84% reconstructable from the MACD and RSI, with a position that agrees with the MACD's 99.4% of the time and an equity curve indistinguishable from it (ρ = 0.994) — whose standalone "edge" is the long-equity beta you'd get from *any* trend filter, not a truer reading of strength.

## What we tested

The "**True** Strength Index" — William Blau's double-smoothed momentum oscillator — claims, via its very name and [QuantifiedStrategies.com's TSI write-up](docs/references.md), to be a cleaner, *truer* read on momentum than the MACD or RSI. We can't test their paywalled rule, so we test the claim the name itself makes: that this is a **distinct, truer** signal. All three oscillators are computed on textbook settings (TSI 25/13/13, MACD 12/26/9, RSI 14), read as a zero-centred level and z-scored per name, over the cached liquid **174-name** US universe (1962–2026) — like-with-like, no fitted parameter.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story — three oscillators, one trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the teardown — spanning-R², sign agreement, equity-curve ρ, the alpha-vs-beta cut, the cost sweep, the Reality Check |

The headline run and every number lives in [docs/results.md](docs/results.md); reproduce it via [examples/verify_real.py](examples/verify_real.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
