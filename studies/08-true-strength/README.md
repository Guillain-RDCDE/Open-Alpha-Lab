# Study 08 — True-Strength ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The z-scored TSI is **84% spanned** by the MACD line and the RSI (pooled R² **0.835**); its zero-cross position agrees with the MACD's **99.4%** of days and its long/short **equity curve correlates 0.994** with the MACD's. And the synthetic control shows the collinearity is **mechanical to the filters** (spanning R² **0.86 on pure random walks**, 0.91 on planted structure): three smoothings of the same one-bar price change co-move on *any* input, so the TSI cannot be a distinct signal on any data. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The TSI crossover nets a **0.63** Sharpe at 10 bps per side (≈ 20 bps round-trip) — but that's the **long-side equity beta** of a filter in the market ~50% of the time: the *same rule* run long/**short** earns **−0.42**, and the zero-cross long/short book sits at **+0.05** (MACD +0.05, RSI **−0.27**). The Reality Check agrees once the beta is stripped: in **excess of buy-and-hold** the best of 24 variants has Sharpe **−0.83** (p = 1.00 — no variant times the market better than holding it), and the **dollar-neutral** grid's best is **−0.21** (p = 1.00). You're paid for holding surviving stocks, not the oscillator, and the long/flat book decays Sharpe 0.79→0.16 across a 0→40 bps/side cost sweep. |
| **"Truer" than MACD/RSI, as the name claims?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Three indicators, one trade — and the part of the TSI that MACD+RSI *can't* reproduce earns a **gross Sharpe of −0.02** (Lo SE 0.12, HAC *t* −0.2): the pre-declared "TSI adds nothing" criterion, satisfied. Costs only add friction: the residual flips sign ~72×/yr, so 10 bps/side drags that zero to **−0.59 net** — cost arithmetic, not an anti-signal. |

> **In one sentence:** the True Strength Index is a **repaint, not a reading** — 84% reconstructable from the MACD and RSI (a collinearity the synthetic control proves is built into the filters themselves), agreeing with the MACD's position 99.4% of the time with an equity curve indistinguishable from it (ρ = 0.994) — whose standalone "edge" is the long-equity beta you'd get from *any* trend filter, and whose unique residual earns a gross Sharpe of ≈ 0: nothing left to be the "true" in True Strength.

## What we tested

The "**True** Strength Index" — William Blau's double-smoothed momentum oscillator — claims, via its very name and [QuantifiedStrategies.com's TSI write-up](docs/references.md), to be a cleaner, *truer* read on momentum than the MACD or RSI. We can't test their paywalled rule, so we test the claim the name itself makes: that this is a **distinct, truer** signal. All three oscillators are computed on textbook settings (TSI 25/13/13, MACD 12/26/9, RSI 14), read as a zero-centred level and z-scored per name, over the cached liquid **177-name** US universe (1962–2026) — like-with-like, no fitted parameter. One honest caveat: the cache keeps only **today's liquid survivors** projected backwards, so any standalone return number reads as an upper bound — fine for *comparing* three oscillators on identical inputs, which is the question here.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story — three oscillators, one trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the teardown — spanning-R², sign agreement, equity-curve ρ, the alpha-vs-beta cut, the cost sweep, the Reality Check |

The headline run and every number lives in [docs/results.md](docs/results.md); reproduce it via [examples/verify_real.py](examples/verify_real.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
