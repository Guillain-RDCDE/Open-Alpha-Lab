# Study 93 — Round-Numbers 🔢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | **Split.** Clustering is real — pooled chi² = **380.8** (p ≈ **1.7e-76**), **11.2%** of closes land within 5% of a whole dollar vs 10% uniform. But proximity *forecasts nothing*: the fade's next-day return is **−0.53 bps**, HAC *t* = **−0.18**. Real on the ruler · None on the future. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The fade doesn't pay even **gross** (−0.53 bps/trade); charge a realistic **5 bps/leg** and every fade bleeds **−10.5 bps**. There is no edge to scale. |
| **Beats a random level?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Fading the *round* levels beats fading **random** levels (same spacing, random phase) only **43% of 200 seeds** — a coin-flip leaning *worse* than random. The round-ness carries no information. |

> **In one sentence:** prices really do **cluster** at round numbers (sixty years of evidence, confirmed here with a chi² of 380) — but that lumpy *ruler* says nothing about the *future*: a fade of round levels earns nothing gross, loses money net, and can't even beat fading random levels, so as a trade it's a **mirage**.

## What we tested

The trading-floor lore, steelmanned: *"prices are magnetised by round numbers — they pile up at whole-dollar and round index milestones, and they stall or reverse there, so you can **fade an approach to a round number and collect the bounce.**"* We split the claim into its two limbs and test each on a basket (^GSPC, SPY, AAPL, MSFT, daily): **(a) clustering** — a chi-square that the distance from each close to its nearest round level is uniform (the [price-discreteness literature](docs/references.md) — Osborne 1962, Harris 1991 — predicts this is real); and **(b) tradability** — a round-number **fade** as a fixed-horizon event study (bet on a reversal when the close is within 0.4% of a round level, enter one day later, hold one day, demeaned by the tape's own drift), pinned against a **random-level control** that fades the same grid spacing at a random phase, then swept for cost. A deterministic synthetic tape with **planted magnetism** (a reflective barrier at each round level) vs magnet = 0 is the positive control — the harness banks the planted bounce (HAC *t* = +5.7) and correctly finds nothing on the plain random walk (*t* = −0.1).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why prices cluster, the chi-square in a picture, the fade that looks clever and earns nothing, the random-level coin |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the chi-square per name + pooled, HAC *t* on the fade, the random-level control (200 seeds), the cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`round_numbers/`](round_numbers/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
