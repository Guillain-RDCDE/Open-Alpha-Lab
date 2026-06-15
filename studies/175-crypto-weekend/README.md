# Study 175 — Crypto-Weekend 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Weekend premium **−0.038% / day**, HAC t = **−0.51**; Bonferroni-adjusted threshold |t| ≥ 2.39 for 3 simultaneous tests. Not significant pre-2023 or post-2023. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Weekend timing rule earns **+0.082% / day gross** — *below* the unconditional buy-and-hold mean (+0.120% / day); inferior to B&H before any cost, firmly negative after 5 bps round-trip. |
| **Vol claim — Inverted** | ![Inverted](https://img.shields.io/badge/Vol_claim%3A-Inverted-8b949e?style=flat-square) | Weekends are significantly **calmer** (vol ratio 0.72, Levene t = **−9.87**). The 'thin-liquidity, wild-swings' narrative is exactly backwards. |

> **In one sentence:** the Bitcoin weekend-pump/dump is a myth — weekends earn less per day than weekdays (t = −0.51, nowhere near significant), are significantly *quieter* not wilder (vol ratio 0.72), and a weekend-only timing rule is inferior to buy-and-hold before the first basis point of cost.

## What we tested

The crypto-market folk wisdom that Bitcoin weekends are "different" — either pumped by retail FOMO or dumped by thin liquidity — because traditional banking rails slow on weekends, stablecoin minting decreases, and professional market-makers supposedly log off. We test all three sub-claims simultaneously on BTC-USD daily (Yahoo Finance, 2014-09-17 to 2026-06-14, n = 4,289 days, 1,226 weekend days) with a Bonferroni-corrected inference bar (|t| ≥ 2.39 for α = 0.05 / 3), a pre/post-2023 regime split (testing the 'FedNow / banking-normalisation killed it' hypothesis), and a weekend-only timing rule vs buy-and-hold.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the pump/dump story, the vol inversion in plain language, why the timing rule is inferior to holding BTC unconditionally |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Bonferroni threshold, Levene variance test, pre/post-2023 regime split, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crypto_weekend/`](crypto_weekend/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
