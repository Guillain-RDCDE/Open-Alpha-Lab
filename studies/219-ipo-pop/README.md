# Study 219 — IPO-Pop

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | First-day pop is real (mean +42%, HAC *t* = +6.24) but unharvestable. Post-close long-run drift: mean inflated by AMZN/EBAY survivors; median 12m = −0.8%; no horizon clears \|*t*\| ≥ 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Pop requires IPO allocation (lottery for retail); post-close returns insignificant and median-negative; severe survivorship bias inflates means. |
| **Does the long-run hangover win?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | Median consistent with Ritter (1991) underperformance (12m −0.8%; 2021 cohort median −60% at 36m), but survivorship bias prevents a clean confirmation. |

> **In one sentence:** the first-day IPO pop is statistically real but not yours to collect; the long-run median is flat-to-negative (consistent with Ritter 1991), but a handful of giant survivors (AMZN, EBAY, GOOGL) inflates every mean and makes the "long-run hangover" ambiguous in this survivorship-biased sample.

## The claim

> *Should you chase the first-day IPO pop, or does the long-run hangover always win?*

## What we tested

Two sub-claims, one study:

**(A) First-day pop.** A hardcoded table of 40 notable US IPOs (1997–2023) with offer prices and first-day closes lets us measure the offer-to-close pop. We find a mean of +42% (median +31%), 94% hit-rate, HAC *t* = +6.24. The pop is statistically real. But it is not harvestable: allocation in hot IPOs is a lottery for retail and a quid-pro-quo for institutions. Public buyers entering at the open face a price already near the first-day close.

**(B) Long-run hangover.** Buying at the first close and holding 6/12/36 months: we compute forward returns from yfinance daily closes. Means are completely dominated by AMZN (+3316% at 36m), EBAY (+528%), GOOGL, and a handful of 2000s-era mega-cap winners. The median at 12m is **−0.8%** and no horizon clears |*t*| ≥ 2. Consistent with Ritter's (1991) long-run underperformance finding, but the survivorship bias in this table means we cannot confirm it cleanly.

**Two big caveats stated prominently:** (1) severe survivorship bias — the table excludes all delisted/failed IPOs; (2) the pop is unharvestable by construction.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the allocation lottery, why you can't pocket the pop, the AMZN effect on means vs medians, the 2021 IPO bust |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-horizon HAC t-stats, survivor-bias accounting, positive control, cost sweep, Ritter underperformance test |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ipo_pop/`](ipo_pop/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
