# Study 123 — Altman-Z

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hedge (high-Z minus low-Z) = **+2.36%/yr**, HAC *t* = **+0.65**; firm-level corr(Z, return) = **+0.013**. Survivorship-biased upper bound. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Annual rebalance, thin gross spread, survivorship overstates it; implementation wipes the edge. |
| **Distress puzzle confirmed?** | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Low-Z firms do *not* earn a risk premium — distressed stocks return slightly *less* (+20.5% vs +22.9%) — consistent with Dichev (1998), but far too noisy to trade. |

> **In one sentence:** the Altman Z-score barely distinguishes winners from losers in a survivorship-biased S&P 500 panel — the classic "distress puzzle" (low-Z firms don't earn more) shows up weakly but the HAC t-stat of +0.65 is nowhere near the inference bar.

## What we tested

The Altman (1968) Z-score combines five accounting ratios into a composite:
Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MktEq/TotalLiab) + 1.0(Sales/TA).
The model was built to predict corporate bankruptcy, but it is widely repurposed as an
equity signal: the steelman claim is that low-Z "distressed" firms earn higher returns
as compensation for bearing distress risk (a risk-premium story), or alternatively that
high-Z "safe" firms are better businesses and earn persistently higher returns. We
build the Z-score from the desk's shared EDGAR caches (2008–2023, 166 S&P 500 tickers
with all seven required concepts), pair each year-*t* Z with the year-*(t+1)* return
(one-year reporting lag), sort into terciles, and measure the annual high-Z minus
low-Z hedge against a HAC t-stat inference bar. Survivorship bias is explicit.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Z-score formula explained, the distress puzzle in plain language, the Z-bucket return chart, why the signal isn't there |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | annual bucket table, HAC t-stat, bootstrap Sharpe CI, firm-level cross-section, survivorship-bias anatomy, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`altman_z/`](altman_z/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
