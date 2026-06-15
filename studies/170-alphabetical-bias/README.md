# Study 170 — Alphabetical-Bias

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Volume effect confirmed (t = +10.34, ratio 1.096x); return spread −6.90 bps/month, HAC *t* = **−1.24** — indistinguishable from zero; 0 letters survive the Bonferroni bar (|*t*| ≥ 2.90, 27 tests); survivorship-biased universe. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No return edge to harvest; low-turnover alphabet portfolio earns zero gross alpha vs equal-weight benchmark; concentration risk with no compensation. |
| **Attention effect?** | ![Real--but_not_tradable](https://img.shields.io/badge/Attention_effect%3F-Real--but_not_tradable-8b949e?style=flat-square) | ~10% more volume for A–C names is unambiguous over 25 years; markets price it away without leaving a return wedge. |

> **In one sentence:** stocks at the top of the broker alphabet really do get more retail eyeballs and trade ~10% more volume — Jacobs & Hillert (2015) are right about the attention channel — but markets price that attention premium away instantly, leaving a return spread of −6.9 bps/month (HAC *t* = −1.24) and zero letters surviving the Bonferroni scan.

## What we tested

The *alphabetical-ordering bias*: every broker screener, every table, every dropdown sorts stocks A-to-Z by default. That puts Apple (AAPL) at the top and ZoomInfo (ZI) at the bottom, where nobody scrolls. The claim (Jacobs & Hillert 2015, *Journal of Financial Economics*) is that the resulting attention asymmetry inflates turnover for early-alphabet names and may even generate a return premium. We test both channels: equal-weight monthly returns for A–C vs D–Z (the return channel) and average daily volume (the attention channel), on 502 S&P 500 names from 2000 to 2026, with a Bonferroni correction for 27 tests (26 letters + the main group) and survivorship bias named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the list-position story, the attention vs return split, why markets price attention away, the per-letter chart |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, Bonferroni scan, survivorship disclosure, synthetic positive control, the full monthly spread time-series |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`alphabetical_bias/`](alphabetical_bias/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
