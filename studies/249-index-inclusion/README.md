# Study 249 — Index-Inclusion

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Full-sample pop mean +1.31% (HAC *t* = +0.86, n=46) — driven entirely by TSLA's +59% outlier. Ex-TSLA mean +0.02% (*t* = +0.03). Literature (Shleifer 1986, Lynch & Mendenhall 1997) documents the effect pre-2002; current evidence is too noisy to confirm. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Ex-TSLA the gross pop is statistically zero; round-trip costs (10 bps) consume what little remains. Event frequency is ~2–3/yr in a liquid universe. The trade is crowded, front-run by arb desks since the late 1990s. |
| **Give-back after inclusion?** | ![Wrong direction](https://img.shields.io/badge/Give--back-Wrong_direction-8b949e?style=flat-square) | Post-effective 3m return: +10.9% (HAC *t* = +2.84). New S&P 500 members continue rising — the reversal trade is absent and the short-inclusion thesis is firmly wrong in this sample. |

> **In one sentence:** the classic S&P 500 inclusion pop documented by Shleifer (1986) is effectively dead in the 2001–2024 sample — ex-TSLA the announce-to-effective trade returns +0.02% gross — and the supposed give-back reversal does not exist; post-inclusion stocks keep rallying.

## What we tested

The S&P 500 committee announces new index members a few days before they join. The classic
finding (Shleifer 1986, Harris & Gurel 1986) is that the stock pops +2–5% from announcement
to effective date as passive index funds must buy regardless of price. Arbitrageurs front-run
this: buy on announcement, sell on effective date. A secondary claim is that the pop reverses
once index buying pressure ends.

We hardcode a table of 46 notable additions (2001–2024) with `yfinance` price data and compute:
(a) the **pop**: close-to-close return from announcement day to effective day; and
(b) the **give-back**: buy-and-hold return at 1m and 3m *after* the effective date.

**Critical TSLA caveat:** TSLA's 2020 addition (announcement 2020-11-16, effective 2020-12-21)
is an extreme outlier — a +59% pop over 25 business days driven by ~$80 billion of forced index
buying. It inflates the full-sample mean from +0.02% to +1.31%. We report both.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why the pop faded, the TSLA caveat, why "sell when it joins the index" is backwards |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-event table, HAC t-stats, period breakdown, positive control, cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`index_inclusion/`](index_inclusion/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
