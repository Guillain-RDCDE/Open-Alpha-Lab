# Study 743 — Lucky-Number-8 🎴

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The lucky-DAY FXI−EEM abnormal return is **+0.39%**, *t* = **+2.27**, placebo *p* = **0.043** — a real nominal hit — but it is the *only* one of two horizons to clear (the week is *t* = +0.69), and **4 of 21** leave-one-out drops (including the 08/08/08 triple-8 itself) sink it below the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | 8/8 is calendar-known so the window *is* the trade — but ~39 bps/yr gross does not survive one round trip: net *t* = **+1.69** (5 bps) / **+1.10** (10 bps). No cut clears *t* ≥ 2 net. |
| **8-clustering in prices?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | China ADRs end in "8" **9.91%** of the time vs **9.76%** for a matched US-control basket (*z* = +0.85, *p* = 0.40) — no 8-excess, no 4-deficit. The clustering that *is* there is the universal round-number pull on **0** (China *z* = +6.86), not the lucky 8. |

> **In one sentence:** the lucky 8 leaves almost no mark on the US tape — no trailing-digit
> "8" clustering (just plain round-number clustering on 0), and only a tiny, fragile,
> uncostable one-day tick around 8/8 that leans on the 08/08/08 Olympics day itself.

## What we tested

A real academic literature finds Chinese investors cluster prices on the "lucky 8" (八
*ba* ≈ 發 *fa*, "prosper") and shun the "unlucky 4," and pay a premium for auspicious
listing digits and dates (Brown & Mitchell 2008; Hirshleifer, Jian & Zhang 2018;
Bhattacharya et al. 2018) — mostly measured on *mainland* order books. We ask whether that
fingerprint survives onto the **tradable US tape**: (1) do 15 US-listed Chinese ADRs show
an "8" excess in the trailing cent digit of their raw closing prices *relative to* 15
matched US-domestic large-caps (differencing out the universal round-number preference)?
and (2) does a China ETF (`FXI`) earn a superstition bump vs emerging markets (`EEM`)
around the doubly-auspicious **8/8** — one calendar-known event per year, 2005→2025, with
a one-sample *t*, a random-window placebo, a jackknife, and a costed trade.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the superstition, the digit histogram that clusters on the *wrong* number, the 08/08/08 story, the trade that costs die on |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the placebo, the jackknife fragility, the two-proportion digit contrast, the round-number confound, both synthetic controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`lucky_number_8/`](lucky_number_8/). The 8/8 calendar and the two clustering
baskets are hardcoded from cited sources; **adjustment named** — digit test on raw Close,
event returns on total-return Adj Close; **survivorship named** — current US-listed ADRs.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
