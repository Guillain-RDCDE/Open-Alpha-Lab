# Study 793 — Cross-sectional commodity value 🛢️📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do cheap (5-year-fallen) commodities beat expensive (risen) ones? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The AMP 5-year value L/S sort earns a nil **+8 bps/mo (~+1%/yr, Sharpe 0.05)** on the 13-ETF basket (2012-2026, 163 rebalances), with **HAC *t* = +0.17** — a flat zero — and it is **indistinguishable from a random-rank sort** (placebo **p = 0.53**; the median coin-flip rank did marginally *better*). The claim is real in AMP's broad spot-futures cross-section; **on this investable ETF tape it is simply absent.** **Survivorship + roll named:** current-membership basket and chronic-contango energy ETFs both bias the value read *upward*, so this is an upper bound. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net is **~0%/yr at any realistic cost** (+0.28%/yr @5bps, +0.09%/yr @10bps; net HAC *t* ≈ 0). Turnover is low (0.31× NAV — a 5-year signal barely trades), but that is irrelevant when the gross is already zero. **Nothing to capture.** |
| **Do the fallen ones actually bounce?** | ![Refuted](https://img.shields.io/badge/Reversal-Refuted-8b949e?style=flat-square) | The direction check *fails*: the **cheap (long) leg's excess-of-basket return is negative** (−9.88 bps/mo, *t* = −0.32). What little the spread made came only from the short (expensive) side, itself noise (*t* = 0.69). Over 2012-2026 the 5-year-fallen commodity ETFs did **not** out-bounce the risen ones — if anything the reverse. |

> **In one sentence:** the commodity **value** leg — buy the commodities whose price has
> fallen most over five years, short those that rose — is a real premium in the academic
> spot-futures cross-section, but on a free single-commodity-**ETF** basket it is a flat
> zero (HAC ***t* = 0.17**, no better than a coin-flip ranking), the cheap leg actually
> *underperforms*, and there is nothing left after costs — **`NONE` / `MIRAGE`**.

## What we tested

Asness-Moskowitz-Pedersen (2013), the commodity **value** leg, stated the way it's told:
*"a commodity is cheap when its price has fallen over the long horizon — rank the basket on
the ratio of the price ~5 years ago to today's price, buy the cheap (fallen) third, short
the expensive (risen) third, rebalance monthly."* We rebuild that exact **5-year value**
sort (log of the ~4.5–5.5-years-ago reference price minus the log current price, measured on
the **raw price level** so five years of carry can't masquerade as value) on the
**investable** proxy — the **same 13 liquid single-commodity ETFs** as sibling 792
(GLD/SLV/PPLT/PALL/CPER, USO/UNG/UGA, CORN/WEAT/SOYB/CANE/DBA), **2012-2026**, 163 monthly
rebalances — long/short equal-weight, dollar-neutral, with **one documented execution lag**
(signal at month-end *t*, position earns *t*+1). The Signal axis puts an HAC/Newey-West *t*
on the monthly L/S mean, a 40-seed random-rank placebo, a cheap-vs-expensive leg
decomposition and a sub-period difference test; the timer charges one-way cost × traded
notional with the short leg paying borrow; a deterministic synthetic price panel with a
plantable reversal edge proves the machinery (null unbiased, planted edge fires at *t* = 6).
**Survivorship + roll-contamination are named on the Signal axis** — both bias the value
read upward. **Dedup:** [792-commodity-momentum](../792-commodity-momentum/) is the
**opposite horizon** (past-year continuation) on the *same basket*;
[638-value-momentum-everywhere](../638-value-momentum-everywhere/) is the *mixed multi-asset*
value+momentum combo (commodities one blended sleeve of many) — this is the **commodity
value leg alone**, isolated. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "buy the cheap, fallen commodities" means, why it *should* work (long-horizon reversal), and why on real ETFs it just doesn't — the fallen ones didn't bounce |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 5-year value L/S with HAC *t*, the cheap-vs-expensive leg decomposition, the 40-seed random-rank placebo, the sub-period difference test, the cost-and-borrow sweep, and a planted-reversal synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`commodity_value/`](commodity_value/). Raw closes drive the value signal,
total-return closes the P&L; the basket is current membership (survivorship + roll named on
the Signal axis — the premium is an upper bound). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
