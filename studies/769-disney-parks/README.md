# Study 769 — Does the Disney-parks crowd tell you when to buy DIS? 🏰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does release-lagged parks momentum lead DIS? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On DIS's return *in excess of SPY* — the return a real DIS tell would move — the lead-lag slope is **+0.00002, Newey-West *t* = +0.02** (a mechanical zero); the pricing-power tell is *t* = +0.07. The one statistic touching \|*t*\| ≥ 2 is a COVID-regime dummy on *absolute* DIS return (*t* = 2.04) that vanishes vs SPY (*t* = 1.80). |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The costed "hold DIS when parks momentum > 0, else SPY" rule nets **Sharpe 0.578** — above buy-and-hold DIS (0.503) but far below just holding **SPY (0.987)**. Three stale, un-scalable rotations, not a signal. |
| **Parks momentum leads DIS?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The Theme Index prints year-Y attendance ~July Y+1 — months *after* Disney's quarterly segment reports told the market the same thing. Stale by construction. |

> **In one sentence:** theme-park attendance is a real, cited series and a genuine consumer story — but by the time the industry attendance print is public (~six months into the following year) Disney has already reported three quarters of the same parks revenue, so the strictly-lagged signal carries **zero** DIS-specific excess-return information (*t* = +0.02), and a timing rule built on it trails a plain S&P index fund.

## What we tested

Retail and fan-finance commentary treats **theme-park attendance and Disney's relentless
ticket-price hikes as a leading tell for `DIS`** — packed parks and pricing power "must" mean
the stock is a buy, since Parks & Experiences is Disney's largest profit centre
([TEA/AECOM *Theme Index*](docs/references.md)). We steelman the strongest **strictly-lagged,
no-look-ahead** version: a small, clearly-cited, **approximate** annual attendance series and
the WDW ticket-price series (labelled proxies reconstructed from the public Theme Index),
released with the report's real ~mid-following-year lag, aligned to `DIS` and `SPY` (month-end
Adj Close via yfinance). We ask whether parks momentum *leads* DIS's excess return — via a
Newey-West lead-lag *t*, a regime split, and a costed timing backtest — and whether any of it
survives being benchmarked against simply owning the market. (Same labelled-proxy discipline as
[Study 358 — Watch Index](../../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the parks are packed, buy DIS" feels true, the six-month lag that kills it, and the chart where an index fund wins — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Newey-West lead-lag on DIS *excess* return, the regime-dummy confound, the costed rotation vs SPY, and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`disney_parks/`](disney_parks/). Attendance & ticket-price series are **hardcoded, cited, approximate proxies** — not live feeds; released with the Theme Index's real lag so there is no look-ahead. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
