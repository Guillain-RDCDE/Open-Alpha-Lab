# Study 763 — Puell-Multiple ⛏️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Puell Multiple forecast BTC tops & bottoms? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | `log(Puell)` doesn't forecast forward BTC returns: best HAC *t* = **0.67** (30d), sign not even stably contrarian across horizons, R² ≈ 0; horse-race *t* = **−1.76**. The "buy the bottom" band **underperforms a neutral day** (+2.0% vs +24.0% at 90d, placebo *p* = 1.00). The one striking number — the top band's **−44%/90d** — is a *single* 2017-18 blow-off top (22 clustered days, n_eff ≈ 1); its Welch *t* = −19 is autocorrelation fake-precision. No robust *t* ≥ 2 on the tape. |
| **Tradability** — can you time the market with it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The buy-low/sell-high timer is **99.4% buy-and-hold** — Puell only clears 4 at the very peak — and it **trails** HODL net of costs at every threshold (**+24,841% vs +25,461%**, −0.97%/yr, HAC *t* = −0.21). Any CAGR is just long exposure to a 100×+ survivor. |
| **Times tops & bottoms?** | ![Busted](https://img.shields.io/badge/Times_tops_%26_bottoms%3F-Busted-8b949e?style=flat-square) | The bottom-call is empty (worse than a random day); the top-call is one historical episode that doesn't survive as a trade. |

> **In one sentence:** the Puell Multiple reconstructs *exactly* from BTC price + the halving
> schedule — and once you do, its "sell the top" band turns out to be a **single** 2017 blow-off
> dressed as a *t* = −19, its "buy the bottom" band does **worse** than a random day, and the
> buy-low/sell-high timer it implies is 99% buy-and-hold that quietly *loses* to just holding —
> a textbook **None / Mirage**.

## What we tested

David Puell's **[Puell Multiple](https://www.lookintobitcoin.com/charts/puell-multiple/)** (2019):
daily Bitcoin miner *issuance value* (new coins × price) divided by its trailing 365-day average,
read as a contrarian cycle timer — **> 4 = overheated top (sell)**, **< 0.5 = undervalued bottom
(buy)**. Because issuance is a *known* step function of the halving schedule (and the blocks/day
constant cancels in the ratio), we **reconstruct the canonical issuance-only metric exactly** from
the yfinance BTC-USD daily tape — no digitised proxy — then ask three honest questions: (1) does
`log(Puell)` predict forward BTC returns in a HAC regression (and add anything to price momentum);
(2) do the top/bottom bands actually mark tops and bottoms, once a random-date placebo is charged
against their tiny, clustered effective sample; (3) does a buy-low/sell-high timer beat
**buy-and-hold** net of costs. A deterministic synthetic control confirms the engine finds a
planted contrarian link when one exists. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Puell Multiple is, why it's really a 365-day price ratio with a halving imprint, why "sell the top" rests on one 2017 blow-off, and why the timer just quietly loses to holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the exact reconstruction, the HAC forward-return regression + momentum horse race, the band event study with a random-date placebo, the timer-vs-HODL sweep, and a 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`puell_multiple/`](puell_multiple/). The Puell Multiple
is a faithful reconstruction from price + the halving schedule (issuance-only), not a proxy;
BTC-USD is a single-survivor asset, named on the Signal axis. **Not investment advice** — research
& education. See [LICENSE](../../LICENSE).*
