# Study 792 — Cross-sectional commodity momentum 🛢️🥇

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does buying past-year commodity winners and shorting losers pay? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The 12-1 L/S sort earns a big-looking **+84 bps/mo (~+10%/yr, Sharpe 0.41)** on a 13-ETF basket (2009-2026), and it clearly beats a random-rank sort (placebo **p = 0.025**) — but the autocorrelation-robust statistic is **HAC *t* = +1.66, below the *t* ≥ 2 bar**. Miffre-Rallis (2007) reports it as real; **this free tape can't certify it**, and the whole certifiable premium is pre-2019 (*t* = 2.27 early vs *t* = 0.43 lately). **Survivorship named:** current-membership ETF basket → the numbers are an upper bound. |
| **Tradability** — can you get paid for it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Monthly turnover is low (0.74× NAV), so **costs barely dent it** — net **~+8.7%/yr at 10 bps/side** (short leg paying 50 bps/yr borrow). But the net **HAC *t* = 1.42-1.50** is uncertified, the edge is concentrated pre-2019, the universe is 13 survivor ETFs with thin early breadth, and the short leg leans on a few big moves (the 2014-15 crude collapse). Survives on paper; not investable at scale. |
| **Did the premium survive into the 2020s?** | ![Not supported](https://img.shields.io/badge/Recent_half-Not_supported-8b949e?style=flat-square) | The post-2019 half is statistically flat (HAC *t* = 0.43, Sharpe 0.17); the whole certifiable edge lives in 2009-2018 (*t* = 2.27). The difference test is underpowered (Welch *t* = −0.78), so the honest read is **faded**, not *provably dead* — but the recent tape offers no tradable premium. |

> **In one sentence:** the classic Miffre-Rallis commodity-momentum sort — long past-year
> winners, short losers, monthly — backtests at a juicy **+10%/yr (Sharpe 0.41)** on a
> free single-commodity-ETF basket and survives costs almost untouched, yet the
> autocorrelation-robust ***t* is only 1.66**, the entire premium is pre-2019, and the
> basket is a 13-name survivor set — so the honest read is **a real-looking premium this
> tape can't certify, and can't sell you lately**.

## What we tested

Miffre & Rallis (2007), stated the way its believers state it: *"rank commodities by their
trailing 12-month return, buy the top third and short the bottom third, rebalance
monthly — a standalone momentum premium."* We rebuild the exact **12-1** sort
(Jegadeesh-Titman: rank on months *t*−11 … *t*−1, skipping the most recent month) on the
**investable** proxy — **13 liquid single-commodity ETFs** (GLD/SLV/PPLT/PALL/CPER,
USO/UNG/UGA, CORN/WEAT/SOYB/CANE/DBA), total-return closes, **2009-2026**, 207 monthly
rebalances — long/short equal-weight, dollar-neutral, with **one documented execution lag**
(signal at month-end *t*, position earns month *t*+1). The Signal axis puts an HAC/Newey-West
*t* on the monthly L/S mean, a 40-seed random-rank placebo, a leg decomposition and a
pre/post-2019 difference test; the timer charges one-way cost × traded notional with the
short leg paying borrow; a deterministic synthetic panel with a plantable momentum edge
proves the machinery (null unbiased, planted edge fires at *t* = 13). **Survivorship is
named on the Signal axis** — the ETF basket is current membership, biasing the premium
upward. **Dedup:** [638-value-momentum-everywhere](../638-value-momentum-everywhere/) is the
*mixed multi-asset* value+momentum combo (commodities are one sleeve of eight);
[507-cross-sectional-momentum](../507-cross-sectional-momentum/) is the same 12-1 sort on
**equities**; and the single-commodity **seasonality** studies (226, 307, 648-651) test
*calendar* effects within one commodity — none test the pure cross-sectional commodity sort.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "buy the strong commodities, short the weak ones" means, why it backtests so well, why the *t*-stat and the last decade quietly undercut the headline |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 12-1 L/S with HAC *t*, the leg decomposition, the 40-seed random-rank placebo, the pre/post-2019 difference test, the cost-and-borrow sweep, and a planted-effect synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`commodity_momentum/`](commodity_momentum/). Total-return ETF closes; the basket
is current membership (survivorship named on the Signal axis — the premium is an upper
bound). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
