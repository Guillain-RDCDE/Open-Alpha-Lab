# Study 791 — Advertising Brand Capital 📺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do heavy advertisers earn a "brand-capital" premium? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Long the heaviest advertisers / short the lightest (advertising ÷ sales, monthly tertiles) returns **−3.00%/yr, HAC *t* = −1.46** — the *wrong sign* for the claimed premium and inside the noise. It fails a label-shuffle placebo (13th percentile, **p = 0.20**), stays negative and sub-2 at every sort width (halves → quintiles), and **neither leg beats SPY** (heavy −1.47%/yr *t* = −0.62; light +1.54%/yr *t* = +0.65). The synthetic control recovers a planted +6%/yr premium at *t* = +2.5 and never fires on 20 nulls, so the flat real-tape result is a **true null**, not a broken pipeline. Literature says premium; this tape says none. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | You cannot allocate to a wrong-signed, insignificant spread. Turnover is trivial (~1.0% one-way/month — the signal moves once a year), but a long/short **pays borrow**, dragging the −3.00%/yr gross to **−4.00%/yr (*t* = −1.94)** net. There is no positive edge to erode in the first place. |

> **In one sentence:** the lovely idea that advertising builds an under-priced "brand capital"
> that heavy advertisers get paid for **does not appear** on a clean, point-in-time basket of
> the consumer firms that actually disclose their ad spend — the heavy advertisers (mostly
> defensive staples) *lagged* the light ones (big-box and off-price retail) by −3.00%/yr at
> HAC *t* = −1.46, the wrong sign, failing the placebo (*p* = 0.20) with neither leg beating
> the market: **no signal, nothing to trade** — the raw sign is just this decade's
> staples-vs-retail style weather.

## What we tested

The intangibles-mispricing thesis, in its advertising form (Belo-Lin-Vitorino 2014;
Chan-Lakonishok-Sougiannis 2001): *advertising builds a durable brand-capital stock that GAAP
never puts on the balance sheet, so the market under-prices heavy advertisers — buy them for a
return premium.* We take it literally: rank **46 US consumer large-caps that actually file an
advertising line** each month by **AdvertisingExpense / Sales** (SEC EDGAR companyconcept), go
long the heaviest tertile / short the lightest, hold one month, and race the spread against SPY,
a 300-draw label-shuffle placebo, a halves→quintiles robustness sweep, and a seeded synthetic
positive control — one execution lag, one-year reporting lag, costs one-way × NAV with the
short leg paying borrow. **Honest scope:** most firms omit the advertising line entirely
(it stopped being required after ASU 2014), so this is a *selected slice of disclosers*, not the
market — named on the Signal axis, alongside the current-membership survivorship caveat.
**Dedup:** siblings [525-r-and-d-intensity](../525-r-and-d-intensity/) (the **R&D** intangible
and its market-value-vs-sales denominator contrast), [526-intangible-value](../526-intangible-value/)
(an **intangible-adjusted book-to-market** value signal) and [400-patent-intensity](../400-patent-intensity/)
(the **patent** intangible) test the same intangible-mispricing *family* on different
intangibles; none sorts on the **advertising** line — the brand-capital proxy — which is this
study's own axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "big advertisers are under-priced" is such an appealing idea, who actually advertises (Etsy/Expedia at ~27% of sales down to Ross at 0.3%), and why the heavy spenders *lagged* rather than led |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC long-short and its two-legged decomposition, the label-shuffle placebo, the sort-fraction robustness sweep, the short-borrow cost check, and the 20-seed synthetic power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`advertising_brand/`](advertising_brand/). EDGAR advertising/sales × yfinance total
returns; survivorship & thin-disclosure coverage named on the Signal axis. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
