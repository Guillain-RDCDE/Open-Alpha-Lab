# Study 525 — R-And-D-Intensity 🔬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a CLS R&D/ME premium? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The high-R&D/ME tertile beats the low-R&D/ME tertile by **+4.61%/yr**, and the signal survives a label-shuffle placebo (**p = 0.005**, 99.8th pct). But on a clean 21-year tape it **fails t ≥ 2** (HAC **t = 1.69**), clears 2 *only* at the most-concentrated quintile (2.24), and the whole edge is the long leg (the short leg matches the market, t = −0.02). Published + placebo-real but insignificant ⇒ **Weak**, not Real. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Turnover cost is small (monthly rebalance, ~1.6% one-way/mo), but a long/short **pays short-borrow**: a 100 bps/yr borrow drags the spread to **+3.59%/yr at t = 1.32** — a net edge indistinguishable from zero. The only deployable object is a long-only value/growth tilt (+4.56%/yr vs SPY, t = 2.52) a style ETF gives you cheaper. |
| **Does R&D/ME beat R&D/sales?** | ![Confirmed](https://img.shields.io/badge/ME_vs_sales-Confirmed-8b949e?style=flat-square) | Scaling R&D by **market value** *does* out-rank scaling by **sales** (+4.61%/yr at t = 1.69 vs +3.99%/yr at t = 1.40) — exactly the Chan-Lakonishok-Sougiannis ordering. But the win is thin (≈0.6%/yr, both below t = 2): CLS's *ranking* holds while neither signal clears significance. |

> **In one sentence:** sort large-caps monthly by R&D-to-market-cap and the high-R&D/ME tertile beats the low-R&D/ME tertile by +4.61%/yr — the signal survives a label-shuffle placebo (p = 0.005) and, as Chan-Lakonishok-Sougiannis predict, *does* edge out the R&D/sales version — but at HAC **t = 1.69** the spread never clears the bar (only the tightest quintile reaches t = 2.24), the whole edge is the long leg (the short leg is just the market), and short-borrow drags the net spread to t = 1.32: a real value/growth tilt wearing an R&D costume, not a distinct tradable innovation alpha.

## What we tested

Chan, Lakonishok & Sougiannis (2001) report that firms with high **R&D-to-market-cap** (R&D / ME) earn higher subsequent returns — and crucially that the predictability lives in R&D scaled by *market value* (a mispricing/cheapness signal), **not** R&D scaled by *sales* (pure spending intensity, which [study 400](../400-patent-intensity/) found weak). From a fixed **40-name** large-cap field chosen *by sector* (not by returns) we pull R&D, revenue and shares from **SEC EDGAR**, form R&D/ME monthly (R&D from the last reported fiscal year ÷ price × shares), go **long the top tertile / short the bottom**, rebalance monthly with a **1-year reporting lag and a 1-month execution lag**, and over **21.3 years** (2005–2026, 220 holding months) race the long-short against SPY, against a **label-shuffle placebo**, and against the **R&D/sales** signal. We charge one-way costs **and short-borrow**, run a Newey-West HAC test, and confirm with a deterministic synthetic control whose `edge` knob plants — or doesn't — a true R&D/ME premium. Survivorship is named on the Signal axis (current-membership basket; the bias is largely common to both legs of the long/short).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the R&D-cheap, short the rest" is mostly "buy value/growth," why +4.6%/yr isn't an edge if it can't clear the luck bar, why the short leg is *just the market*, and why short-borrow finishes it off — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the monthly R&D/ME ranking with reporting + execution lags, the long-short and long-minus-SPY HAC t-stats, the label-shuffle placebo, the R&D/ME-vs-R&D/sales contrast, a fraction-robustness sweep, costs + borrow, and a synthetic planted-premium / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`r_and_d_intensity/`](r_and_d_intensity/). R&D/ME is the audited Chan-Lakonishok-Sougiannis signal (R&D ÷ market value); the 16 R&D-light names are floored to ~0 (economically correct). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
