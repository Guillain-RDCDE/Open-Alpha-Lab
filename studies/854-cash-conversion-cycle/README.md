# Study 854 — Cash Conversion Cycle 🔄

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a shortening CCC predict higher forward returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short (long the shortening-CCC third, short the bloating-CCC third) earns **−47.1 bps/mo (−5.65%/yr gross)** — *wrong-signed* (the falling-CCC leg **under**-earns the rising-CCC leg) and **statistically zero**: Newey-West *t* = **−1.03** (one-sample −1.37). It doesn't survive perturbation (staleness-120 NW *t* = **−1.45**, percentage-change variant **−0.07**), the pooled event drift is **flat and sign-flipping** (long-short *t* +0.40 → −0.27 → +0.09, placebo *p* 0.34-0.62), and the sign **flips across the 2018 break** (+0.84 → **−1.54**). No coherent, robust effect. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It is wrong-signed and insignificant **before** costs, and net of 20 bps + 100 bps borrow it is **−7.45%/yr**, NW *t* = **−1.36**, Sharpe **−0.45**. Nothing to trade. |
| **Does shortening the CCC precede a better margin?** | ![Faint](https://img.shields.io/badge/Precedes_margin%3F-Faint-8b949e?style=flat-square) | **Barely.** A shortening CCC is *correctly signed* against next-quarter gross margin (slope −0.00010) and the shortening third widens margin **+0.27 pp** more than the bloating third — but the correlation is a whisker (**−0.05**, R² 0.003). The "frees cash → out-earns" mechanism exists in direction, is negligible in size. |

> **In one sentence:** the textbook working-capital virtue — shorten your Cash Conversion Cycle
> (DSO + DIO − DPO) to free cash and out-earn — is, on a liquid large-cap tape, a **wrong-signed
> null on returns** (NW *t* = −1.03, sign flips across the 2018 break, sign-flipping event drift)
> and only a **faint, correctly-signed whisper on margin** (+0.27 pp, correlation −0.05): **a good
> operating metric that a public ratio on blue chips prices away entirely.**

## What we tested

The operations-finance staple, stated the way its believers state it: *"the cash conversion cycle is
how long your money is trapped in operations — shorten it (collect faster, hold less inventory,
stretch payables) and you free cash to out-earn; let it bloat and you're a working-capital drag, so
buy the CCC-shorteners and short the CCC-bloaters."* We take it literally on **30 large US filers
that carry real inventory and payables** (of a ~46-name basket; financials and pure-software
excluded, DIO/CCC is undefined for them) that report all five ingredients —
`AccountsReceivableNetCurrent`, `InventoryNet`, `AccountsPayableCurrent`, quarterly `Revenues` and
`CostOfRevenue` (COGS fallback `CostOfGoodsAndServicesSold`) — on EDGAR, 2009→2026, ranked
**point-in-time on the 10-Q/10-K filing date** (zero look-ahead), with CCC = DSO + DIO − DPO in days.
We split the claim: a pooled regression of *next-quarter gross-margin change* on the CCC change (the
"frees cash → out-earns" mechanism) and a monthly tercile **long-short** held one month forward (the
return claim), graded on an autocorrelation-robust **Newey-West *t***, cross-checked by a pooled
event drift + label-shuffle placebo, an era split at 2018, a percentage-change variant, and a
12-seed synthetic control. Costs are one-way × NAV × turnover with the short leg paying borrow.
**Coverage is thin and uneven** — the CCC needs five matched facts a quarter, so only 30 names
survive and the matched cross-section *shrinks* (≈16 → ≈12 names) as ASC-606 revenue-tag switches
break the YoY match — and we say so throughout. **Dedup:**
[153-net-operating-assets](../153-net-operating-assets/) ranks on the balance-sheet-bloat *level*
scaled by assets; [529-inventory-growth](../529-inventory-growth/) is the *inventory* leg alone;
[853-days-sales-outstanding](../853-days-sales-outstanding/) is the *receivables* leg alone;
[524-operating-leverage](../524-operating-leverage/) is a cost-*structure* signal. None ranks on the
**year-over-year change in the netted DSO + DIO − DPO cycle** itself — this study does. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why freeing cash from the cash cycle *sounds* like an edge, why the stocks lean the wrong way, and how a signal can be a faint truth about the business and pure noise about the stock |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the precedes-margin regression, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ccc_signal/`](ccc_signal/). EDGAR XBRL `companyconcept` (receivables, inventory, payables,
revenue, COGS) + yfinance adjusted closes; a **current-survivors** basket — survivorship named on the
Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
