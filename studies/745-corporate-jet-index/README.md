# Study 745 — Corporate-Jet-Index ✈️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a jet-loving CEO forecast underperformance? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The *sign* matches Yermack — frugal basket CAGR **16.2%** vs flyers **12.2%**, and ex-mega-cap flyers lag the market **−3.8%/yr** (his −4%) — and **survivorship biases *against* the claim** (the delisted abusers are missing). But the tradable long/short is **HAC *t* = 0.47** (1.08 ex-mega-cap), **fails *t* ≥ 2**; the only significant number (+8.8%/yr α, *t* = 2.10) is a **−0.45-beta artifact**. Literature + right sign, sub-2 tape ⇒ weak. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net **+1.4%/yr** at *t* ≈ 0.5, and it's a low-beta / low-vol bet dressed as governance — **one Tesla flips the short basket**. Nothing to harvest you couldn't buy cheaper as an explicit low-vol factor. |
| **"Governance discount?"** | ![Misattributed](https://img.shields.io/badge/Governance_discount%3F-Misattributed-8b949e?style=flat-square) | What looks like a jet-red-flag premium is **betting-against-beta**: frugal basket = low-beta staples (β **0.87**), flyer basket = high-beta growth (β **1.31**). The jet is a *correlate* of the founder-growth profile, not an independent tradable signal. |

> **In one sentence:** Yermack's *Flights of Fancy* found jet-perk firms underperform ~4%/yr, and on a survivor tape the frugal basket really does out-compound the flyers by ~4pp — but the tradable long/short is statistical noise (HAC *t* = 0.47), the lone *t* > 2 is a −0.45-beta low-vol artifact rather than governance, and one Tesla flips the whole short basket, so it's real-as-a-sign, weak-as-a-signal, and a misattributed mirage to trade.

## What we tested

David Yermack's [*Flights of Fancy* (2006, JFE)](https://doi.org/10.1016/j.jfineco.2005.05.002)
found that firms first disclosing their CEO's **personal use of the company jet**
underperform, risk-adjusted, by ~**4%/year** — the corporate jet as a canary for entrenched,
value-destroying management. We steelman that as a monthly, equal-weight **long/short
characteristic sort**: long a hardcoded, cited basket of frugal-reputation large-caps, short
a basket with a documented CEO personal-aircraft red flag (Ellison, Musk, Zuckerberg,
Adelson, Wynn, Jeffries, Irani…), each flyer entering only *after* its perk is public. We
judge the spread with a **Newey-West (HAC)** *t* on the raw excess-of-market return and on a
**market-model alpha**, net of costs and short borrow — and name up front that the archetypal
jet abusers (Tyco, WorldCom, Enron, Chesapeake) *delisted*, so the survivor tape is biased
**against** the claim.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "short the jet-loving CEO" is a great story, how the frugal basket really did win — and why the winning flyer (Tesla) breaks the trade, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long/short HAC *t*, the market-model α that's really a −0.45-beta BAB artifact, the founder-growth + survivorship confounds, costs + borrow, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`corporate_jet_index/`](corporate_jet_index/). The perk table is hardcoded & transparent; the priced tape is **survivor-biased against the claim** (the delisted abusers can't enter it), named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
