# Study 848 — Costco Hot-Dog Index 🌭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is COST a distinctive inflation hedge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | COST's beta to inflation *surprises* (ΔYoY) is HAC *t* = **+0.13** (≈ zero), **no better** than a boring staples basket (COST−XLP gap *t* = **−0.27**), placebo *p* = **0.88**, and flat in both sub-eras (*t* = −0.05 / +0.36). No inflation-hedge mechanism. |
| **Tradability** — can you harvest the "hedge"? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | An inflation-timing overlay nets Sharpe **0.34** — it *loses* to just buy-holding COST (**0.46**): the "hedge" signal actively hurts. COST beating SPY (**0.40**) is one **hindsight-selected** mega-cap (excess-return *t* = 1.37), not a repeatable edge. |
| **Real-price erosion** — is the folk icon literally true? | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Yes — the $1.50 combo has lost **67.6%** of its real value since 1985 (it would cost **$4.64** to hold 1985 purchasing power). But that is an **arithmetic identity** of a frozen nominal price — a marketing fact, not a stock signal. |

> **In one sentence:** the $1.50 hot dog really has melted **−67.6%** in real terms and COST really did turn $10k into **$287k** — but "buy COST as an inflation hedge" is a `NONE` × `MIRAGE`: COST's inflation-surprise beta is a dead-zero **+0.13** (and *below* a plain staples fund), the timing overlay *underperforms* just holding the stock, and the whole case is a **survivorship-selected** winner we're admiring after the fact.

## What we tested

The folk "anti-inflation" icon: the Costco **$1.50 hot-dog-and-soda combo**, nominally frozen
since ~1985, and the leap that Costco's **pricing power / membership model** lets its stock
(**`COST`**) outrun CPI and the market "regardless" — so you should buy it as an inflation hedge.
We separate three claims against **real month-end total-return tape** (COST, SPY, and a
consumer-staples control **`XLP`**, yfinance) and the **real `CPIAUCSL` series** (BLS
`CUSR0000SA0`, fetched from the BLS public API and cached — one interpolated point, 2025-10):
(H₀) the **real-price
identity** — the frozen combo's purchasing-power cost decays as `1/CPI` (a mechanical erosion we
quantify); (H₁) the **total-return race** COST vs CPI vs SPY (descriptive, survivorship named);
and (H₂) the tradable **signal** — is COST's beta to **inflation surprises** robustly non-zero
and **distinct from staples**? A block-rotation placebo, a two-era cut, a costed inflation-timing
overlay and a 20-seed synthetic planted-beta control complete it. **Dedup:** distinct from
[215-big-mac-ppp](../215-big-mac-ppp/) (cross-country PPP FX, not one price vs CPI),
[725-eggflation](../725-eggflation/) (a commodity *spike* trade, not a *frozen*-price hedge),
[726-chicken-wing-index](../726-chicken-wing-index/) (a wing-price folklore trade) and
[266-misery-index](../266-misery-index/) (macro misery timing — shares only the cited-CPI pattern).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the $1.50-hot-dog story *feels* like proof, the real-price melt, the "it went up ≠ it hedges" trap, and the punchline that COST is no better than a soup fund at inflation — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the real-price identity, the total-return race, contemporaneous & predictive inflation-surprise betas with an XLP control, a block-rotation placebo, a two-era cut, a costed timer, and a 20-seed synthetic positive control |

The fingerprinted real-data run (COST + SPY + XLP + real CPIAUCSL, 2000–2026, fp `cbd3b6799d57`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); reproduce via
[examples/verify.py](examples/verify.py) (`--fetch` to download). The offline machinery proof
runs on the synthetic world in [`hotdog_index/data.py`](hotdog_index/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`hotdog_index/`](hotdog_index/) (with [`quantlab/`](../../quantlab/) for the repro stamp). CPI is the **real** FRED/BLS `CPIAUCSL` series (BLS `CUSR0000SA0`), fetched from the BLS public API and cached (2025-10 interpolated); COST is a **hindsight-selected** single name. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
