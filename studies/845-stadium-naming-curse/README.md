# Study 845 — Stadium Naming-Rights Curse 🏟️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do sponsors underperform after buying a stadium's name? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | 1-yr sponsor BHAR vs SPY **−11.3%**, one-sample *t* = **−2.54** (NW −2.41), hit rate **75%**, placebo **p = 0.003** — real and correctly signed, but **fails robustness**: gone by 2 years (*t* = −1.82), a **post-2010-only** effect (pre-2010 *t* = −0.84), and below the bar once the two COVID-hit tail names are dropped (*t* = −1.99). n=28, survivorship-trimmed. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A short-sponsor/long-SPY overlay nets **+10.1%** at 1 year (*t* = 2.27) but dies at 2 years (*t* = 1.45), rides the same non-robust tail, and needs to short the **least-borrowable** names (Caesars, crypto-adjacent) in exactly the crises that pay it. ~1 bet/year. Not bankable. |
| **Curse — real or cherry-picked?** | ![Mixed](https://img.shields.io/badge/Curse-Mixed-dab617?style=flat-square) | More than the Enron/FTX anecdote — even the *surviving* sponsors lag — but not a robust law: one era, a fat left tail of crisis-timed cyclicals, on a sample that already excludes the actual bankruptcies. |

> **In one sentence:** across 28 listed sponsors that bought stadium naming rights, the sponsor underperforms the S&P 500 by a placebo-significant **−11.3%** in the following year (*t* = −2.54, *p* = 0.003, 75% hit) — genuinely more than the cherry-picked Enron/FTX legend — yet the effect evaporates by year two, lives entirely post-2010, and falls apart when the two COVID-gutted tail names are removed, so it stamps **Weak**, not Real, and its short-side trade is **Fragile**.

## What we tested

The folklore — **managerial hubris / peak-earnings signaling**: a firm splashing out on
a vanity naming deal is near a self-confident top and then underperforms, with Enron
Field (bankrupt 2001), the MCI Center (WorldCom), and the FTX / Crypto.com arenas as the
poster children. We hand-curate **34** deals with announcement dates and sponsor tickers,
flag the **5 untradable cautionary tales** (private or delisted-into-bankruptcy sponsors)
honestly and exclude them from the return test, and measure each listed sponsor's
**buy-and-hold abnormal return vs SPY** over the 1- and 2-year windows after its deal —
cross-sectionally, with a one-sample and Newey-West *t*, a Wilson hit rate, a **sub-era
split**, a random-entry-date placebo, and a costed short overlay. Survivorship is named
on the Signal axis (the actual bankruptcies have no tape and are excluded — so the
measured curse *understates* the worst cases). A deterministic synthetic tape with a
*planted* post-deal drift is the positive control. **As-of 2026-06-30.**

**Dedup.** Distinct from the closest cousins on the desk:
[160-skyscraper-curse](../160-skyscraper-curse/) (a *building*, and a macro/market-timing
signal, not a firm-level stock), [746-hq-relocation](../746-hq-relocation/) (a *capex/HQ*
decision, not a marketing spend on a building the firm doesn't own), and
[722-logo-rebrand](../722-logo-rebrand/) (a *rebrand* of the firm's own identity, not a
stadium's name) — same hubris/vanity family, different corporate action and event set.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "curse" *feels* true (Enron, FTX), what the 28 tradable sponsors actually did, and why "significant in year one" is not the same as "real" |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the BHAR event study, the random-entry placebo, the sub-era split and tail-jackknife that demote it to Weak, the costed short overlay, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`stadium_curse/`](stadium_curse/). The deal calendar is hand-curated from
public naming-rights records; SPY and the sponsor tickers are fetched via yfinance.
Survivorship is named (the delisted bankruptcies have no tape and are excluded), one
execution lag documented (announcement snapped to the next tradable close). **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
