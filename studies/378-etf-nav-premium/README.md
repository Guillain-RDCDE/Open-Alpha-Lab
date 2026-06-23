# Study 378 — ETF-NAV-Premium ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a below-NAV discount come back? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The discount *partly* mean-reverts — pooled hedged harvest **+5 bp/5-day**, HAC *t* = **2.03** — but it **fails t ≥ 2 by 10 days** (t = 1.24), **reverses sign across ETFs** (HYG *t* = **−0.58**), and **weakens as the discount deepens** (t = 1.53 at z < −1.5). A mild, fragile wobble on a NAV **proxy**, not a robust edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gross harvest (~5 bp) is **smaller than one round-trip** on the ETF + hedge: **net-negative at a routine 2 bp/leg half-spread**. And the deepest discounts **cluster in stress** (~a quarter of each ETF's worst gaps fall in March-2020), exactly when the arbitrage breaks and you can't transact at NAV. |
| **"Free discount"?** | ![Busted](https://img.shields.io/badge/Free_discount%3F-Busted-8b949e?style=flat-square) | "Below NAV = on sale" is a **structural illusion**: most of the gap is the *basket* moving, what little reverts is basis points, costs erase it, and the biggest gaps appear when they're least harvestable (stale bond NAVs + frozen APs). |

> **In one sentence:** an ETF trading below NAV *does* see its discount partially close, but on three liquid bond/EM/muni ETFs (HYG, EMB, MUB) the *hedged* earn-back is only ~5 basis points, it's carried by a single ETF and dies by the 10-day horizon (t drops 2.03 → 1.24), and it is smaller than the bid-ask you'd pay on both legs — so the discount is real arithmetic about a fair-value proxy, not free money, and it widens precisely in the March-2020-style stress when you can least act on it.

## What we tested

True intraday iNAV / official NAV isn't on yfinance, so we **construct a transparent NAV proxy**: for each target ETF we fit a hedge basket (a sister credit ETF + a rate leg) and define the transient **premium/discount basis** as the ETF's detrended cumulative return *residual* to that fair value — a stationary, mean-reverting stand-in for the real prem/disc, labelled a proxy throughout. A **discount** fires when the basis z-score (rolling 252-day) drops below **−1**; we enter the ETF one day later and measure the **hedged harvest** — the residual earned back, net of the bond market's own move — over 5/10/20 days, with a Newey-West *t*, a placebo null, a cost sweep on both legs, and a stress-clustering check. A deterministic synthetic control with an *injected* per-day earn-back confirms the engine is faithful **and** that a near-zero true edge can't reach significance.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "trading below NAV" means, why most of the "discount" is the basket moving, and why a 5-bp gap can't beat the spread — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | NAV-proxy basis construction, hedged-harvest with a HAC *t* + placebo null, the per-ETF sign reversal, a cost/threshold/stress battery, and a faithful-engine / planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`etf_nav_premium/`](etf_nav_premium/). The NAV here is an explicit **proxy** (a hedge-basket fair value), not official iNAV/NAV. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
