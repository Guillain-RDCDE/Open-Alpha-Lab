# Study 878 — Economic Policy Uncertainty ❓📰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does high policy uncertainty predict higher forward vol *and* higher forward returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The **risk-premium leg fails**: forward-SPY-return HAC *t* = **+0.65 … +1.24** across 1–12m (R² ≈ 0.01), a block-shuffle placebo p of **0.258**, and the lone post-2009 significance (*t* = +4.36) is **absent pre-2009** (*t* = +0.09) — a single-era recovery-drift artefact. The **vol leg** is strongly significant (*t* ≈ **+10**) but **mechanical**: the VIX proxy *is* implied vol, so "it predicts realized vol" is a near-tautology and a *coincident* stress reading, not a forward edge. Uncertainty is a thermometer, not a crystal ball. *Provenance: signal is a labelled **VIX proxy** — the real Baker-Bloom-Davis newspaper EPU feed was **network-unreachable** in-environment; named on the Signal axis.* |
| **Tradability** — can you get paid for leaning into it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No uncertainty-timed SPY rule beats buy-and-hold: **Sharpe 0.49** (lean-in) / **0.57** (de-risk) vs **0.77** for simply holding the index. Turnover is low, so it loses on *signal*, not costs. |

> **In one sentence:** high policy uncertainty **coincides** with volatility but does **not
> forecast** the returns you'd be paid for bearing it — the vol "prediction" is a near-mechanical
> restatement of implied vol, the return premium is a single-era mirage, and no timing rule
> beats buy-and-hold, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Baker, Bloom & Davis (2016), **"Measuring Economic Policy Uncertainty"**: a newspaper-based
**EPU** index is sold on two stories — high uncertainty should precede **higher equity
volatility** (the vol story) and **higher forward returns** as compensation (the risk-premium
story). We test both with **predictive regressions** of the H-month-ahead SPY **return** and
**realized vol** on the uncertainty **level** and **change** (Newey-West HAC *t*, R², a
one-month execution lag, a block-shuffle placebo, a two-era cut, a costed long/flat timer, and
a seeded synthetic positive control), on **401 aligned months, 1993–2026**.

**Data honesty — the signal is a labelled VIX proxy.** The intended series is the free
Baker-Bloom-Davis newspaper EPU (`policyuncertainty.com` / FRED `USEPUINDXM`); `data.fetch_epu`
tries it (4 retries, real UA). That feed was **network-unreachable** from the build
environment, so — rather than fabricate a series — `load_uncertainty()` falls back to a
**market-based VIX proxy** (real yfinance tape), labelled `vix_proxy` everywhere, exactly as
[387-economic-surprise-index](../387-economic-surprise-index/) proxies the proprietary CESI.
VIX ≠ newspaper EPU (market-implied, not text-based), disclosed on the **Signal** axis.
**Dedup:** [567-uncertainty-word-count](../567-uncertainty-word-count/) counts "uncertainty"
in **single-firm 10-Ks** (firm-level text), not a macro market index;
[318-election-volatility](../318-election-volatility/) is an **election-date event window**, not
a continuous regression; [313-geopolitical-shock](../313-geopolitical-shock/) reacts to discrete
**shock events**, not a slow index; [255-fear-greed-index](../255-fear-greed-index/) is a
**sentiment** contrarian timer, not a policy-uncertainty forward-vol/return test. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an uncertainty index *feels* predictive — and why it's a coincident thermometer, not a crystal ball |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | both predictive regressions, the HAC *t*, the two-era cut, the block-shuffle placebo, the cost math, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`epu/`](epu/). Signal is a documented **VIX proxy** (the newspaper EPU feed was
unreachable in-environment); SPY/VIX real tape via yfinance, cached under `_cache/`.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
