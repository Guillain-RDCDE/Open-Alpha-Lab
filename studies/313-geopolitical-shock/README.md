# Study 313 — Geopolitical-Shock

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 10-session post-shock abnormal return is **+0.04%**, cross-event *t* = **+0.07**, the **51st percentile** of a random-date placebo; the bootstrap 95% CI **[−1.13%, +1.05%]** straddles zero. The lone +1-day relief bounce (*t* = +1.88) falls short of the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Buy the geopolitical dip" never clears net *t* = 2, fires **< 1×/yr**, and its only positive numbers are the **+10.8%/yr** equity drift it would have earned by just holding — not a shock edge. |
| **Do markets shrug shocks off in days?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | A small **−0.20%** dip on the shock day, recovered within ~1 session, then a flat abnormal-return path — textbook semi-strong efficiency to public geopolitical news. |

> **In one sentence:** across 28 wars, invasions and attacks the market takes a small same-day dip and then does nothing a coin-flip of random dates can't reproduce — the folk "shrug it off in days" is dead right, and that very efficiency is why there is no post-shock edge to trade.

## What we tested

Every crisis brings the same advice — *"markets sell off on geopolitical shocks, but the dip is shallow and short-lived, so buy the dip"* (the LPL/Vanguard post-shock tabulations, and a chorus of crisis-day commentary). We steelman it on a curated table of **28 major shocks** (Kuwait, 9/11, Iraq, Madrid, Crimea, MH17, Ukraine, Israel-Hamas, Iran), each mapped to its first tradable NYSE session. We run a constant-mean **event study** of SPY around each one (abnormal returns, the CAR path), falsify it against a **placebo distribution** of thousands of random non-event dates, bound it with a **block bootstrap**, and put a long-only "buy the dip" overlay through a cost sweep. A deterministic synthetic tape with a *planted* post-shock drift is the positive control. *(The data-driven GPR index would pick the events for us, but its feed is network-blocked, hence the hand-built table.)*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the average path around a shock, the random-date race, why "buy the dip" is buy-and-hold in disguise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | constant-mean CAR, the placebo distribution, block-bootstrap CI, buy-the-dip cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`geopolitical_shock/`](geopolitical_shock/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
