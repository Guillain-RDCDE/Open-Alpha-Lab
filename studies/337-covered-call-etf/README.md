# Study 337 — Covered-Call-ETF 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the under-performance / income-illusion statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | QYLD, XYLD, RYLD **and** JEPI each trail **SPY total return** with HAC *t* of **−3.21 / −3.84 / −2.90 / −2.08** and a bootstrap CI **wholly below zero**; the distribution is **83–100% return of capital**. |
| **Tradability** — is it a better way to own equities? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Lower Sharpe and **capped upside** (keeps ~60% of up months, ~62–76% of down months); the high "yield" is mostly your own NAV being liquidated, not income. |
| **"Yield without giving up returns"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | You give up the returns *to manufacture* the yield. JEPQ's brief out-performance is a 4-year QQQ-bull artefact (up-capture 0.93, *t* = −0.15), not a working design. |

> **In one sentence:** buy-write "income" ETFs pay a fat 8–12% distribution that — for QYLD and RYLD — is **100% your own capital handed back** while the NAV erodes, and every fund with a real track record trails SPY total return at a lower Sharpe, so the marketed "yield without giving up returns" is the exact thing the data rejects.

## What we tested

The pitch behind the buy-write ETF boom — JEPMorgan's **JEPI/JEPQ**, Global X's **QYLD/XYLD/RYLD** — is *"high monthly income without giving up your equity returns."* We take that literally: each fund is raced against **SPY total return** (distributions reinvested) on its full monthly history, with an autocorrelation-robust (Newey-West) *t*-stat and a block-bootstrap CI on the return spread, and we **decompose the headline distribution** into a price (NAV) leg and a *return-of-capital* share to test whether the "income" is real yield or self-financed NAV erosion. A deterministic synthetic buy-write replicator (hold the index, write a call, collect the premium) is the offline machinery control.

> This is the **income-illusion + JEPI-generation** angle. The sister study [62 — Premium-Seller](../../62-premium-seller/) races QYLD against *its own underlying* (QQQ) on upside/downside capture; here the object is the distribution-vs-NAV decomposition across the newer JEPI/JEPQ/XYLD/RYLD funds vs **SPY**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a 12% "yield" can be 100% your own money, and what capped upside costs you |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the SPY-total-return race with HAC *t* + bootstrap CI, the return-of-capital decomposition, up/down capture, the synthetic replicator |

The fingerprinted real-data run (JEPI/JEPQ/QYLD/XYLD/RYLD vs SPY, as-of 2026-05-31) is in [docs/results.md](docs/results.md); reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`covered_call_etf/`](covered_call_etf/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
