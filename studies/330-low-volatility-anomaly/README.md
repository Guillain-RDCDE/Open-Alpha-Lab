# Study 330 — Low-Volatility-Anomaly 🐌

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do calm stocks beat wild ones risk-adjusted? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The *ranking* holds — SPLV's Sharpe **0.85** beats SPHB's **0.66** — but the tradable expression doesn't certify: a beta-neutral long-SPLV/short-SPHB book earns **+3.5%/yr at HAC t = 1.43** (bootstrap CI **[−0.9%, +8.1%]**), short of the *t* ≥ 2 bar. |
| **Tradability** — does the spread pay? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The *naive* dollar-neutral trade **loses −6.4%/yr** (shorting the decade's beta winner); beta-neutralised it's +3.5%/yr gross → **+2.5%/yr (t 1.02)** after 5 bps/leg + 50 bps borrow. The salvageable piece is a long-only defensive tilt, not a money machine. |
| **"Boring beats exciting, risk-adjusted?"** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Yes, on the question as asked: SPLV took **<½ the vol** (11.7% vs 24.8%) and a shallower drawdown (−21% vs −37%) yet earned the higher Sharpe. The calm fund really is the better risk-adjusted hold. |

> **In one sentence:** boring really does beat exciting *risk-adjusted* — the low-vol ETF (SPLV) out-Sharpes the high-beta ETF (SPHB) 0.85-to-0.66 on under half the risk — but the trade that would monetise it (long calm, short wild) either loses to the high-beta leg's beta or, once you hedge that out, earns a thin +3.5%/yr the tape can't tell from zero (*t* 1.43).

## What we tested

The **low-volatility anomaly** (Baker–Bradley–Wurgler 2011; Frazzini–Pedersen 2014): calm stocks
out-earn wild ones *per unit of risk*, because leverage-constrained investors overpay for
high-beta excitement and flatten the security-market line. We test its most literal, retail-tradable
form — the **S&P 500 Low Volatility ETF (SPLV)** raced head-to-head against its opposite, the
**S&P 500 High Beta ETF (SPHB)** — leg by leg (Sharpe, vol, drawdown) and as a self-financing
long-low/short-high book, with HAC *t*, a block-bootstrap CI, costs and short borrow, and the
structural beta gap hedged out. The offline control is a synthetic SPLV/SPHB/SPY world with a
dial-able low-vol edge (and a null). *Distinct from [18 Dull-Roar](../18-dull-roar/) (S&P 500
cross-section), [58 Bunker](../58-bunker/) (USMV vs market) and [54 Static](../54-static/) (idio-vol).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the calm fund is the better ride, and why the obvious "short the wild one" trade backfires |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the leg cards, the raw-vs-beta-neutral spread with HAC *t* and bootstrap CI, the alpha-vs-beta split, costs + borrow, and the synthetic positive control |

The fingerprinted real-data run (SPLV/SPHB/SPY, 2011–2026, fp `a5ce034427ab`) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the synthetic world in
[`low_volatility_anomaly/data.py`](low_volatility_anomaly/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
