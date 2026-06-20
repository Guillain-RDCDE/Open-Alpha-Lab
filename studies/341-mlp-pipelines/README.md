# Study 341 — MLP-Pipelines ⛽

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real, robust exposure under the "income"? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — and it's the *opposite* of the pitch. AMLP's beta to the **energy sector** (XLE) is **+0.74** (HAC *t* = **+5.26**, CI [0.49, 0.97], R² 0.64); the 7.8% distribution is **100% return of capital** (NAV fell −2.3%/yr). |
| **Tradability** — is it the bond-like income sleeve it's sold as? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. CAGR **5.3%** vs SPY's **15.4%**, Sharpe **0.34** vs **1.06**, and a **−73% drawdown** — *deeper than the energy sector itself* and 3× the market. A worse bond *and* a worse stock. |
| **"Fat-yield free lunch"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The yield isn't free, isn't fee-based, and isn't bond-like — it's **leverage on oil with a return-of-capital coupon**. |

> **In one sentence:** the AMLP "7–8% bond-like income" is a leveraged bet on the energy complex (energy beta 0.74 at *t* > 5) that finances its headline yield by liquidating principal (100% return of capital) — the exposure is real, the safety is a costume, and the free lunch is a mirage.

## What we tested

The pitch, steelmanned: *"Pipeline MLPs are the toll-roads of energy — fee-based cash flows that don't depend on the oil price, passed through as a high, stable 7–8% distribution. Bond-like income with a touch of inflation protection."* We take the category bellwether **AMLP** (plus peers **MLPA** and the leveraged **AMZA**) apart on the real Yahoo monthly tape (2010–2026, total return): regress each on the energy sector (**XLE**) with a HAC *t* on the beta, race it against **SPY** and **XLE** total return (Sharpe, drawdown, capture), and decompose the distribution into NAV vs **return of capital**. The offline control is a synthetic MLP world with a `beta` / `dist` / `nav_drift` knob that plants either the yield trap or a null. **Distinct from [Study 337 (Covered-Call-ETF)](../../337-covered-call-etf/)** (option-overlay income illusion) and **[Study 57 (Yield-Trap)](../../57-yield-trap/)** (high-dividend stocks): this is the *MLP/midstream* asset-class identity test, with **energy beta** as the third axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | where the "yield" really comes from, the energy costume coming off, the crash |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the energy-beta regression (HAC *t* + bootstrap CI), the return-of-capital decomposition, the SPY/XLE race, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2010–2026, TR fp `328f05124cbf`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [mlp_pipelines/data.py](mlp_pipelines/data.py).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mlp_pipelines/`](mlp_pipelines/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
