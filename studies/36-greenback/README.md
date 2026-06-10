# Study 36 — Greenback 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the carry premium (and its momentum complement) real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — uncovered interest-rate parity fails, so high-rate currencies earn a real premium (Lustig–Roussanov–Verdelhan 2011). Our synthetic control recovers it (high-minus-low **+5.3%/yr**, carry Sharpe **+1.18**) and the full-UIRP null is flat (**+0.8%/yr**); FX momentum is a second, decorrelated premium (**+1.53**). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Cheap to run (carry turns over slowly), but it carries the **steamroller**: carry skew **−1.55**, max drawdown **−60%** on the control. The carry⊕momentum combo *dulls* it (DD **−60% → −20%**) without removing the jump — `FRAGILE`, not `INVESTABLE`. |
| **Real-tape run?** | ![Pre-reg](https://img.shields.io/badge/Real--tape_run%3F-Pre--reg-8b949e?style=flat-square) | The carry signal needs **short rates from FRED**, whose download **times out** in this sandbox — so the real G10 run is pre-registered and PENDING one networked fetch. The verdict above is earned on the validated synthetic control + the literature. |

> **In one sentence:** the FX carry premium is real but rent for standing in front of a steamroller — and the classic fix is *not* a vol overlay (Study 27 showed that fails) but **diversification**: blending carry with its decorrelated complement, momentum, lifts the Sharpe above either leg (**+1.18 / +1.53 → +1.69**) and cushions the carry crash (drawdown **−60% → −20%**) without ever pretending the jump is gone.

> ⚠️ **Real run PENDING one fetch.** This study's carry signal needs **G10 short rates from FRED**, whose download is unavailable (times out) in this environment; FX spot from yfinance works, but without the rates there is no carry. Run `python examples/verify.py --fetch` where FRED is reachable to write the fingerprinted [docs/results.md](docs/results.md). The verdict above is earned on the fully-validated synthetic control and the literature; the offline core reproduces via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

## What we tested

The desk's take on Kakushadze & Serur, *151 Trading Strategies* **§8.3 (dollar carry)** and **§8.4
(combining momentum and carry)**. The steelman: high-short-rate currencies out-earn low-rate ones (the
carry trade), a **dollar-carry** tilt (long/short USD vs a basket by the average rate gap) is a second
premium, and — the part believers actually trade — combining **carry with momentum** earns *more* than
either alone because the two pay at different times. This builds on [Study 27 (Steamroller)](../../27-steamroller/),
which already established the carry premium itself and that vol-targeting can't dodge its crash; Greenback
is specifically the **dollar-carry + carry⊕momentum combo** angle. We prove the machinery on a synthetic
currency panel with a baked carry premium, sticky risk-off crashes, and an independent trend for the
momentum sleeve (and a full-UIRP null that earns nothing), then show the combo beats either leg and dulls
the steamroller.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: the rate-gap premium, the steamroller crash, and why pairing carry with momentum cushions it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the carry premium by rate bucket, the three sleeves, the combo's Sharpe uplift, the leg correlation, and the negative-skew crash |

The real run — every fingerprinted, as-of'd G10 number — is PENDING one `--fetch` in [docs/results.md](docs/results.md);
the **beat-7 worked complement** (the carry⊕momentum diversification) is in [docs/extension.md](docs/extension.md).
Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via
[examples/verify.py](examples/verify.py) (`--fetch` to download G10 rates from FRED + FX from yfinance).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
