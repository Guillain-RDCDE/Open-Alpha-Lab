# Study 36 — Greenback 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the carry premium real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — UIP fails, so high-rate currencies earn a premium (LRV 2011). On the **real G10 tape (2001–2024)** the carry book earns Sharpe **+0.22** (high-minus-low **+3.0%/yr**) with the textbook negative-skew crash (skew **−0.70**, worst month **−10.6%**, Oct-2008). Thin but real (bootstrap 95% CI [−0.17, +0.69]). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | A *thin, crash-prone, cost-sensitive* edge: carry turns over slowly (0.55×/yr) but its **break-even is only ≈13 bp**; the combo's net Sharpe decays from **+0.17 (0bp) → +0.06 (10bp) → −0.10 (25bp)**. The crash never leaves (skew stays negative). `FRAGILE`, not `INVESTABLE`. |
| **Combo diversifies the crash?** | ![Partial](https://img.shields.io/badge/Combo_diversifies_the_crash%3F-Partial-dab617?style=flat-square) | **Mechanically yes, on Sharpe no.** On 2001–2024 the legs decorrelate (**+0.05**) so the combo *cushions* the steamroller (worst month **−10.6% → −6.4%**; in carry's worst 5 months **−7.6% → −3.0%**) — but **FX momentum lost money** this sample (Sharpe **−0.14**), so the combo **+0.06** can't beat carry **+0.22**. `PARTIAL`. |

> **In one sentence:** on the real 2001–2024 G10 tape the FX carry premium is real but thin and rent for standing in front of a steamroller (Sharpe **+0.22**, skew **−0.70**, worst month **−10.6%** in Oct-2008) — and the classic fix, the **carry⊕momentum combo**, *cushions* the crash exactly as designed (decorrelated legs **+0.05**; worst month **−10.6% → −6.4%**) but cannot lift the Sharpe because FX momentum itself decayed to **−0.14** over this sample, so the combo `PARTIAL`-ly delivers — it dulls the jump without ever pretending it's gone.

> ✅ **Real run · offline from cache · as-of 2024-01-31 · fingerprint `ef7450ae792e`.** OECD 3-month short rates + yfinance FX, a USD-funded monthly book over 2001–2024 (the rates' OECD-MEI source ended 2024-01, so the as-of is pinned there). Reproduce with `python examples/verify.py` (no network); the controlled machinery proof is `python examples/run_synthetic_demo.py`. Full numbers in the fingerprinted [docs/results.md](docs/results.md).

## What we tested

The desk's take on Kakushadze & Serur, *151 Trading Strategies* **§8.3 (dollar carry)** and **§8.4
(combining momentum and carry)**. The steelman: high-short-rate currencies out-earn low-rate ones (the
carry trade), a **dollar-carry** tilt (long/short USD vs a basket by the average rate gap) is a second
premium, and — the part believers actually trade — combining **carry with momentum** earns *more* than
either alone because the two pay at different times. This builds on [Study 27 (Steamroller)](../../27-steamroller/),
which already established the carry premium itself and that vol-targeting can't dodge its crash; Greenback
is specifically the **dollar-carry + carry⊕momentum combo** angle. We prove the machinery on a synthetic
currency panel (a baked carry premium, sticky risk-off crashes, and an independent *profitable* trend),
then run the **real G10 tape** — OECD 3-month short rates + yfinance FX, a USD-funded monthly book over
2001–2024 — and find carry is real-but-thin with its crash intact, **FX momentum decayed to negative** over
this sample, and so the combo *cushions* the steamroller (decorrelated legs, shallower worst months) but
cannot out-Sharpe carry while its momentum leg is losing.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: the rate-gap premium, the steamroller crash, and why pairing carry with momentum cushions it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the carry premium by rate bucket, the three sleeves, the combo's Sharpe uplift, the leg correlation, and the negative-skew crash |

The real run — every fingerprinted, as-of'd G10 number — is in [docs/results.md](docs/results.md) (as-of
2024-01-31, fingerprint `ef7450ae792e`); the **beat-7 worked complement** (the carry⊕momentum
diversification) is in [docs/extension.md](docs/extension.md). Reproduce the real tape offline via
[examples/verify.py](examples/verify.py) (reads the two `_cache/g10_*.parquet` files, no network); the
controlled machinery proof is [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
