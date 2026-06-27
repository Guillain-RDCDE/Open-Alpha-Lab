# Study 504 — Coskewness 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-coskewness (crash-sensitive) names pay a premium? | ![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a surviving large-cap proxy the premium has the **right sign but is statistically absent**: the long-low / short-high coskewness book earns a gross **+1.3%/yr at HAC *t* = +0.40** (placebo *p* = 0.35) — a coin. Low-coskew Q1 (**+15.2%/yr**) barely beats high-coskew Q5 (**+13.9%/yr**) while carrying the **highest** vol of the five buckets. Flat across tertile/quintile/decile cuts (deciles reach only *t* = 0.62) and both sample halves. The published CRSP premium is small and dispersion-driven; on **survivors** (named on the axis) it compresses into the noise. |
| **Tradability** — does the spread pay? | ![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A **+1.3%/yr gross** edge is far too thin to deploy; net of 20 bps/leg + 50 bps/yr borrow on a full-turnover monthly book it is **−4.0%/yr (*t* = −1.24)** — the frictions swallow the whole (insignificant) premium. Nothing to allocate. |
| **"Coskewness premium"?** | ![Coskewness_premium%3F: Busted](https://img.shields.io/badge/Coskewness_premium%3F-Busted-8b949e?style=flat-square) | On liquid large-caps the crash-sensitive (low-coskew) tail did **not** pay a usable premium — it edged the insurance-like tail by a hair while running the **highest** volatility (worst Sharpe). The systematic-skew premium is a broad-universe, dispersion-rich effect that does not generalise to mega-cap survivors. |

> **In one sentence:** Harvey-Siddique's *coskewness* premium — buy the names that crash with the market (bad hedges, so they should pay extra), short the insurance-like names — is a real but small broad-universe risk factor, yet on a surviving S&P-100-style large-cap basket it **vanishes into the noise**: the long-low / short-high book earns a gross +1.3%/yr at HAC *t* = +0.40 (the right sign, a coin's significance), the low-coskew leg actually carries the *worst* Sharpe, and net of frictions it merely loses — so this is `NONE` on signal, `MIRAGE` to trade, and a `BUSTED` premium at large-cap scale, with a seed-robust synthetic control (mean *t* = +8.6 over 20 seeds at a planted edge) proving the engine *would* have caught a real one.

## What we tested

The **coskewness** risk factor (Harvey & Siddique, *Journal of Finance*, 2000): a stock's
*systematic* tail — how hard it falls when the market falls hard, i.e. its contribution to the
**market's** own skewness — is priced, so names with **low (negative) coskewness** are bad hedges
and should pay a **premium**. We measure coskewness the simplest honest way — each month, Harvey-
Siddique's **direct standardised coskewness** ``E[ε_i·ε_m²]/(√E[ε_i²]·E[ε_m²])`` of each name's
daily returns with the market (SPY) over the **trailing 12 months** — then sort the cross-section
and go **long the low-coskew quintile, short the high-coskew quintile**, rebalanced monthly with
one execution lag. The signal is *systematic* co-movement with the market's tail, **distinct**
from [Study 503 — Expected Idiosyncratic Skewness](../503-expected-idiosyncratic-skewness/) (the
skew of the market-model *residual* — the diversifiable, behavioural-lottery tail, the opposite
axis of the same regression). True coskewness pricing is a CRSP-universe object, so we run it on a
fixed **79-name S&P-100-style basket** and call it a **proxy** throughout — explicitly
survivorship-tilted, named on the Signal axis. We measure each quintile's next-month return, test
the long-short mean with a Newey-West (HAC) *t* and a sign-flip placebo null, charge one-way costs
× turnover plus a short borrow, and confirm the engine on a deterministic synthetic panel with a
*planted* coskewness premium — averaged over 20 seeds — which it recovers cleanly (and finds
nothing at edge = 0).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "stocks that crash with the market should pay you extra" is a real risk-premium idea, why it disappears on the big steady stocks you own, and what coskewness actually flags among survivors — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the monthly direct-coskewness quintile sort, the long-short spread with a Newey-West *t* + sign-flip placebo, the gross-vs-net (cost-artefact) split, robustness across cut granularity and sub-periods, and a seed-robust synthetic planted-edge control |

The fingerprinted real-data run (79-name basket + SPY, 2005–2026, fp `abdf2637ac09`) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the synthetic world in
[`coskewness/data.py`](coskewness/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine:
[`coskewness/`](coskewness/). The cross-section here is an explicit **large-cap proxy** (a 79-name
survivor basket), not the CRSP universe. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
