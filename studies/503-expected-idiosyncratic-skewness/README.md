# Study 503 — Expected-Idiosyncratic-Skewness 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the skewed (lottery) tail underperform? | ![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a surviving large-cap proxy the claimed edge is **absent**: the long-low / short-high idio-skew book earns a gross **−1.0%/yr at HAC *t* = −0.41** (placebo *p* = 0.66) — a coin, if anything on the wrong side. Low-skew Q1 (**+15.6%/yr**) barely trails high-skew Q5 (**+16.6%/yr**). Flat across tertile/quintile/decile cuts and both sample halves. The published CRSP anomaly lives in small, retail-held lottery names; on **survivors** (named on the axis) the residual-skew sort carries no forward signal. |
| **Tradability** — does the spread pay? | ![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is **no positive gross edge**; the only significant number — net **−6.3%/yr (*t* = −2.54)** after 20 bps/leg + 50 bps/yr borrow — is a **cost artefact** on a near-zero gross spread, not a deployable skewness short. Nothing to allocate. |
| **"Skewed losers"?** | ![Skewed_losers%3F: Busted](https://img.shields.io/badge/Skewed_losers%3F-Busted-8b949e?style=flat-square) | On liquid large-caps the positively-skewed tail did **not** underperform — Q5 posted the **highest** quintile return. The skewness-preference morality tale is a small-cap / retail-lottery effect that does not generalise to the names large-cap investors hold. |

> **In one sentence:** Boyer-Mitton-Vorkink's *expected idiosyncratic skewness* — buy the symmetric names, avoid the lottery-skewed ones — is a real *small-cap / retail* anomaly, but on a surviving S&P-100-style large-cap basket it **vanishes**: the long-low / short-high book earns a gross −1.0%/yr at HAC *t* = −0.41 (a coin), the skewed Q5 tail actually edges out Q1 on return, and the only "significant" number is a losing **cost artefact** net of frictions — so this is `NONE` on signal, `MIRAGE` to trade, and a `BUSTED` folk tale at large-cap scale, with a seed-robust synthetic control (mean *t* = +8.6 over 20 seeds at a planted edge) proving the engine *would* have caught a real effect.

## What we tested

The **expected-idiosyncratic-skewness** anomaly (Boyer, Mitton & Vorkink, *Review of Financial
Studies*, 2010): stocks whose own-return distribution carries a fat idiosyncratic *right tail*
are lottery-like names investors over-pay for, so high expected idio-skew predicts **low** future
returns. We proxy *expected* idio-skew the simplest honest way — each month, regress each name's
daily returns on the market (SPY) over the **trailing 12 months** and take the **skewness of the
residuals** — then sort the cross-section and go **long the low-skew quintile, short the high-skew
quintile**, rebalanced monthly with one execution lag. The signal is the *shape* of the whole
residual distribution, **distinct** from [Study 365 — Lottery-MAX](../365-lottery-max-effect/)
(the single highest *daily* return, one point of the distribution) and from **coskewness** (a
*systematic*, market-co-movement tail; here the market is regressed out by construction). True
expected idio-skew is a CRSP-universe object, so we run it on a fixed **79-name S&P-100-style
basket** and call it a **proxy** throughout — explicitly survivorship-tilted, named on the Signal
axis. We measure each quintile's next-month return, test the long-short mean with a Newey-West
(HAC) *t* and a sign-flip placebo null, charge one-way costs × turnover plus a short borrow, and
confirm the engine on a deterministic synthetic panel with a *planted* skewness penalty — averaged
over 20 seeds — which it recovers cleanly (and finds nothing at edge = 0).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "skewed lottery stocks lose" is a real small-cap idea, why it disappears on the big stocks you own, and what a fat right tail actually flags among survivors — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the monthly residual-skew quintile sort, the long-short spread with a Newey-West *t* + sign-flip placebo, the gross-vs-net (cost-artefact) split, robustness across cut granularity and sub-periods, and a seed-robust synthetic planted-edge control |

The fingerprinted real-data run (79-name basket + SPY, 2005–2026, fp `abdf2637ac09`) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the synthetic world in
[`expected_idiosyncratic_skewness/data.py`](expected_idiosyncratic_skewness/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine:
[`expected_idiosyncratic_skewness/`](expected_idiosyncratic_skewness/). The cross-section here is
an explicit **large-cap proxy** (a 79-name survivor basket), not the CRSP universe. **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
