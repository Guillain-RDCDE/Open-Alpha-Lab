# Study 876 — Industry-Relative MAX 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does industry-adjusting MAX sharpen the negative MAX→return relation? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | It sharpens the *knife* but there is no right-signed apple. On 50 liquid US mega-caps the MAX effect **inverts**: both the raw and the industry-relative long-Q1 / short-Q5 spreads are **significantly negative** (raw −104.8 bps/mo, NW *t* = −2.42; industry-relative **−89.7 bps/mo**, NW *t* = **−2.51**) — the lottery **high-MAX** names *out-earned* the boring low-MAX ones (2010–2026). The industry adjustment trims the magnitude and nudges the *t* slightly more extreme, but the sign stays **wrong** vs the claim; it sits ≈**2.75σ into the left tail** of a 20,000-draw sign-flip placebo, holds **only in the late era** (*t* = −1.26 / −2.24), and a 20-seed synthetic control recovers a *planted* effect and confirms the adjustment sharpens it (*t* +11.18 → +20.60). A significant **wrong-sign** result fails the claim. *Survivorship: current-membership mega-caps — a small-cap effect probed on the wrong universe; magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified long-low / short-high book loses money gross and net (**−95.9 bps/mo** at 1 bp one-way, −103.9 at 5 bps). The data-mined sign-flip (long the lottery names) is a late-era-only artefact that fails the sub-era cut — no honest paycheck in either direction. |

> **In one sentence:** stripping sector-wide volatility out of MAX gives a *cleaner*
> lottery-demand signal — but on liquid US mega-caps the MAX effect runs **backwards**
> (the lottery names out-earned, NW *t* = −2.51), the industry adjustment doesn't fix the
> sign, and no version of the book survives costs, so the honest read is **claimed signal
> absent, paycheck a mirage**.

## What we tested

Bali, Cakici & Whitelaw (2011), **"Maxing Out"**: sort stocks on **MAX** (the highest daily
return over the prior month); the lottery-like high-MAX names are over-priced and under-earn, so
a long low-MAX / short high-MAX book earns a *positive* spread. This study **refines** that: we
subtract each name's **sector-peer median MAX** to build the **industry-relative MAX**, isolating
idiosyncratic lottery demand from sector-wide volatility, and grade it **head-to-head** against
the raw sort. We take the self-contained monthly version on a **liquid 50-name US cross-section
across 8 GICS sectors (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: monthly MAX
and industry-relative MAX, quintile-sorted point-in-time (MAX at month-end `t` → hold month
`t+1`, one documented lag, zero look-ahead), with a Newey-West *t* on the monthly spread, a
20,000-draw sign-flip placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed
synthetic positive control (which also proves the adjustment *sharpens* a planted effect). The
universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) — named on
the **Signal** axis. **Dedup:** [365-lottery-max-effect](../365-lottery-max-effect/) is the
**raw** MAX parent this refines; [503-expected-idiosyncratic-skewness](../503-expected-idiosyncratic-skewness/)
uses **modelled** skewness, not the extreme MAX; [806-prospect-theory-value](../806-prospect-theory-value/)
scores the **whole** return distribution, not the industry-adjusted tail; and
[538-industry-relative-reversal](../538-industry-relative-reversal/) shares the industry-adjustment
mechanic but on the **reversal** characteristic, not MAX. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why splitting MAX into "sector weather" + "name-specific pop" *should* sharpen the lottery signal — and why on mega-caps the whole effect ran backwards |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the raw-vs-industry head-to-head, the Newey-West spread *t*, the quintile monotonicity card, the 20,000-draw placebo, the two-era cut, the cost math, and the synthetic sharpening proof |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`max_industry/`](max_industry/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
