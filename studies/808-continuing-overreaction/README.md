# Study 808 — Continuing Overreaction 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a weighted signed-momentum score predict returns (Byun-Lim-Yun)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "continuing overreaction" premium **fails to appear** on 50 liquid US mega-caps. The specified long-high-CO / short-low-CO spread is **+15.46 bps/month** (Newey-West *t* = **+0.58**): the *right sign* (a whisper of continuation) but **statistically indistinguishable from zero** — a monthly Sharpe of 0.14, high-CO and low-CO books within a rounding error (+152.9 vs +137.5 bps, Welch *t* = +0.31). It sits ≈**+0.68σ** in a 1,000-permutation placebo (p = 0.261) and is flat in both eras (*t* = +0.26 / +0.54). A 20-seed synthetic control recovers a *planted* continuation cleanly (*t* = +8.61, fires on **1/20** nulls) — so this is a genuine null, not a broken sort. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The book is insignificant gross; the monthly round-trip friction then eats it to **+9.29 bps/month** (net *t* = +0.33) at 1 bp one-way and **+1.29 bps** (*t* = +0.05) at 5 bps. No costed version of the book pays — a Mirage. |

> **In one sentence:** the celebrated continuing-overreaction signal — a recency-weighted
> count of monthly-return signs that is supposed to sharpen momentum — **does not survive on
> liquid US mega-caps**; here the spread points the claimed way but is statistical noise (NW
> *t* = +0.58), and costs finish it off, so the honest read is **claimed signal absent,
> paycheck a mirage**.

## What we tested

Byun, Lim & Yun (2016), **"Continuing Overreaction and Stock Return Predictability"**: for each
stock build a **weighted signed-momentum** score — a recency-weighted sum of the **signs** of
its recent monthly returns (recent months weighted most, the `w_j = (n−j)` shape) — and it is
supposed to predict the cross-section *positively* (a persistent up-streak keeps running). We
take the self-contained monthly version on a **liquid 50-name US cross-section (yfinance daily
OHLC, total-return, 2010-01-04 → 2026-06-30)**: each name's **continuing-overreaction score**
over its **trailing 12 monthly returns, skipping the most recent month** (`CO ∈ [−1, +1]`,
signs only), sorted point-in-time (score known through month `i−2`, month `i−1` skipped, hold
month `i`, zero look-ahead), with a Newey-West *t* on the monthly spread, a 1,000-permutation
placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive
control. The universe is a **current-membership** survivor set (`quantlab.universe` opt-in
guard) — named on the **Signal** axis. **Dedup:** [507-cross-sectional-momentum](../507-cross-sectional-momentum/)
sorts on the **cumulative magnitude** of the trailing return, not the weighted **signs** of the
monthly steps; [508-momentum-crashes](../508-momentum-crashes/) studies the factor's
**conditional crash risk**, not a signed-consistency signal; [196-long-term-reversal](../196-long-term-reversal/)
is the 3-5-year **reversal** (opposite horizon and sign); [510-frog-in-the-pan](../510-frog-in-the-pan/)
uses information **discreteness** (path smoothness), not a count of monthly signs. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a consistent recent up-streak *should* keep running — and why on mega-caps it is just noise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`continuing_overreaction/`](continuing_overreaction/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
