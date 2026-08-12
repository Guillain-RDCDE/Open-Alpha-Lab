# Study 866 — Flight-to-Quality Beta 🛟

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the true defensives (high flight-to-quality beta) under-earn, as you'd pay for a hedge? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The pay-for-the-hedge spread (long low-FTQ / short high-FTQ) is **+0.52 %/mo** (+6.2 %/yr) with the **right sign** — the cheap risky names out-earned the expensive hedges — and sits ≈**+2.35 sd-units** into the right tail of a 1,000-permutation placebo (p = 0.008), so cross-sectionally it is real, not a lucky sort. **But** the conservative Newey-West *t* is only **+1.36** (one-sample +1.25): it fails \|*t*\| ≥ 2 and neither era clears significance (+0.73 / +1.18). A weak, not-robust premium on liquid mega-caps. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you harvest the premium? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Insignificant even **gross**; at 1 bp one-way net **+0.44 %/mo** (*t* = +1.05) and at 10 bps net **+0.08 %/mo** (*t* = +0.19) — trading costs plus 50 bps/yr borrow eat the thin spread and it never clears significance. |
| **Crash protection** (descriptive) — do the high-FTQ names really cushion crashes? | ![Confirmed](https://img.shields.io/badge/Crash_protection-Confirmed-8b949e?style=flat-square) | The *other* half of the claim is **strongly confirmed**: on the worst 5% of SPY days (mean −2.59%) the high-FTQ book lost only **−2.13 %/day** vs the low-FTQ book's **−3.25 %/day** — a **+1.13 %/day** cushion (Welch *t* = **+6.78**). FTQ beta is a genuine risk characteristic; it just is not a robustly *priced* one. A risk-management fact, not a return edge. |

> **In one sentence:** the flight-to-quality beta really does pick the stocks that cushion
> crashes (**+1.13 %/day** less loss on the worst market days, *t* = +6.78) — but the market does
> **not** robustly make you *pay* for that protection: the "hedges under-earn" premium is the
> right sign yet only **+0.52 %/mo at NW *t* = +1.36**, era-unstable, and gone after costs, so the
> honest read is **real hedge, weak premium, mirage of a paycheck**.

## What we tested

Some stocks are *true defensives* — they rise with long Treasuries on **risk-off** days. For
each name we estimate a **flight-to-quality beta** (`beta_ftq` = its beta to the **TLT** daily
return, computed **only on down-SPY days**) and test the two-sided CAPM-of-insurance claim: the
high-FTQ hedges should (a) earn a **lower** average return — you pay an insurance premium — yet
(b) deliver **real crash protection**. We take the daily version on a **liquid 50-name US
cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)** plus cached **TLT**
and **SPY** closes: each name's **trailing-252-day FTQ beta**, sorted point-in-time (signal at
month-end `t−1`, one lag, zero look-ahead) into a monthly long-low-FTQ / short-high-FTQ book,
with a Newey-West *t* on the spread, a 1,000-permutation placebo, a two-era cut, a crash-day
drawdown comparison, a costed timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the **Signal**
axis. **Dedup:** [332-downside-beta](../332-downside-beta/) conditions on the same down-market
regime but measures beta to the **equity market**, not to the **Treasury**;
[238-betting-against-beta](../238-betting-against-beta/) ranks **market** beta, not a cross-asset
sell-off loading; [246-defensive-sectors](../246-defensive-sectors/) uses a **GICS sector label**
rather than a *revealed* bond co-movement; [69-safe-haven](../69-safe-haven/) asks whether a whole
**asset class** hedges equities, not how each stock loads on that bid. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a stock that rallies with bonds in a panic is a hedge — and why the hedge is real but the premium is thin |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the crash-day cushion, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ftq_beta/`](ftq_beta/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound); TLT/SPY conditioners
cached alongside. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
