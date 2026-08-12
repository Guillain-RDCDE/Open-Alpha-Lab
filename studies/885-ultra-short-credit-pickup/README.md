# Study 885 — Ultra-Short Credit Pickup 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do JPST/ICSH/MINT pay a real excess-of-bills pickup? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Real in the point estimate — the sleeve out-earns bills by **+49.8 bps/yr** at an excess-of-BIL Sharpe of **+0.62** (vs +0.11 for SHV, 0 for BIL), and on MINT's 16.6-yr tape **+77 bps/yr at HAC *t* = 2.82** with a bootstrap CI clear of zero — but it **fails the robustness bar**: the full-sleeve HAC *t* is only **+1.30** (serial NAV marks eat the naive Sharpe), the bootstrap Sharpe CI **crosses zero** ([−0.26, +2.21], 9.1% negative), and the whole edge is **pre-2018** (MINT *t* = 7.44 → 0.73 post-2018). Sign right everywhere (not None), robustness fails (not Real). *Young ETFs → short live history.* |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are trivial (buy-and-hold, fees inside the tape; at 1 bp net is **+47.8** of the +49.8 gross) — **not** a cost Mirage. Fragile because the carry is thin (~50 bps/yr), era-contingent (dead post-2018), and **not riskless**: MINT **−1.0% in 2022** while bills made +1.4%, and the sleeve drew down **−3 to −4%** vs bills' −0.2% in the COVID crunch. |

> **In one sentence:** ultra-short IG credit really does out-carry T-bills — the JPST/ICSH/MINT
> sleeve beats bills by ~50 bps/yr at a genuinely higher reward-per-risk, and on MINT's 16-year
> tape that pickup clears *t* = 2.8 — but on the young 3-ETF window it is under-powered
> (HAC *t* = 1.30, bootstrap Sharpe CI crosses zero), the significance lives entirely in the
> pre-2018 post-GFC era, and it is *not* the near-riskless carry it's sold as (−1% in 2022, −4% in
> March 2020), so the honest read is **Weak / Fragile**, not the free lunch the pitch implies.

## What we tested

The structural claim: ultra-short investment-grade credit ETFs (**JPST / ICSH / MINT**) hold ~AA-/A
short-maturity IG paper and are paid a **spread over T-bills** for a sliver of credit + duration
risk, so the credit sleeve should deliver a **higher excess-of-bills Sharpe** than bills (**BIL /
SHV**) with only marginally more drawdown — a near-riskless structural pickup. We run daily
total-return closes (yfinance, `auto_adjust=True`, 2009→2026; common 5-ETF sample 2017-05 → 2026-06,
2,289 days) and compute the excess-of-BIL Sharpe race, the HAC *t* on the credit-minus-bill pickup, a
circular-block bootstrap Sharpe CI, a pre/post-2019 sub-era cut (plus MINT's 16.6-year long history),
the drawdown + calendar-year table (2020 & 2022 stress named), and a costed buy-and-hold version. A
deterministic synthetic world with a planted-pickup knob proves the machinery is unbiased (never cited
as evidence). One documented lag: the sleeve is bought at the close and returns accrue from the next
day. **Dedup:** distinct from [340-bank-loans](../340-bank-loans/) (floating-rate *high-yield* loans,
a duration-for-credit swap), [338-preferred-stocks](../338-preferred-stocks/) (long-duration
subordinated preferred yield), [577-mbs-oas-signal](../577-mbs-oas-signal/) (MBS spread as a *timing*
signal), and [625-starting-yield](../625-starting-yield-bond-decade/) (a within-fund starting-yield
identity) — this is the *ultra-short IG credit over cash* pickup. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why parking cash one notch up the risk ladder pays a spread, what that spread actually earned, and why it is *not* the riskless free lunch it's sold as |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-BIL Sharpe race, the HAC pickup *t*, the block-bootstrap Sharpe CI, the sub-era cut + MINT's 16-year history, the stress windows, the cost math, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ultra_short/`](ultra_short/). Total-return, net-of-fee ETF tape (yfinance), as-of
2026-06-30, fingerprint `22e1cddb739d`. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
