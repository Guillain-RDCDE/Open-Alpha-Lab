# Study 367 — CEF-Discount 🏷️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do deep-discount CEFs out-earn? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On a **transparent NAV proxy** (price vs each fund's benchmark), the widest-discount tercile beats the narrowest by **+6.6%/yr at Welch _t_ = 3.72** (block-bootstrap _p_ = 0.002), **market-neutral** (β ≈ 0), and **placebo-clean** (shuffling the discounts kills it). Clears the _t_ ≥ 2 bar — but it's a *proxy*, rests on a **surviving** CEF basket, and **fades to _t_ = 1.56 post-2010**, all named on this axis. |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Gross +6.6%/yr becomes **+1.8%/yr** after 20 bps × monthly two-leg turnover (break-even **27.5 bps**); the short leg needs **hard-to-borrow** illiquid CEFs; capacity is tiny (18 small funds); and the edge is **mostly pre-2010**. Survives on paper, not at scale. |
| **"Pays you to wait?"** | ![Confirmed](https://img.shields.io/badge/Pays_you_to_wait%3F-Confirmed-8b949e?style=flat-square) | The folklore is genuinely real on the proxy — deep-discount funds *do* out-earn as the discount mean-reverts — but the payment is **small, costs-sensitive, short-constrained and fading**: a real effect you can measure but can barely bank. |

> **In one sentence:** buying the widest-discount closed-end funds really does pay you to wait — on a transparent NAV proxy the wide-minus-narrow discount sort earns ~6.6%/yr market-neutral at _t_ = 3.72, and it's no artefact (shuffle the discounts and it vanishes) — but it's a proxy on a survivor basket, it fades to insignificance after 2010, and it needs hard-to-borrow shorts and tiny illiquid funds, so it's real-but-fragile, not a strategy you can scale.

## What we tested

True per-fund daily NAV isn't on yfinance, so we **build a transparent NAV proxy**: each closed-end fund is mapped to a published benchmark for its mandate (an S&P-500 fund → SPY, a utilities fund → XLU, an energy fund → XLE), and the proxy *discount* is the fund's price measured **relative to** that benchmark, demeaned per fund — labelled a proxy throughout. Over **31.5 years** (1995–2026, **18** long-listed equity CEFs, **377** monthly rebalances) we sort funds by proxy-discount each month and, with a 1-month execution lag, buy the **widest-discount** tercile and short the **narrowest**, then judge the spread with a Welch _t_, a circular block-bootstrap null, a discount-shuffle placebo, one-way costs × turnover, and a sub-period decay split. Survivorship (a surviving-fund basket biases a reversion test *toward* a result) is named on the **Signal** axis. A deterministic synthetic panel with a mean-reverting discount and a *planted* edge confirms the engine banks a real edge and finds nothing when the discount carries no information.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a discount *is*, why a fund can cost less than the stuff it owns, and why "buy the discount" both works and barely pays — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the proxy-discount sort, wide-minus-narrow long-short returns, a Welch _t_ + block-bootstrap + discount-shuffle placebo, costs and the hard-to-borrow short leg, the post-2010 decay, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`closed_end_fund_discount/`](closed_end_fund_discount/). The discount here is an explicit **proxy** (price relative to a per-fund benchmark), not reported NAV. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
