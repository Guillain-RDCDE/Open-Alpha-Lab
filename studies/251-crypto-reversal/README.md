# Study 251 — Crypto-Reversal 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the paper's reversal premium real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The 21-day cross-sectional spread on a public survivor panel is *momentum*, not reversal: Fama-MacBeth slope **+7.15 bps/day** (*t* = **+2.44** — recent winners win), reversal long-short *t* = **−2.10**. The paper's REV *t* = +6.19 flips sign back to Liu et al. (2022) momentum. Survivorship flatters the winner side; named, not fixed. |
| **Tradability** — does any spread survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~38%/day one-way turnover (daily reform of a 21-day signal). Even the *profitable* momentum side is gone by ~50 bps one-way (−7.4%/yr) and −76.5%/yr at the paper's own >100 bps crypto spread; the small-cap short leg is often unborrowable. |
| **Reversal or momentum?** | ![Momentum](https://img.shields.io/badge/Momentum-8b949e?style=flat-square) | On public, liquid, survivor tokens the Liu et al. momentum sign holds at every horizon (7→63d) — not the paper's reversal — and even that momentum never clears \|*t*\| ≥ 2 era-by-era. |

> **In one sentence:** the QuantNest-7 paper's headline — *short-term reversal, not momentum, prices the crypto cross-section* — does not survive contact with free data: on 20 liquid survivor tokens the sign flips back to momentum (Fama-MacBeth *t* = +2.44), and even that momentum spread is a mirage once you pay crypto's brutal daily-rebalancing turnover.

## What we tested

The QuantNest-7 paper ([Babayev & Aliyev, 2026, SSRN 6818558](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6818558)) claims a seven-factor model "with essentially zero overfitting" prices digital-asset returns, and that its short-term **reversal** factor — past losers beat past winners at 21 days, *t* = 6.19, +151%/yr — overturns the cross-sectional **momentum** of Liu et al. (2022). Its two genuinely novel factors (on-chain *value* and *quality*) rest on paid Coin Metrics data and undisclosed "proprietary" composites, so they are not reproducible; the reversal/momentum claim is price-only and **is**. We rebuild it on a public yfinance panel of 20 liquid non-stablecoin tokens (2019-2026), run the paper's own Fama-MacBeth apparatus, and stress it against a shuffle control, a synthetic positive/negative control, and the crypto turnover cost the paper's own limitations section flags.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why "reversal vs momentum" matters, the sign-flip in plain language, the cost wall |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Fama-MacBeth REV slope + HAC *t*, lookback sweep, shuffle & synthetic controls, cost/turnover, sub-era decay, survivorship & liquidity caveats |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crypto_reversal/`](crypto_reversal/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
