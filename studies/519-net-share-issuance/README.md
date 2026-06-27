# Study 519 — Net-Share-Issuance 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-issuance (buyback) firms beat high-issuance (diluting) firms? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a **40-name large-cap survivor** basket over **9** annual rebalances, the long-short (long low-issuance, short high-issuance) spread is **−3.6%/yr at *t* = −0.86** — the **wrong sign** for the Pontiff-Woodgate factor and deep inside its standard error. A **label-shuffle placebo** beats the real sort **79%** of the time; the win-rate is **44%**. The faithful synthetic control shows a *real* low-issuance edge would print a clearly **positive** *t*, so this is **noise**, not a hidden sign-flip. **Survivorship** named on this axis. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A spread you can't tell from zero (and that points the wrong way) is not a book at any size. One-way **10 bps × 2 legs + 50 bps** short borrow only deepens it (net **−4.3%/yr**). Stable-but-insignificant across every quantile width (∣t∣ ≤ 1.3) and both half-samples — nothing to size. |
| **Does the published factor replicate here?** | ![Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square) | A canonical, pointed academic factor replicated honestly on a small survivor basket with real costs lands exactly where the desk expects **most** of them to: sign reverses, *t* never clears 2, placebo unbeaten. The effect that lives in CRSP's full thousands-name point-in-time cross-section **does not survive** a 40-name survivor replication — a small-sample / survivorship illustration, not a free lunch. |

> **In one sentence:** the share-issuance factor — *issuers underperform, buyback-ers outperform* —
> is real and robust in the **full** US cross-section (Pontiff-Woodgate 2008), but replicated
> honestly on **40 large-cap survivors** over **9** years with a 1-year lag, split-adjusted shares
> and real costs, the long-short spread comes out **−3.6%/yr at *t* = −0.86** — the *wrong sign*,
> inside its own noise, with a label-shuffle placebo that beats it 4 times in 5 and a faithful
> synthetic control confirming a genuine edge would have printed a positive *t* — so on this tape
> it is **None × Mirage**, a textbook small-sample / survivorship illustration rather than a
> tradable edge.

## What we tested

A point-in-time, survivorship-free issuance universe is a CRSP/Compustat product, not a free
yfinance feed, so we fix a **transparent 40-name large-cap survivor basket** and compute each
firm's **net issuance** = the year-over-year change in its **split-adjusted** shares outstanding
(yfinance `get_shares_full` + the firm's split history — so a 4-for-1 split does *not* look like
300% dilution). Each formation year-end we sort the cross-section, go **long the bottom 30%**
(buybacks / low issuance), **short the top 30%** (dilution / high issuance), and hold the
**next** year (one execution lag — the share count is public at year-end *t*, we trade *t→t+1*).
Inference is a one-sample *t* against zero plus a **label-shuffle placebo null** (permute which
name carries which issuance value), with one-way costs × turnover + a short-leg borrow, a
quantile-width sweep, and a split-half cut. A deterministic **synthetic control averaged over 25
seeds** confirms the engine recovers a planted edge and refuses to manufacture significance when
there is none. This is the **realised composite-issuance factor** — distinct from
[Study 368 — Buyback-Drift](../368-buyback-drift/), which times discrete *announcements*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "companies that buy back their stock beat companies that print new shares" is a real Wall-Street fact that *vanishes* on a few dozen big survivors — split traps, the wrong sign, and why 9 years can't settle it, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the annual cross-sectional sort, long-short vs zero, a one-sample *t* + label-shuffle placebo null, quantile-width & split-half robustness, costs + borrow, and a 25-seed synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`net_share_issuance/`](net_share_issuance/). The basket is an explicit **large-cap survivor sample**, not a point-in-time universe (survivorship named on the Signal axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
