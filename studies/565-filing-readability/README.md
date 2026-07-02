# Study 565 — Filing-Readability 📄

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do harder-to-read 10-Ks predict lower returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The Loughran-McDonald readability anomaly is a well-documented *Journal of Finance* result, and our engine recovers it cleanly on a calibrated synthetic world (IC **−0.34**, *t* **−7.1**; long-short **+12.1%**, *t* **+6.4**; risk leg positive, *t* +36) while staying flat at the null (IC-*t* **−0.17**, placebo *p* 0.30). But **there is no free real 10-K-text tape** — a point-in-time EDGAR-text + survivorship-free return panel isn't a no-key retail artifact — so the tape can't certify it. `REAL` needs *t* ≥ 2 on a **real** tape; literature-real + no tape reads **`WEAK`**. |
| **Tradability** — does the spread pay? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | In the plausibility model the long-readable/short-murky book is net-positive (gross **+12.1%** → net **+10.4%** after 5 bps/leg + 150 bps borrow), but the murky short leg is exactly the small/illiquid, hard-to-borrow tail, and with no real tape there's no capacity or turnover estimate to stand on. |

> **In one sentence:** the obfuscation story — managers bury bad news in longer, murkier, bigger 10-Ks and the market underreacts, so the hard-to-read filers drift down — is a real, replicated *Journal of Finance* anomaly that our engine banks cleanly on calibrated synthetic data (IC −0.34, *t* −7.1) and would bank on the tape if it could see one; but a free point-in-time 10-K-text panel doesn't exist for a no-key retail stack, so the tape can't certify it and the stamp is `WEAK` × `FRAGILE`.

## What we tested

The **readability anomaly** (Loughran & McDonald 2014; Li 2008; You & Zhang 2009): 10-K filings that
are *harder to read* — higher Gunning **fog**, longer documents, bigger **file size** (LM's preferred
robust proxy) — precede **lower** future returns and **higher** post-filing volatility. We build a
standardised **obfuscation score** (z(fog) + z(length) + z(file size), higher = less readable), sort a
synthetic cross-section into terciles, and test whether readable filers beat murky ones: the Spearman
**information coefficient** with a Fisher-z *t*, a two-sample *t* on the long-short spread, a
**label-shuffle placebo** null, the firm-level slope, the LM **risk leg** (post-filing vol on
obfuscation), a **per-leg IC sweep** (fog vs length vs file size — LM say file size is cleanest), costs +
a punitive short borrow, and a deterministic, seed-robust synthetic positive control that plants the
anomaly and proves the engine catches it (and stays flat at the null). **Data-availability limit,
stated openly:** a free point-in-time 10-K-text + survivorship-free return panel doesn't exist for a
no-key retail stack, so this study is **synthetic-only** — it can never earn `REAL` (that needs a robust
*t* ≥ 2 on a **real** tape). *Distinct from the desk's **sentiment** studies — [257 AAII-sentiment](../257-aaii-sentiment/),
[335 Buzz-sentiment-ETF](../335-buzz-sentiment-etf/), [392 Glassdoor-sentiment](../392-glassdoor-sentiment/)
test tone/opinion; this is the **readability / length** structure of the filing.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the readability anomaly is, why "murkier filing = worse returns" is a red flag, and why we can only test it on synthetic data |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the obfuscation score, the IC + Fisher-z *t*, the tercile long-short with a two-sample *t*, the placebo null, the risk leg, the per-leg IC sweep, costs + borrow, and the seed-robust synthetic positive control |

The synthetic headline run (400 firms, `obf_alpha = -0.10`, panel fp `53136605f951`, as-of 2026-06-30)
is in [docs/results.md](docs/results.md); the real-tape stub in
[`filing_readability/data.py`](filing_readability/data.py) is **empty by construction** on a free stack.

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`filing_readability/`](filing_readability/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
