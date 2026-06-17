# Study 274 — Fine-Wine

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does fine wine (Liv-ex 100) diversify or just lag?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Raw correlation with the S&P is **+0.10** (looks like a clean diversifier) — but the Liv-ex 100 is a *mid-price* index with a **+0.31** first-order autocorrelation; Geltner de-smoothing lifts the correlation to **+0.28**. Real-but-modest equity beta with literature support, *not* the zero-correlation pitch; no diversification-benefit HAC t reaches +2 (best +**−0.82**), n = 22. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Wine **lagged** equities by ~3.9pp/yr (4.7% vs 8.6% CAGR, price-only); the index is an uninvestable mid-price benchmark; ~10–15% round-trip spreads plus ~1%/yr storage erase the paper benefit. Net of de-smoothed risk and costs a 20% sleeve **lowers** the portfolio Sharpe (−0.047). |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The "low correlation = free diversification" pitch is mostly an appraisal-smoothing artefact; de-smoothed and costed, the benefit vanishes. |

> **In one sentence:** fine wine *looks* like a low-correlation diversifier only because its mid-price index is smoothed — de-smooth it and it is a higher-beta asset that lagged the S&P by ~4pp/yr and is uninvestable at the index level.

## What we tested

We hardcode the **Liv-ex 100 Fine Wine Index** year-end levels (2003–2025) in
`data.py`, join them with S&P 500 calendar-year price returns from the repo-level
`^GSPC` cache, and ask whether adding a wine sleeve to an equity book actually
helps. We report the **raw** correlation (the marketing number), the **Geltner
de-smoothed** correlation (the honest number, since mid-price indices are
smoothed by construction), risk/return (CAGR, vol, Sharpe), and a
mean-variance **diversification-benefit test** across four lenses — gross/net of
wine's large frictions and raw/de-smoothed risk — each with a Newey-West HAC
t-stat. The synthetic positive control plants a known correlation under known
smoothing and confirms the de-smoothing machinery recovers it; the real tape
confirms there is no robust, net-of-cost benefit.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the diversification pitch, "lag vs diversify", the smoothing trap, the cost reckoning in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Geltner unsmoothing, the two-asset frontier, gross/net × raw/de-smoothed benefit, HAC t-stats, the n=22 power problem |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fine_wine/`](fine_wine/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
