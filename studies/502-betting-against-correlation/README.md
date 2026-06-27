# Study 502 — Betting-Against-Correlation

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Is the famous "bet against beta" premium really a bet against **correlation** — not volatility?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the correlation premium statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The BAC book earns **+3.75%/yr gross** (HAC *t* = **+1.54**), **+3.36%/yr net** (*t* +1.39) — below the *t* ≥ 2 bar, bootstrap CI **[−1.0%, +8.5%]** straddles zero. A label-shuffle placebo puts the real mean in the right tail (**p = 0.04**) and AFGP (2020) give strong prior support, so `WEAK` not `NONE`. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Sharpe **+0.41**, max drawdown **−20.9%**, needs a short leg (borrow charged); brutal in correlation-spike years (2020 −11.5%, 2023 −16.2%). Net of 5 bps/leg + 50 bps borrow it survives but is thin and uncertain. |
| **Is it correlation, not vol?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | On the *same* panel: sort on **correlation** earns **+3.75%/yr**; sort on **beta** earns **−0.67%/yr**; sort on **volatility** earns **−0.87%/yr**. The premium lives in the correlation slice — exactly the AFGP (2020) claim. |

> **In one sentence:** Asness–Frazzini–Gormsen–Pedersen are *right about the mechanism* — on a 48-name survivor panel the correlation sort earns +3.75%/yr while the beta and volatility sorts earn nothing, so the low-risk premium really is a correlation effect (placebo p = 0.04) — but on 150 months of large-cap survivors the tradable book sits at HAC *t* = 1.54, below the bar: a real, identified mechanism with a magnitude this sample can't certify.

## What we tested

The **Betting-Against-Correlation** decomposition (Asness, Frazzini, Gormsen & Pedersen 2020).
Since `beta = correlation × (vol_stock / vol_market)`, AFGP split Betting-Against-Beta into a
correlation leg (BAC) and a volatility leg (BAV) and argue the **correlation** slice carries the
premium. We rank a 48-name large-cap survivor cross-section by trailing 252-day
**correlation-to-market**, go long the low-correlation half and short the **beta-neutralised**
high-correlation half (one execution lag, enter the close one day after the signal), and report
it honestly: HAC *t*, a block-bootstrap CI, a label-shuffle placebo p, costs × turnover + short
borrow, the correlation-vs-beta-vs-vol decomposition, and a seed-robust synthetic positive
control. yfinance daily prices, 2012–2025, 150 monthly observations. The universe is
survivorship-biased — we name it and treat results as upper bounds. *Distinct from
[238 Betting-Against-Beta](../238-betting-against-beta/) (sorts on beta) and
[330 Low-Volatility-Anomaly](../330-low-volatility-anomaly/) (the ETF vol race).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "bet against beta" is secretly "bet against correlation", the corr-vs-beta-vs-vol race in plain language, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling-correlation sort, beta-neutralisation, HAC *t* + bootstrap CI, the label-shuffle placebo, costs + borrow, the AFGP decomposition, the seed-robust synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run
(fp `3f75d60c45d8`): [docs/results.md](docs/results.md).

---

*Engine: [`betting_against_correlation/`](betting_against_correlation/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
