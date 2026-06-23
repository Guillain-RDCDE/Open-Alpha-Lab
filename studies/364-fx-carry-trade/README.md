# Study 364 — FX-Carry-Trade 💱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the carry premium statistically real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | UIP genuinely fails — the long-high / short-low G10 basket earns **+1.97%/yr** gross — but on 22 years it is small, **fails t ≥ 2** (HAC *t* = **1.14**), and carries a **−0.75 skew** with a **−35% drawdown**. Literature-real, this-tape-insignificant ⇒ WEAK, not REAL. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Net of a realistic short-leg borrow the premium is **+0.99%/yr at Sharpe 0.12** — and it comes bolted to a fat left tail that erased years of grind in a single month (**−11.7%** in Oct-2008). Thin, crash-prone, costs-sensitive: survivable on paper, not a NAV-scale allocation. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | In calm months carry earns **+8%/yr**; in the worst decile it loses **−52%/yr**, giving back **2.66×** the cumulative gain. The premium is **compensation for selling crash insurance** — a high Sharpe with deeply negative skew, not arbitrage. |

> **In one sentence:** borrowing low-yield currencies to hold high-yield ones really does earn a positive premium (UIP fails) — but on a 22-year G10 tape it is a thin **+1.97%/yr** at HAC *t* = 1.14, net of borrow it is **+0.99%/yr at Sharpe 0.12**, and it is **negatively skewed (−0.75, −35% drawdown)**, losing −52%/yr in the worst decile of months (Oct-2008, Mar-2020, the 2015 CHF de-peg); the carry trade is not a free lunch, it is a premium for bearing crash risk.

## What we tested

The textbook macro free lunch: each month, rank the **nine G10 currencies** by their **carry** (their short-rate differential vs USD), go **long the top-3 high-yielders** (AUD, NZD, NOK) and **short the bottom-3 funding currencies** (EUR, JPY, CHF), dollar-neutral, and pocket the rate gap. True deposit rates aren't on yfinance, so we attach a **transparent fixed carry proxy** to each currency (labelled a proxy throughout) and run it over **real yfinance FX spot, 2004→2026**. The decisive question isn't whether the mean is positive (it is — UIP fails) but the **shape**: we measure skewness, the worst month, the max drawdown, and a calm-vs-risk-off split, judge the mean with a **Newey-West HAC *t***, and charge **one-way costs + borrow on the short leg**. A deterministic synthetic control plants a *known* carry premium and a *known* fat-tailed crash, confirming the engine recovers both — and manufactures no significance from a zero-carry null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what carry is, why "borrow cheap, lend dear" pays for years, and why the gains vanish in a crash — in plain language, every chart drawn by the code |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | carry-ranked basket construction, HAC *t* on the mean, skew / drawdown / crash-conditional split, a cost + borrow sweep, and a synthetic premium-and-crash positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`fx_carry_trade/`](fx_carry_trade/). Carry here is an explicit **proxy** (a fixed per-currency short-rate differential), not live deposit rates. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
