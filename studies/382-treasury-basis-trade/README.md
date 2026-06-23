# Study 382 — Treasury-Basis-Trade 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the net carry actually there? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Yes — the 10y-over-funding carry is positive on **86.5%** of days and the unlevered basis return survives autocorrelation-robust inference overwhelmingly (Newey-West **t = 11.7**, sign-flip placebo **p = 0.000**, Sharpe **2.86**). But "real" here means a **real risk premium** — beta you're paid for — not free alpha. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The "**+73%/yr**" is just the **+1.5%** premium **× 50 leverage**; the Sharpe is *unchanged* (no reward for the leverage, only risk). Max drawdown is **−75%** at 50× / **−95%** at 100×, a single bad day is **−18.5%**, and a March-2020 funding shock takes **−16%** at 50× in two weeks while margin calls force the unwind. Costs are trivial; the **tail × leverage** is fatal. |
| **"Free carry"?** | ![Busted](https://img.shields.io/badge/Free_carry%3F-Busted-8b949e?style=flat-square) | The smooth accrual hides **excess kurtosis ≈ 10** and negative skew — the fingerprint of a **short-volatility carry**, not free money. It's "picking up pennies in front of a steamroller" in a Treasury costume; only a central-bank backstop has ever absorbed the tail. |

> **In one sentence:** the Treasury basis trade's carry is unmistakably **real** (a term/funding risk premium, NW *t* ≈ 12, Sharpe 2.86 unlevered) — but Sharpe is **leverage-invariant**, so the hedge-fund "free carry" is that 1.5%/yr premium multiplied by 50–100×, which leaves the risk-adjusted reward untouched while turning a silky stream into a fat-tailed, −75%-drawdown short-volatility position that a March-2020-style repo shock force-unwinds: **real as a premium, a mirage as free money.**

## What we tested

Clean cash-vs-futures bond prices and GC repo aren't free on yfinance, so we build a **transparent carry / implied-repo model** (labelled a model throughout): the basis trade's daily **net carry** is the 10-year yield (`^TNX`) minus a short funding rate (`^IRX`, the 13-week T-bill as a **repo proxy**), earned with a 1-day lag, plus a small-duration residual mark-to-market. We measure whether that carry is statistically real on the **unlevered** (leverage-invariant) series with a Newey-West *t* and a sign-flip placebo — then apply the **50–100× leverage the trade actually runs** and watch the *same* Sharpe coexist with a −75% drawdown, fat tails (excess kurtosis ≈ 10), and a March-2020 funding-shock unwind. A deterministic synthetic control with a *planted* unlevered Sharpe confirms the inference is faithful **and** that leverage-invariance alongside an exploding drawdown is mechanical, not an artefact. (Same carry-crash signature as [Study 364 — FX-Carry-Trade](../364-fx-carry-trade/), in rates.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the basis trade is, why "free carry" is the premium times leverage, and how a 0.4% bad day becomes −18% — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | net-carry construction on the implied-repo model, HAC *t* + sign-flip placebo on the unlevered series, the leverage-invariant Sharpe, the fat-tail / CVaR diagnostics, a March-2020 funding-shock stress, and a synthetic planted-Sharpe power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`treasury_basis_trade/`](treasury_basis_trade/). The basis here is an explicit **carry / implied-repo model** (10y − bill funding), not a live cash-vs-futures feed. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
