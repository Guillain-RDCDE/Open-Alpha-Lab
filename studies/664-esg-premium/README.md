# Study 664 — ESG Premium 🌱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does ESG investing pay a premium (or cost you performance)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | ESGU vs SPY: active return **−0.057%/yr**, Newey-West *t* = **−0.08** (n=2,402d, 9.5y). SUSA vs IVV: **−0.644%/yr**, NW *t* = **−1.07** (n=5,387d, 21.4y). Neither clears **\|t\| ≥ 2** in *either* direction — no detectable premium, and just as tellingly, no detectable penalty. |
| **Tradability** — is there an edge to capture? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There's nothing to trade: net of a documented 5.5–22 bps/yr expense-ratio gap, one-way costs and a 30 bps/yr borrow drag, both spreads go further negative (−0.38%/yr, −0.95%/yr) while carrying **3.75–4.78%/yr of real, uncompensated tracking error**. |
| **"Any ESG return gap is just a growth/quality tilt?"** | ![Confirmed](https://img.shields.io/badge/Growth%2Fquality_tilt%3F-Confirmed-8b949e?style=flat-square) | Both funds carry a statistically real quality-factor beta (*t* = **3.41**, **8.50**); ESGU also a small growth beta (*t* = **3.63**). ~70% of SUSA's (statistically insignificant) raw gap is quality-factor beta, not stock-picking. |

> **In one sentence:** across 9.5 years of ESGU vs SPY and 21.4 years of SUSA vs IVV, ESG
> investing shows **no statistically detectable premium** (Newey-West *t* = −0.08 / −1.07) —
> but it does carry a measurable quality (and mild growth) factor tilt, real tracking error,
> and a real expense-ratio drag, so the honest read is a large-cap index fund with a labeled
> screen bolted on, not a source of alpha in either direction.

## What we tested

Do the two flagship US large-cap **ESG** equity ETFs — ESGU (iShares ESG Aware MSCI USA,
since 2016) and SUSA (iShares MSCI USA ESG Select, since 2005) — outperform (or underperform)
their plain-vanilla peers (SPY, IVV) on a risk-adjusted basis? We measure the tracking
difference, excess-of-cash Sharpe and tracking error of each ESG fund against its benchmark,
run a Newey-West active-return spread test (the planned primary — daily active returns are
autocorrelated), and then run a factor decomposition (fund return on benchmark + a
growth-value spread + a quality spread) to test the skeptical counter-claim head-on: is any
apparent ESG edge really just a relabelled large-cap growth/quality tilt? **Dedup:**
[211-sin-stocks](../211-sin-stocks/) tests the *mirror* claim (do shunned "sin" stocks
outperform?) with individual tickers, not the flagship ESG fund products; the full dedup map
against 200-roe-quality, 246-defensive-sectors, 335-buzz-sentiment-etf and 334-ark-innovation
is in [docs/references.md](docs/references.md). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "doing well by doing good" sounds so plausible, what ESG funds actually screen for, and why the answer turns out to be "nothing happens, either way" |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Newey-West active-return spread test, the growth-value/quality factor decomposition, tracking error and cost accounting, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`esg_premium/`](esg_premium/). ESGU/SUSA/SPY/IVV/IVW/IVE/QUAL are single,
currently-traded ETFs over their own listed history — no survivorship panel. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
