# Study 365 — Lottery-MAX-Effect 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the flashy tail underperform? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a surviving large-cap proxy the claimed edge is **absent** — and what *is* significant runs the **wrong way**: the long-low / short-high MAX book loses **−17.6%/yr** at HAC *t* = **−3.44**. High-MAX (Q5) nearly **tripled** low-MAX (Q1) return (+29.8% vs +12.2%/yr). The published CRSP small-cap anomaly **inverts** here because, on **survivors**, "high MAX" tags the decade's growth winners (named on the Signal axis). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The only statistically real spread is a *losing* one; the textbook lottery short is negative gross and worse net of 20 bps/leg + 50 bps/yr borrow. No deployable low-minus-high MAX edge exists on these names. |
| **"Lottery losers"?** | ![Busted](https://img.shields.io/badge/Lottery_losers%3F-Busted-8b949e?style=flat-square) | On liquid large-caps the one-day-pop tail did **not** underperform — it posted the **highest return *and* the highest Sharpe**. The morality tale is a small-cap / illiquid phenomenon. |

> **In one sentence:** the lottery-MAX effect — buy boring, avoid the flashy one-day pop — is a real *small-cap* anomaly, but on a surviving S&P-100-style large-cap basket it **inverts**, with the long-low / short-high book losing −17.6%/yr at HAC *t* = −3.44 because on survivors "high MAX" just re-labels the bull market's high-beta growth winners; a synthetic control with a planted lottery penalty confirms the engine would have caught a real effect (and finds none at edge = 0), so this is `NONE` on signal, `MIRAGE` to trade, and a `BUSTED` folk tale at the large-cap scale most people invest at.

## What we tested

The Bali-Cakici-Whitelaw (2011) **MAX** sort: each month rank the cross-section by each name's
*highest single-day return over the prior month*, form quintiles, and go **long the low-MAX
(boring) names, short the high-MAX (lottery) names**. True MAX is a CRSP-universe object
(thousands of names, small-caps included), but yfinance gives only a large-cap slice, so we run
the sort on a fixed **S&P-100-style 66-name basket** and call it a **proxy** throughout —
explicitly survivorship-tilted, named on the Signal axis. We measure each quintile's next-month
return with a one-month execution lag, test the long-short mean with a Newey-West (HAC) *t* and
a sign-flip placebo null, charge one-way costs × turnover plus a short borrow, and confirm the
engine on a deterministic synthetic panel with a *planted* lottery penalty (which it recovers,
and finds nothing when the penalty is zero).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "boring beats flashy" is a real small-cap idea, why it flips on big liquid stocks, and what "high MAX" actually picks out among survivors — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the monthly MAX quintile sort, the long-short spread with a Newey-West *t* + sign-flip placebo null, robustness across cut granularity and sub-periods, costs, and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`lottery_max_effect/`](lottery_max_effect/). The cross-section here is an explicit **large-cap proxy** (a 66-name survivor basket), not the CRSP universe. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
