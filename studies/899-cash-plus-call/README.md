# Study 899 — Cash + Call "90/10" 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does "protect capital, rent upside" beat stocks risk-adjusted? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Capital protection is **real and mechanical**: max drawdown **−19.8%** vs buy-and-hold's **−55.2%** at half the vol, 2008 cut from −47% to −10%, confirmed by a 30-seed synthetic control. But the risk-adjusted-*return* claim is **not** met: even at the optimistic Black–Scholes-fair premium the excess-of-cash **Sharpe (0.492) ties** buy-and-hold's (0.543) — gap **−0.05**, bootstrap 95% CI **[−0.334, +0.210]** through zero (P(win) 34%) — and the option's **convexity adds no alpha** (spanning α +0.46%/yr, ***t* = +0.33**; symmetric up/down capture). Both eras insignificant (α *t* < 2). *Single GFC-anchored ~19-yr window; the BS mark flatters (realized-vol pricing, no dividend) — named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The Sharpe parity is a **mirage of the fair-premium assumption**. A real listed call carries the **variance risk premium** (IV > RV): price it at a realistic **1.25–1.5× markup** and 90/10's excess Sharpe drops to **−0.21…−0.35**, clearly below buy-and-hold — plus the call-holder forgoes the ~1.8%/yr dividend. Not a friction story (annual roll ⇒ **0.18×/yr** turnover, costs are a rounding error); renting upside is simply **negative-carry**. At best it *matches* buy-and-hold's Sharpe while giving up **~4.6%/yr** of CAGR. |

> **In one sentence:** 90/10 genuinely floors the drawdown (−20% vs −55%, 2008 cut to −10%) — but on
> risk-adjusted return it only **ties** buy-and-hold (Sharpe 0.49 vs 0.54, a statistical dead heat with
> no convexity alpha), and that tie exists **only** at a fictional fair option price; charge the real
> variance risk premium and it loses — so the floor is **real** and the free lunch is a **mirage**.

## What we tested

Bill **Gross's "90/10"** (Bodie's "T-bills + calls"): keep ~90% in T-bills so capital returns to par
and spend ~10% **renting convex upside** with calls. Free listed-option history doesn't exist, so the
~10% sleeve is a **documented proxy** — a rolling **1-year ATM SPY call marked daily with
Black–Scholes** (strike = spot at each annual roll, priced off trailing realized vol & the ^IRX bill
rate; the 10% budget buys as much notional as the fair price affords, capped near 100% of NAV). We
race it **excess-of-cash** against **buy-and-hold SPY** and a **matched-average-exposure static mix**
(yfinance daily total-return, 2007-05-31 → 2026-06-30), with a leverage-clean **convexity** spanning
alpha, a block-bootstrap Sharpe-difference CI, a two-era cut, a **premium (variance-risk-premium)
sweep** and a cost sweep, the up/down capture asymmetry, and a 30-seed synthetic control. The window
is one GFC-anchored cycle (BIL lists 2007) and the BS mark flatters the strategy — both named on the
**Signal** axis. **Dedup:** [897-cppi-floor](../897-cppi-floor/) buys a put-*like* payoff by trading
the *underlying* (no options, cash-drag cost); [617-crash-insurance-cost](../617-crash-insurance-cost/)
buys *puts* to hedge an equity book's *downside*; [173-four-percent-rule](../173-four-percent-rule/) is
a *withdrawal* rule on a static mix; [68-all-weather](../68-all-weather/) is a *diversified
risk-parity* allocation. This study is **T-bills + a rolling long call** — rent the upside, insure the
capital — priced against its option premium. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why capital protection really holds, why it only *ties* stocks risk-adjusted, and why a real option costs more than the model |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the three-book race, the excess-of-cash identity, the convexity spanning alpha, the bootstrap Sharpe-difference CI, the two-era cut, the premium/cost sweeps, and the 30-seed control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cash_call/`](cash_call/). Real tape via yfinance (`auto_adjust=True` total-return SPY/BIL +
^IRX rate), cached under `_cache/`. The 10% sleeve is a Black–Scholes-marked proxy for a listed call.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
