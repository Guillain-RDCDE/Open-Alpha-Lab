# Study 952 — After-Tax Equivalent 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Hook:** *Municipal bonds pay less than corporate bonds and hand you the difference back as a tax break — at what bracket does that trade become worth taking?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the tax break buy a real after-tax win? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | One number here is precise, and it is not a win. On **realised, reconstructed** distributions the **income-leg break-even** is **26.7% [23.4, 29.7]** vs VCIT and **28.9% [27.1, 30.7]** vs LQD — below the top brackets. Everything above that fails. The **total-return** break-even is *unidentified*: 35.0% point, 95% CI **[−10.9%, +81.5%]** (41% of draws exceed the highest US rate), because its numerator never clears \|*t*\| ≥ 2 in any pairing. The top-bracket after-tax edge is **+0.21 pp/yr, HAC *t* = +0.25**, CI [−12.5, +15.6] bps/mo. And the three rows that *do* clear \|*t*\| ≥ 2 clear it on a near-deterministic **tax constant** worth 60–86% of the mean and <2% of the variance — the same lift appears on a twin null with nothing planted. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The **asset-location** call is real and free: in a top bracket, hold the muni fund in the taxable account — the income-leg break-even backs that with a tight interval, and the switch is one round trip costing **0.03 bps/month** amortised. Fragile because nothing past the coupon comparison holds: the total-return margin is inside the noise, the answer rides on an **assumed** bracket, the crossover cannot be pinned to better than [−11%, +82%], and as a *trade* **25 bps/yr of borrow** on the short taxable leg flips MUB−VCIT negative. |

> **In one sentence:** reconstructing each fund's income leg as total-return minus price-return, the *coupon* comparison says a top-bracket investor should indeed prefer national munis to like-for-like corporate credit — but the *total-return* race is a dead heat, the crossover bracket is barely identified, and every significant *t*-stat on the page is the tax code being nonzero rather than a market being wrong.

## What we tested

Race four muni ETFs (**MUB, VTEB, SUB, HYD**) against taxable credit (**AGG, LQD, VCIT**) and cash (**BIL**) on an **after-tax** basis, 2004-02 → 2026-06 monthly. Each leg's income return is *reconstructed* as `total return − price return` (both tapes from `yfinance`, `auto_adjust` on and off); only that leg is taxed — munis exempt from federal + the 3.8% NIIT, taxable credit paying `federal + NIIT + state`, T-bills state-exempt, the price leg untaxed by default. The break-even bracket solves in closed form and is reported **with a block-bootstrap interval** for all seven pairings, alongside the income-leg-only break-even, a five-bracket ladder, state / in-state / capital-gains knobs, a decomposition of every *t*-stat into pre-tax difference plus tax constant, an era cut, cost and **borrow** sweeps, and one signal arm (a trailing-yield switch) carrying the study's single execution lag. Every tax rate is a labelled **PROXY** and every one is swept.

**Dedup:** distinct from **[576 — Muni-Treasury-Ratio](../576-muni-treasury-ratio/)** (the muni/Treasury ratio as a *timing signal*, not an after-tax level), from **[887 — HY-Muni-Premium](../887-high-yield-muni-premium/)** (a *within-muni credit* spread that checks tax-equivalent yield at one bracket only) and from **[616 — Muni-CEF-Tax-Loss](../616-muni-cef-tax-loss/)** (a December *calendar* effect). It belongs with the desk's mechanical-identity studies **[613](../613-currency-hedged-etf-carry/)** and **[889](../889-dollar-hedge-overlay/)** — and unlike those two, the identity here does *not* survive contact with a matched comparator.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a lower muni yield can still be the better bond, how we recovered the coupon from prices, how wide the crossover bracket really is, and why a *t*-stat here is not what it looks like |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the linear-in-tau identity, the closed-form break-even **and its bootstrap interval**, the income-leg-only break-even, the tax-constant decomposition of the *t*-stat, the bracket ladder, the era cut, the state / cap-gains / cost / borrow sweeps, and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`after_tax/`](after_tax/). **Not investment advice** — research & education, not tax advice. See [LICENSE](../../LICENSE).*
