# Study 946 — Distribution is not Return 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is "distribution is not return" true on the tape? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The tape supplies **two** facts, not three. (1) The payout rank forecasts **the next payout**: +24.6 bps/sd, HAC *t* = **+11.3**, era-stable — the one number here that restates nothing else. (2) It forecasts **total return not at all**: −18.3 bps/mo, *t* = −1.24, CAPM α = −2.8 bps (*t* = −0.18). The celebrated **−69.9 bps/mo NAV erosion** (*t* = −4.53) is **an identity given those two** — `hml_price ≡ hml_total − hml_payout`, correlation 0.99995 — so it is arithmetic, not independent confirmation. Limits: the null is **bounded, not precise** (CI [−5.67%, +1.25%]/yr — it kills the marketed reading, not a downside penalty); **give-back 1.36 = "1.00 + an insignificant total leg"**; the erosion is partly cross-cohort (*t* = −1.93 inside the nine option-income wrappers alone, 74 months); the universe is **survivorship-selected**; the payout rate is a reconstructed **PROXY**; the corporate-action guard is a **hindsight** filter (fires once — the no-guard read is −57.0, *t* = −2.79). |
| **Tradability** — can you bank it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to trade in either direction. Long low-payout / short high-payout earns **+18.3 bps/mo gross** (*t* = 1.24) with a CI straddling zero, and the short leg pays a borrow fee that only deepens the hole (−37.3 bps at 25 bps cost + 200 bps borrow). Long-only, the high-payout tercile compounds at **+9.06%/yr** against the low-payout tercile's **+11.07%** and SPY's **+14.02%**, at excess-of-cash Sharpe **0.737 vs 0.862**, with α = −5.3 bps/mo on β = 0.65. |

> **In one sentence:** across fifteen income ETFs the advertised payout forecasts the **payout** almost perfectly (*t* = +11.3) and the **total return** not at all (*t* = −1.24, CI [−5.7%, +1.3%]/yr), which makes the famous price erosion the arithmetic remainder rather than a second discovery — the distribution rate is a **transparency tool, never a signal**.

## What we tested

Each month, rank fifteen listed income funds (**QYLD XYLD RYLD JEPI JEPQ SPYI DIVO NUSI PBP
PFF SPHD SCHD VYM DVY NOBL**) by their **trailing-12-month distribution rate** —
reconstructed from the gap between the `auto_adjust=True` **total-return** tape and the
`auto_adjust=False` **price-only** tape — buy the top tercile, sell the bottom, hold for the
following month (**one execution lag**). The same sort is then scored against three
left-hand sides: next month's payout, next month's **price** return, next month's **total**
return, Fama-MacBeth *and* as a tercile spread, with HAC *t*'s, block-bootstrap CIs, a CAPM
control (the fat payers are structurally 0.65-beta), an era cut, a sort-width/guard/universe
grid and a cost × borrow sweep. The three left-hand sides are **not independent** — the
payout is defined as the total/price gap, so the price leg is the total leg minus the payout
leg, and the study says so instead of counting the same *t* twice. A synthetic panel with a
tunable planted yield-to-return link is the machinery proof — quiet on the pure
return-of-capital null, and it recovers the
planted slope to +30.2 against +30.0. **Dedup:** [337-covered-call-etf](../337-covered-call-etf/)
races each fund **individually against SPY** and decomposes its own distribution — a
per-fund verdict; 946 asks the **cross-sectional** question those races cannot, whether the
payout *rank* orders subsequent total returns. [910-managed-distribution-cef](../910-managed-distribution-cef/)
holds a **closed-end** basket long on total-return closes only; 946 sorts an **ETF**
cross-section and measures the **price-only** erosion channel 910 could not see.
[62-premium-seller](../62-premium-seller/) is upside/downside capture versus a fund's own
underlying; [900-quality-income](../900-quality-income/) screens payers on quality, not
payout level. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a $1 distribution makes the fund worth $1 less, the price-CAGR column that sorts itself by payout, the give-back ratio, and what the fat-payout basket actually returned |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Fama-MacBeth slopes on three left-hand sides, the decomposed tercile spread, bootstrap CIs, the CAPM β/α control, the era cut, the guard/universe robustness grid, the cost × borrow sweep, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dist_illusion/`](dist_illusion/). Total-return vs price-only is labelled on every number; the distribution rate is a reconstructed **PROXY**; the corporate-action guard (a **hindsight** filter that fires once) and the borrow fee are **ASSUMPTIONS** and both are swept; the price leg is flagged as an **identity**, not a second experiment; survivorship is named on the Signal axis; the data stamp fingerprints returns, not back-adjusted levels. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
