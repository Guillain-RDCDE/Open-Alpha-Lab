# Study 886 — Agency-MBS-Carry 🏠

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the MBS-over-Treasury carry a real, positive premium? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The story is coherent and the sign is right — the duration-neutral **MBB** carry is **+0.30 %/yr** and clears *t* ≥ 2 *inside* the calm 2007-13 / 2014-19 sub-eras (*t* = +2.50 / +2.84) — but full-sample **HAC *t* = +0.64**, the bootstrap CI **[−0.62, +1.20]** straddles zero, the carry **collapses to +0.17 %/yr (*t* +0.18) across the 2020-2026 rate-vol era** it exists to compensate for, and it flips **negative (−0.30 %/yr)** under a static-OAD duration match. VMBS corroborates (+0.17 %/yr, *t* +0.37). Real mechanical premium, but it can't clear the robust bar. |
| **Tradability** — can you bank it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The ~0.30 %/yr gross carry is *smaller* than round-trip ETF frictions (~0.45 %/yr with 1–2 bp spreads + 40 bps/yr borrow), so costed net is **−0.16 %/yr (MBB) / −0.30 %/yr (VMBS)**; and a proper static-duration hedge turns the spread negative before any cost. The apparent pickup is residual duration, not a bankable premium. |

> **In one sentence:** the "agency MBS pay you a spread over duration-matched Treasuries"
> premium is *real ex-ante* (the OAS is genuine compensation for prepayment risk) but on the
> live ETF tape the realized duration-neutral carry is **+0.30 %/yr at HAC *t* = +0.64** —
> paid in calm regimes and clawed straight back in the 2008 / 2020 / 2022 rate shocks — with
> a Sharpe advantage of **+0.016** over IEF, a bootstrap CI that straddles zero, and a net
> that goes **negative** after costs: **negative convexity eats the spread.**

## What we tested

Agency MBS (**MBB**, **VMBS**) are Treasuries plus a short prepayment option, so they carry an
option-adjusted spread as compensation for **negative convexity**. We harvest that as the
**duration-neutral, cash-neutral** monthly spread `carry = (MBS − cash) − β·(IEF − cash)`,
both legs excess of **BIL** cash, with `β` fit two ways — the **empirical** realized rate
sensitivity (~0.52, which already prices in the convexity drag) and the **static** published
OAD ratio (6.0 / 7.5 = 0.80). Inference is **Newey-West HAC** (6 lags, 3/12 sensitivity) on
the carry mean, a **circular block-bootstrap** 95% CI, an excess-vs-excess **Sharpe race** vs
duration-matched IEF, a **three-era cut**, and a **costed** version (ETF spreads on both legs +
borrow on the short Treasury). A deterministic synthetic world with a **planted, tunable carry**
proves the estimator recovers the knob and stays silent on the null. **Dedup:** distinct from
[577-mbs-oas-signal](../577-mbs-oas-signal/) (which times risk-off on MBS *OAS widening* — a
signal; here we harvest the *carry* itself), from [340-bank-loans](../340-bank-loans/) (leveraged-loan
credit carry), [796-corporate-bond-low-risk](../796-corporate-bond-low-risk/) (IG low-risk anomaly)
and [581-term-premium](../581-term-premium/) (the Treasury duration premium, which we *hedge out*
here). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a mortgage bond really is (a Treasury plus a short refi option), why it "should" pay you a spread, and why that spread quietly vanishes once you duration-hedge and pay costs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the duration-neutral carry, empirical vs static-OAD hedges, HAC *t* + bootstrap CI, the excess-vs-excess Sharpe race, the three-era cut, calendar years, the costed net, and the planted-carry synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`mbs_carry/`](mbs_carry/). The carry is `(MBS − cash) − β·(IEF − cash)` per month,
both legs excess of BIL; `β` is the realized (regression) rate beta or the static OAD ratio.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
