# Study 931 — The CEF IPO Hole 🕳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

A closed-end fund sells you a basket of securities you could buy yourself. What happens to the price of the wrapper after the sales desk stops calling?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the hole real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | A new closed-end fund loses ground to its *own asset class* on every estimator that survives the dependence problem, at the horizons where the tests have power: **−5.1%** by month three (*t* = −3.94; calendar-time HAC *t* = **−2.86**; vintage-cluster CI [−7.9, −3.5]) and **−7.1%** by month six (*t* = −4.06; HAC *t* = **−2.12**). Twelve-month magnitude **−10.3%**, 25/28 funds negative (Wilson [0.73, 0.96]). Both vintage halves clear alone (−2.10 / −2.99), every benchmark mapping reproduces it, each fund's **own fitted beta removes only a fifth** of it (−8.4% → −7.3%, *t* = −2.56, so it is not leverage), and a pseudo-IPO placebo on the same funds' *seasoned* life reads **+1.9%**. **Named:** at 12 months the calendar-time book (1.4 names/day) reads *t* = **−1.57** and does **not** clear — the 12m number is the magnitude, not the evidence; the **universe is a hand-built sample of 28 large launches, not a census** (2017 and 2018 are empty); **survivorship** — surviving tickers only, BIGZ converted away, XFLT dropped for corrupt splits — which biases the hole toward zero. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Real and unbankable. The only position is a short of the new issue against its benchmark: **+9.1%** over 12 months at zero borrow, but break-even borrow is **~908 bps/yr** and a CEF weeks after its IPO has essentially no lendable float — the whole issue sits in the retail accounts the syndicate placed it in. There is no long expression: you cannot buy the hole. What's left is an **abstention rule**, and refusing to lose 10% pays you nothing. |

> **In one sentence:** Buy a closed-end fund at its IPO and you hand the underwriter 3-6% for a basket you could have bought yourself — the tape says the price then holds for about a month while the syndicate stabilises it, falls **5.1% behind the asset class by month three** and about **10% behind by month twelve**, in **both vintage halves**, not from leverage, and **only** in the IPO window (the same funds are flat-to-positive once seasoned) — but the short that would monetise it cannot be borrowed, so the finding is worth exactly one behaviour: wait.

## What we tested

A hardcoded list of **28 US closed-end funds that IPO'd between 2012 and 2022** (nine
vintages, none in 2017-2018), each measured against **one asset-class benchmark ETF** — a
tech CEF against XLK, a high-yield CEF against HYG, a preferred fund against PFF, and so on.
Entry is the **offering price the subscriber pays** ($20, or $25 for preferred funds — an
ASSUMPTION, corroborated by the tape for 21/28 funds within 1%, and swept against a
day-one-close entry, which recovers only 0.7 pp of the 10.3 pp). Both legs are
**total-return** closes (`auto_adjust=True`), which matters enormously when the fund
distributes 8-12% a year. The list is a **convenience sample of large, still-listed
launches, not a census** — that limit, and survivorship, are named on the Signal axis. We
report the abnormal return at 1/3/6/12 months with a one-sample *t*, a **vintage-cluster
bootstrap**, a **calendar-time portfolio** with a Newey-West *t* (established at t+1 and
earning from the session after — the single execution lag), a **beta control** that refits
each fund's own beta on its seasoned life, a time-to-discount proxy, a vintage era cut,
benchmark-mapping and entry sweeps, a **pseudo-IPO placebo**, and the costed short with a
**borrow sweep**. The 3-6% load is quoted as context and **never subtracted** — it is the
mechanism, and the tape already contains it.
**Dedup:** distinct from **367-closed-end-fund-discount** (a cross-sectional discount sort on
*seasoned* funds) and **910-managed-distribution-cef** (a seasoned CEF's payout vs its asset
class) — both study the steady state, 931 studies the event that creates it; from
**929-rights-offering-discount** (a *later* share sale by an existing fund); from
**616-muni-cef-tax-loss** (a calendar effect); from **378-etf-nav-premium** (open-end ETFs,
where arbitrage closes the gap); and from the operating-company IPO studies
(**219**, **545**, **623**, **783**, **874**), where underperformance is confounded with the
business — a closed-end fund has none, so the benchmark holds the very securities the fund does.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what you actually buy for $20, the month of quiet before the fall, the seven weeks to a discount, and why "just wait" is the whole lesson |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the abnormal-return table, cluster bootstrap vs calendar-time HAC, the beta control, the pseudo-IPO placebo, era and mapping sweeps, the borrow-swept mirror trade, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cef_ipo/`](cef_ipo/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
