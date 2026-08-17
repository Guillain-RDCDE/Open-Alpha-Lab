# Study 918 — Creation Halt 🚧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Visible where the ruler is **exact**: VXX repriced **+15.4%** against VIXY in the 20 lagged sessions after its 2022 suspension and gave back **−18.5%** after issuance resumed — percentile **0.990 / 0.000 of 99 *independent* windows** (not the 1,978 overlapping ones), so against this design's **30 looks** neither tail clears alone; the **joint** pattern does. It does not pool: mean z **+1.81**, cross-event ***t* = +0.60**, **1/5 positive** at K = 20, sign flips with the horizon, and dropping VXX turns the pool negative (*t* = −2.17). USO's 2020 halt went the *wrong* way by −19% in six sessions. Hand-curated event list ⇒ selection/survivorship bias toward the spectacular, and it still cannot pool. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Held to the resumption date — **a date nobody knew at entry** — VXX nets +17.8%. On a **blind 60-session hold it nets +0.57%**: the whole profit was the hindsight exit, and the premium round-trips to −1.1% by day 40. Median per-event net **−4.29%** at 10 bps, daily rebalancing and 3%/yr borrow, 2/6 positive; **ex-GBTC the mean is −4.51%, 1/5 positive**. The decisive input — borrow on a hard-to-borrow capped ETP — is unobservable; at 30%/yr the mean net is **−27%**. |

> **In one sentence:** when an ETP stops issuing shares the arbitrage really does break and the price really can float free — we can watch it happen in the one suspension where the capped fund had an *exact* uncapped twin — but across six hardcoded suspensions the effect refuses to pool, the most famous one went violently the wrong way, and even the clean case pays nothing to a trader who is not told in advance when the halt will end.

## What we tested

Six hardcoded, publicly reported suspensions of share issuance — **UNG 2009**, **USO
2020**, **VXX** and **OIL 2022** (the same Barclays announcement, so *not* an independent
draw), **BITO 2021** (a CME position-limit capacity constraint, flagged SOFT) and **GBTC
2015–2024** (a *redemption* freeze, signed −1) — measured as the daily log-return spread
of the capped fund against an **uncapped** instrument tracking the same thing (VIXY, spot
bitcoin, DBO/USL/BNO, NG=F). One execution lag (announced day `t`, acted at `t+1`, so the
announcement-day move is excluded); the spread is self-financing, so the race is
excess-of-cash — an **identity, not a finding**, which is why the short leg is charged
borrow separately. Each K-day window is standardised against that pair's own non-event
windows, reported on the **non-overlapping** pool and against the design's **30-look**
family-wise bar; then a suspended-regime HAC drift **net of the two legs' expense-ratio
difference** and against a control that excludes the fade, a resumption fade, a
leave-one-event-out jackknife, a ruler-quality split, a ±10-day resumption-date sweep, a
**blind fixed-horizon exit** that uses no resumption date, and a cost × borrow sweep with
a daily rebalancing charge. Non-tape **ASSUMPTIONS** — the event list, the APPROX
resumption dates, the expense ratios, the 3%/yr borrow, the 10 bps cost — are labelled
and swept. **Dedup:** [618-gbtc-premium-cycle](../618-gbtc-premium-cycle/) reconstructs
GBTC's premium path (we use its era as *one signed event*, not the subject);
[378-etf-nav-premium](../378-etf-nav-premium/) tests discounts while the creation channel
works *normally*; [367-closed-end-fund-discount](../367-closed-end-fund-discount/) is the
permanent-no-channel case; [661](../661-uso-roll-decay/), [375](../375-vxx-roll-decay/)
and [619](../619-bito-roll-drag/) measure these instruments' *roll* cost, which is exactly
the confound that ruins our curve-mismatched rulers; [917](../917-nav-staleness-timezone/)
is a clock-driven price-versus-value gap, not a suspended primary market. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what breaks when a fund stops printing shares, the one case you can see it in, the one that ate you alive, and why "buy the halted fund" is not a strategy |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | signed spreads, placebo-standardised CARs at K = 5/10/20 on overlapping and independent pools, the multiplicity bar, HAC regime drift net of fees, the fade, jackknife and ruler split, the date/blind-exit/cost × borrow sweeps, and the planted/null synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`creation_halt/`](creation_halt/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
