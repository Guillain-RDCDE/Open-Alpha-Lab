# Study 874 — IPO-Price Anchoring ⚓

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the gap from the IPO offer price predict drift (and is below-offer a drag)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Neither leg survives. The Fama-MacBeth **anchoring-pull** slope (forward market-adjusted return on `gap = log(price/offer)`) is **−0.0023** (Newey-West *t* = **−0.39**; permutation two-sided *p* = 0.61) — the right sign, no significance. The **below-offer drag** looks loud at **−56.8 bps/mo** (below basket +13 vs above +70) but **NW *t* = −0.56**, same sign in both sub-eras yet insignificant in each. A clean 20-seed synthetic control fires on **0/20** nulls and **100%** on a planted pull, so the flatness is real, not machinery. *Curation bias (a small, one-dominant-cohort curated set) named on the Signal axis.* |
| **Tradability** — can you get paid for the drag? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | SHORT below-offer / LONG above-offer earns **+56.8 bps/mo gross** at *t* = **+0.56** (a coin-flip); realistic recent-IPO borrow (3–5%/yr) + 10–20 bps costs push net to **+11.8 → −24.8 bps/mo** (*t* ≈ +0.12 → −0.24) on a **~60% drawdown**. No edge to harvest. |

> **In one sentence:** the offer price *feels* like an anchor — below-offer names really did
> trail above-offer names by a chunky-looking ~57 bps/mo — but on a curated sample of ~45
> famous recent IPOs neither the anchoring pull nor the below-offer drag clears |t| ≥ 2, so the
> honest read is **claimed signal absent (underpowered, one-cohort), paycheck a mirage**.

## What we tested

The behavioural claim is that investors **anchor** on an IPO's **offer price**: a stock stretched
far above its round offer number should get pulled back down and one below it pulled back up
(**anchoring pull**), and crossing **below** the offer — the cohort's collective cost basis —
should be a **persistent drag** (loss-aversion / disposition lore). We hard-code a **curated
table of 44 well-known recent US listings** (offer / reference price + first-trade date, from SEC
424B4 prospectuses and exchange reference notices — public record) and pull their **yfinance
daily closes vs `SPY`** (2014-01 → 2026-06-30). Every forward return is **market-adjusted** (name
− SPY), so generic IPO-cohort underperformance is netted out. Two honest tests, each collapsing a
calendar month to one number before a Newey-West *t* (the names are one heavily time-correlated
cohort): a monthly **Fama-MacBeth** slope of forward abnormal return on the gap-from-offer, and a
**below-offer minus above-offer** basket spread — plus a 1,000-permutation placebo, a two-era cut,
a costed short-below/long-above timer, and a 20-seed synthetic positive control. One documented
lag (gap at close of `t` → hold `t+1`); as-of **2026-06-30**. **Dedup:**
[219-ipo-pop](../219-ipo-pop/) is the **first-day pop**, not the aftermarket anchor;
[265-ipo-volume](../265-ipo-volume/) is the **issuance-calendar** timing signal, not a per-name
anchor; [623-ipo-long-run-underperformance](../623-ipo-long-run-underperformance/) asks whether
IPOs **as a class** underperform (which we market-adjust *away* here);
[783-ipo-deal-of-year](../783-ipo-deal-of-year/) is the single-name **headline-deal** story, not a
systematic offer-price cross-section.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the round offer price *feels* like an anchor — and why a loud-looking −57 bps/mo below-offer gap is still a shrug once you see its *t* |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Fama-MacBeth anchoring slope, the below-offer basket spread, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ipo_anchor/`](ipo_anchor/). Curated public-record anchor table + yfinance daily closes
vs SPY, cached under `_cache/`. Small one-cohort curated cross-section → low power → the honest
prior is None; the synthetic control only proves the machinery is unbiased. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
