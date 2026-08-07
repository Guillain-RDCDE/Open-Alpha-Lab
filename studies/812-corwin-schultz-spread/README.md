# Study 812 — Corwin-Schultz Spread 📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-estimated-spread (illiquid) names earn a premium (Corwin-Schultz + Amihud-Mendelson)? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | The illiquidity premium **replicates, with the correct sign**, on 50 liquid US mega-caps. A long high-CS-spread / short low-CS-spread book earns **+4.45 bps/day** (Newey-West *t* = **+3.24**): the illiquid names *out-earned* the liquid ones (2010–2026). It is significant in **both** eras (*t* = +2.03 / +2.52), sits **+4.88σ** into the right tail of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* premium cleanly (estimator recovers the injected per-name spread at corr > 0.99). A rare green — the effect survives even on a mega-cap survivor panel, where an illiquidity effect should be *weakest*. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The gross premium is real but it lives **inside its own cost band**. At an idealised 1 bp one-way it is still net-positive (**+2.31 bps/day**, ~+5.8%/yr) but no longer significant (*t* = **+1.59**); at a realistic 5 bps it turns to **−5.69 bps/day** (*t* = −3.91). And the long leg *is* the illiquid names — where real execution spreads are widest — so the paycheck the premium promises is the same friction you pay to collect it. |

> **In one sentence:** the Corwin-Schultz high-low spread estimator recovers a genuine
> **illiquidity premium** even on liquid mega-caps — long illiquid, short liquid earns
> +4.45 bps/day, NW *t* = +3.24, robust across eras — but the edge sits entirely inside its
> own trading costs, so the honest read is **real signal, fragile paycheck**.

## What we tested

Corwin & Schultz (2012), **"A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low
Prices"**: the daily **high** transacts near the ask and the **low** near the bid, so the
high-low range embeds the spread — and comparing single-day squared log-ranges (`β`) with the
two-day log-range (`γ`) isolates it as `S = 2(e^α−1)/(1+e^α)` (negatives floored at 0). A high
estimated spread proxies **illiquidity**, so a long high-spread / short low-spread book is the
textbook illiquidity-premium bet. We take the self-contained daily version on a **liquid
50-name US cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: each
name's daily CS spread averaged over a trailing 21 days, sorted point-in-time (signal known at
the close of `t−1`, one shift, zero look-ahead), with a Newey-West *t* on the daily spread, a
1,000-permutation placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[377-bid-ask-bounce](../377-bid-ask-bounce/) tests the spread's short-horizon **mean-reversion
bounce**, not a cross-sectional illiquidity level; [140-amihud-illiquidity](../140-amihud-illiquidity/)
uses the **volume-based** price-impact ratio, whereas Corwin-Schultz needs no volume at all;
[811-zero-return-days](../811-zero-return-days/) counts **no-trade days**, not the high-low
range. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how a bid-ask spread hides inside the high-low range, and why the illiquid names quietly out-earned |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`corwin_schultz/`](corwin_schultz/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
