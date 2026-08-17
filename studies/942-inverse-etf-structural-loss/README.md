# Study 942 — The Inverse Tax 🔻

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Is an inverse ETF really a worse way to be short than shorting the index yourself?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the "inverse tax" real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | The gap is real and precisely measured — SH beats a directly-short book by **+1.06 %/yr** (HAC *t* = **+5.10**, i.i.d. *t* +2.10, bootstrap CI [+0.71, +1.45]), replicated on PSQ (+0.58, *t* +2.50) and SDS (+1.19, *t* +3.78) — but **its sign is a choice, not a measurement**. Charge the fund the dividends it does not owe and the same tape reads **−0.79 %/yr (*t* −2.76)** on SH, −2.51 (*t* −4.87) on SDS. On top of that the headline needs an off-tape rebate credit below **1.72**, dies in ZIRP, and is not era-robust (2007-2015: +0.16 %/yr, *t* +0.39). |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | An implementation choice on a hedge sleeve, not a return stream: both arms lost 11-12 %/yr absolute while the index quintupled. Worth ~1 pt/yr at retail terms and **+3.15 %/yr** when bills pay 3.3%, but it evaporates at zero rates, reverses for a prime broker, reverses again under the same-mandate framing, and the −2× fund still carries a −2.21 %/yr wrapper residual. |

> **In one sentence:** The three charges the folklore levies on inverse ETFs do not add up as stated — the expense ratio is the *smallest* term, the daily-reset drag belongs to constant leverage rather than to the wrapper (a self-managed −1× book pays it identically), and the financing-and-dividend leg is genuinely two-sided rather than one-way traffic.

> **One honesty note, up front.** The headline race is *product vs product*: buy SH, or short SPY shares. Those two are **not the same mandate** — SH tracks the **price** index, a share-short owes the **total return** — and that dividend term (1.85 %/yr on SPY, 3.70 at −2×) is the single largest piece of the gap. Remove it and the sign flips, significantly. Both framings are defensible; the study reports both, and that is exactly why Signal is **Mixed** rather than Real or None.

## What we tested

Race each inverse fund — **SH** (−1× S&P 500), **PSQ** (−1× Nasdaq-100), **SDS** (−2× S&P 500)
— against the **honest alternative**: a directly-short book rebalanced daily to the same −k×
NAV that **owes the index's dividends** (total-return leg), pays a stock-loan fee and collects
whatever rebate its broker credits, with the rebate modelled off **^IRX**. One execution lag
on both arms. We take the HAC *t* on the daily gap (and publish the whole lag profile, +2.10 to
+5.50, because the gap's lag-1 autocorrelation is −0.44), bootstrap it, decompose it into
dividends / financing passthrough (γ measured by regression, not assumed) / expense ratio /
residual, run the **same-mandate** counterweight, cut it by era **and by rate regime**, sweep the
two PROXY inputs (rebate credit 0-2, borrow 0-100 bps) and quote the **break-even credit** at
which the answer flips. Survivorship is named, not fixed: these are three funds that survived.
**Dedup:** distinct from **61-slow-burn** and **100-melting-ice** (3× *long* decay priced against
the index, not against a self-managed book), **941-double-short-leveraged-pair** (harvesting the
decay as a trade), **943-leverage-reset-frequency** (varying the reset, which we hold fixed),
**945-leverage-financing-cost** (the same financing question on the *long* side, where the sign
does not flip) and **914-sec-lending-offset** (the lender's side of the same stock-loan market).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the three charges the story levies, which one survives, why dividends and short rates decide it, and the one number that tells you which side you are on |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the replicate's accounting, the HAC lag profile and block-bootstrap CI, the γ regression and its spec sensitivity, the same-mandate counterweight, the break-even credit, the rate-regime cut, both proxy sweeps, the path-drag table and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`inverse_tax/`](inverse_tax/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
