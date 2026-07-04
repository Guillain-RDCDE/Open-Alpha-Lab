# Study 616 — Muni-CEF-Tax-Loss 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — December dump, January snap-back? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the January snap-back · None on the December-month dump.* January excess over MUB is **+191 bps** at one-sample *t* = **2.27** (19 winters, hit 84%), **+180 bps at *t* = 3.51** on the 26-winter VWLTX tape, exact 132-month-pair placebo *p* = 0.030/0.045. The December calendar month itself is flat (**−16 bps, *t* = −0.33**) — the dump and the start of the recovery net out inside December. **Survivorship** named (12-survivor panel) and a named **fade** (post-2016 January = +41 bps, *t* = 0.45; fade difference Welch *t* = 1.83, not itself certified). |
| **Tradability** — can you bank the swap? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The MUB→basket January swap nets **+171 bps/winter at *t* = 2.03** at 5 bps one-way, but slips under the bar at a realistic 10 bps (*t* = 1.79); the post-2016 gross (+41 bps) sits *below* the 40–100 bps cost drag; it fires **once a year** on tiny, thin-volume funds. Real historically, barely bankable now. |
| **"Buy Dec-15, not Jan-15"?** | ![Mixed](https://img.shields.io/badge/Buy_Dec--15%2C_not_Jan--15%3F-Mixed-8b949e?style=flat-square) | Nearly the whole seasonal payoff lives **between Dec-15 and Jan-15** (**+234 bps** paired difference, hit 74%; a Jan-15 entry keeps only +33 bps to end-Feb) — but certification is borderline (*t* = 1.78 full sample; **+114 bps at *t* = 2.02** ex-2008). Directionally confirmed, statistically not sealed. |

> **In one sentence:** the muni-CEF tax-loss season is half real — the **January snap-back** over the muni market clears the bar on both benchmarks (+191 bps, *t* = 2.27 on MUB; +180 bps, *t* = 3.51 on 26 VWLTX winters) and the payoff is front-loaded between Dec-15 and Jan-15, but the December *calendar month* never shows a certified dump, the effect has faded since publication (+41 bps post-2016), and a once-a-year swap on thin retail funds barely survives realistic costs — **Mixed, Fragile**.

## What we tested

Starks, Yong & Zheng (2006, JF): municipal-bond closed-end funds — held almost exclusively by **taxable retail investors** — get dumped for tax losses in December, widening their discounts, and snap back in January. Per-fund daily NAV isn't public, so we use a transparent **discount-motion proxy**: each fund's monthly **total return minus MUB** (the NAV-priced muni ETF; VWLTX, a NAV-priced muni mutual fund, extends the test to 2000). On a fixed **12-fund seasoned muni-CEF panel** we equal-weight into **one December and one January observation per winter** (killing the cross-fund pseudo-replication), run one-sample *t*'s across 19 (MUB) / 26 (VWLTX) non-overlapping winters, rank the Dec→Jan contrast among **all 132 ordered month pairs** (an exact, RNG-free placebo), split the sample at its midpoint for the post-publication fade, and charge the tradable January swap **4 one-way legs × 5/10/25 bps × NAV**. The third axis races a **Dec-15 entry against a Jan-15 entry** (common end-February exit) — the actionable version. A deterministic synthetic panel with plantable dump/snap knobs proves the machinery fires only when a season is truly there. This is the **seasonal-flow cousin** of the CEF discount **level** edge — see [367-closed-end-fund-discount](../367-closed-end-fund-discount/) (Real · Fragile). Survivorship named on the Signal axis. As-of **2026-06-30** (last complete month).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why tax-loss selling hits these odd little muni funds harder than anything else, what a discount snap-back looks like, and why the trade is real but hard to bank — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-winter basket one-sample *t*'s, the exact 132-pair seasonal placebo, the VWLTX 26-winter extension, the midpoint fade split, costs × 4 legs on the January swap, the Dec-15 vs Jan-15 paired race, and the planted-knob synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`muni_cef_tax_loss/`](muni_cef_tax_loss/). The discount proxy is fund-minus-benchmark **total-return excess** (benchmark prices at NAV); the primary unit is **one winter, one observation**. Panel is 12 **surviving** muni CEFs — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
