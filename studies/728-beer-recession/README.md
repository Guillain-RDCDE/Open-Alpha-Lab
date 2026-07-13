# Study 728 — Is beer recession-proof? 🍺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — are beer stocks defensive / counter-cyclical? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No leg clears \|*t*\| ≥ 2 in the beer's favour: recession-window excess vs SPY is *t* = **+0.14** (TAP) and **+1.87** (SAM — and it's two idiosyncratic rallies); CAPM alphas are insignificant (NW *t* = +0.71, +0.84). Downside defensiveness holds for only **one** of the two names. |
| **Tradability** — could you harvest it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The NBER dates a recession **~12 months ex-post** (a look-ahead — you can't rotate in on time), and the buy-and-hold "beer hedge" turned **\$1 → \$9.19 / \$6.91** vs SPY's **\$27.45** over 31 years, with *deeper* drawdowns. |
| **Counter-cyclical?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The only real effect is *low beta* (TAP β≈0.64) — not counter-cyclicality: TAP fell **more** than SPY in 2 of 3 recessions, and SAM's down-beta (0.87) *exceeds* its up-beta (0.79). |

> **In one sentence:** "people drink more beer in a recession" doesn't reach the beer *stocks* — one of the two (Molson Coors) is a mildly *low-beta* name (which isn't the same as counter-cyclical), the other (Boston Beer) is *anti*-defensive on the downside, neither beats the S&P in recessions at any real significance, and the whole "hedge" would have cost you two-thirds of your money over 31 years.

## What we tested

The folklore — a staple of lifestyle-finance and "sin-stocks-are-defensive" commentary — says
**beer is recession-proof**: people cut the vacation before the six-pack, so a brewer holds up
(or beats the market) when everything else is falling. The testable version is specific: a beer
stock should have a **low downside beta** and **out-return the S&P during recessions**. We test
the two tradable beer pure-plays — **Molson Coors (`TAP`)** and **Boston Beer (`SAM`)** — against
**`SPY`** on month-end yfinance data (1995–2026), over the three **NBER** recession windows
(2001, 2008, 2020), on downside beta, Newey-West CAPM alpha, the recession-window paired *t*, and
the look-ahead the pitch never charges. The NBER dates are **hardcoded, cited** official dates —
labelled facts, never a live feed.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "recession-proof beer" feels true, the recession-by-recession bars where it isn't, and the 3× wealth gap to the index — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | bull/bear beta, Newey-West CAPM alpha, the recession-window *t* with its per-recession provenance, the NBER look-ahead, and a planted-asymmetric-beta positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`beer_recession/`](beer_recession/). Prices are real (yfinance, split+dividend adjusted — *total-return-ish*); the NBER recession windows are **hardcoded, cited official dates**, not a live feed. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
