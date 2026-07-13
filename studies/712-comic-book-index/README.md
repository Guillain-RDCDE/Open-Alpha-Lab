# Study 712 — CGC-graded comics are an asset class? 📚

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do graded key comics beat the S&P? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Over 2018–2025 the (cited, approximate) comic index returned **+6.2%/yr** vs **+17.2%/yr** for SPY. Mean annual excess **−11.3%/yr** (*t* = −1.79, short of \|*t*\| ≥ 2). The one listed proxy shows **no** alpha (\|*t*\| < 1) *and* a losing CAGR. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even the gross index CAGR goes **negative (−5.2%/yr)** after a ~25% dealer/auction spread + a 2% CGC grading fee + 1%/yr carry. The only listed proxy, `FNKO`, **lost money** (−8%/yr, −88% drawdown, β≈1.4). Illiquid, high-carry, round-tripped 2022–24. |
| **Comics beat the S&P?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On return, drawdown **and** net-of-cost, the S&P wins every column. The winners bought a specific key before 2021 and sold near the top — bubble survivors, not an asset class. |

> **In one sentence:** the "CGC-graded key comics are an asset class that beats stocks" claim fails on its own tape — the graded-comic index lagged the S&P by ~11%/yr, the speculative middle round-tripped −20% off its early-2022 top, there is essentially **no** listed way to own the trade (the nearest proxy, Funko, lost money with an −88% drawdown), and a realistic grading-fee + dealer-spread haircut turns the comics' gross return **negative**.

## What we tested

Real graded-comic price indices (the **GoCollect** indices, **Heritage Auctions** realised prices) are **paywalled or per-lot archives, not freely API-available**, so we are transparent about it: we (a) hardcode a small, **clearly-cited, approximate** annual price-index series — base 100 @ 2018, anchored on public reporting of the 2020–21 pandemic melt-up, the **early-2022** blow-off top, the 2022–23 giveback, and the 2024–25 blue-chip stabilisation (record slabs: **Amazing Fantasy #15 CGC 9.6 at ~$3.6M**, 2021; **Action Comics #1 at ~$6.0M**, 2024) — and (b) test the **only listed** way to own the trade, honestly named: there is essentially **none** — CGC's/PSA's parents and Heritage are private/delisted, so the nearest proxy is **Funko (`FNKO`)**, benchmarked against **`SPY`** on CAGR, volatility, drawdown, alpha — and the part the pitch never charges, the **CGC grading fee + dealer/auction spread + illiquidity carry**. The comic index is a *labelled proxy*, never presented as the real index. (Same collectible-as-asset-class shape as [Study 358 — Watches](../../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "comics beat stocks" story feels true, the chart where the bubble round-trips, and where the grading fee + spread eat the return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | index vs SPY (CAGR/vol/MDD + annual-excess *t*), Newey-West proxy alpha, the grading + spread haircut on NAV, and a synthetic bubble positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`comic_book_index/`](comic_book_index/). Comic index is a **hardcoded, cited, approximate proxy** — not a live feed; the equity ticker is a **labelled proxy** for the trade, not the resale price of a slab. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
