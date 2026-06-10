# Sources & literature map — Study 33 (Slingshot)

## The claim's source

- **Z. Kakushadze & J. A. Serur (2018), *151 Trading Strategies*, §3.9 — "Mean-reversion (single
  group)."** The catalogue entry for fading each name against the cross-section of its peers within one
  universe. SSRN `3247865` · arXiv [1912.04492](https://arxiv.org/abs/1912.04492). *(Copyrighted; not
  redistributed.)* See also §3.18 (the optimised stat-arb version, tested in Study 26 Sand-Castle).

## Short-term reversal — real in stocks, and why

- **Jegadeesh, N. (1990), "Evidence of Predictable Behavior of Security Returns," *Journal of Finance*
  45(3).** The foundational monthly reversal result in individual equities.
- **Lehmann, B. (1990), "Fads, Martingales, and Market Efficiency," *QJE* 105(1).** Weekly contrarian
  profits in stocks; the early liquidity-provision reading.
- **Lo, A. & MacKinlay, A. C. (1990), "When Are Contrarian Profits Due to Stock Market Overreaction?,"
  *Review of Financial Studies* 3(2).** Decomposes contrarian profit into own-/cross-autocorrelation
  and a cross-sectional variance term — the formal basis for the dollar-neutral book here.
- **Avramov, Chordia & Goyal (2006), "Liquidity and Autocorrelations in Individual Stock Returns,"
  *Journal of Finance* 61(5).** Short-term reversal concentrates in *illiquid* stocks and largely
  vanishes after trading frictions — the direct reason the break-even cost (3.31 bp) makes it a
  `MIRAGE`, and the contrast with deep futures (Study 32).

## The decay

- **Khandani, A. & Lo, A. (2007), "What Happened to the Quants in August 2007?," *Journal of Investment
  Management*** (and 2011 follow-up, *Journal of Financial Markets*). Documents the long decline of the
  contrarian/reversal premium as statistical-arbitrage capital and then HFT crowded in — the `CONFIRMED`
  decay this study measures (net Sharpe −0.85 in 2020–2026).
- **Nagel, S. (2012), "Evaporating Liquidity," *Review of Financial Studies* 25(7).** Returns to
  short-term reversal proxy the expected return on liquidity provision; rich in crises, thin otherwise.

## The contrast study

- **Study 32 — Rip-Tide (§10.3 reversion futures)**, [`../../32-rip-tide/`](../../32-rip-tide/). Same
  fade-the-move idea on deep liquid futures: gross Sharpe 0.08 (`NONE`). The pair localises the
  reversal premium to the single-stock cross-section, exactly as the liquidity theory predicts.
- **Study 26 — Sand-Castle (§3.18)** and **Study 19 — Rubber-Band (IBS)** — adjacent mean-reversion cuts.

## The shared method

- **Newey-West (1987)** HAC SEs · **Lo (2002)** Sharpe inference · **White (2000)** Reality Check —
  the shared [`quantlab/`](../../../quantlab/) engine; see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
