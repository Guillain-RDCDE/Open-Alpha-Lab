# Sources & literature map — Study 35 (Contango)

## The claim's source

- **Z. Kakushadze & J. A. Serur (2018), *151 Trading Strategies*, §9.1 (roll yields) and §9.4 (value /
  carry in commodities).** The catalogue entries for the commodity term-structure carry: a long futures
  position earns a roll yield as it slides along the curve (positive when backwardated, negative when
  contangoed), and ranking the cross-section by that carry is a documented commodity strategy. SSRN
  [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). *(Copyrighted; not redistributed.)*

## The carry / roll-yield premium — real, and why

- **Gorton, G. & Rouwenhorst, K. G. (2006), "Facts and Fantasies about Commodity Futures," *Financial
  Analysts Journal* 62(2).** The foundational study of commodity-futures returns: the equal-weight basket
  earns an equity-like risk premium, and the return is dominated by the **roll yield**, which is positive
  for backwardated and negative for contangoed contracts.
- **Erb, C. & Harvey, C. (2006), "The Strategic and Tactical Value of Commodity Futures," *Financial
  Analysts Journal* 62(2).** Shows that the cross-section of commodity returns is explained far more by
  the **term structure (roll return)** than by spot price appreciation — the direct basis for a
  long-backwardation / short-contango carry book.
- **Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018), "Carry," *Journal of Financial
  Economics* 127(2).** Generalises carry across asset classes; documents a robust commodity carry premium
  and shows carry is lowly correlated with momentum and value — the basis for the beat-7 carry+momentum
  blend.

## The honest counter — why `FRAGILE`

- **Carry crashes & volatility.** Like FX carry, commodity carry is a volatile, crash-prone premium that
  unwinds in commodity-wide risk-off (the 2008 and 2014–15 routs). The premium is concentrated and the
  drawdowns deep — the reason tradability is `FRAGILE`, not `INVESTABLE`. The cross-asset carry-crash
  reading is in Koijen et al. (2018) and, for FX, Brunnermeier–Nagel–Pedersen (2008) (see
  [Study 27](../../27-steamroller/)).
- **Capacity & the liquid-contract constraint.** Carry is strongest in the smaller, less-liquid
  contracts; the deeply liquid ones (crude, gold) carry less of it — the same illiquidity tension as
  [Study 33 (Slingshot)](../../33-slingshot/), and a pre-registered mirage check for the real run.

## The desk's sibling studies

- **Study 27 — Steamroller (§8.2, FX carry)**, [`../../27-steamroller/`](../../27-steamroller/). The
  *currency* carry trade: same long-high / short-low cross-sectional carry book, a real premium with a
  steamroller crash tail. Contango is its commodity sibling — the same `REAL`-signal / crash-prone story,
  measured here on the real energy tape (front vs laddered ETF pairs).
- **Study 29 — Hedgers-Toll (§9.2, commodity COT hedging pressure)**,
  [`../../29-hedgers-toll/`](../../29-hedgers-toll/). The *other* commodity-futures premium: the
  hedging-pressure / COT signal. Roll yield (Contango) and hedging pressure (Hedgers-Toll) are the two
  classic term-structure-linked commodity premia and are theoretically related (the Keynesian
  normal-backwardation story links the curve shape to hedger positioning).

## The data constraint (house rule: stated in the open)

- **Roll yield needs the term structure** — the slope between the **front** and a **deferred** contract, so
  it cannot be read from a single front-month series. Rather than a paid futures-curve feed, the desk
  observes it on the real tape through **front-month vs 12-month-laddered ETF pairs** on the same underlying
  (WTI USO/USL, gas UNG/UNL): the laddered fund sits further out the curve, so `laddered − front` *is* the
  realized roll. Liquid, key-free, clean yfinance history — no FRED, no EIA. The cross-sectional 12-name
  bucket book is the synthetic **machinery proof**; the real energy run is in [`results.md`](results.md).

## The shared method

- **Newey–West (1987)** HAC SEs · **Lo (2002)** Sharpe inference · **White (2000)** Reality Check — the
  shared [`quantlab/`](../../../quantlab/) engine; see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
  Reproducibility via [`quantlab/repro.py`](../../../quantlab/repro.py).

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
