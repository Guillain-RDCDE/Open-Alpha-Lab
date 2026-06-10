# References & literature map — Study 36 (Greenback)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entries are **§8.3 (dollar carry)** — the
  level of the average forward discount as a tradable USD-vs-basket tilt — and **§8.4 (combining
  momentum and carry)** — blending the FX carry and FX-momentum signals into a single book. *(Copyrighted;
  not redistributed.)*

## The claim under test — the steelman

- **Carry premium / forward-premium puzzle.** The empirical failure of uncovered interest-rate parity is
  one of the oldest in international finance (Eugene Fama, *"Forward and Spot Exchange Rates"*, **Journal of
  Monetary Economics** 1984). The cross-sectional carry portfolio earns a large, significant premium:
  Hanno Lustig, Nikolai Roussanov & Adrien Verdelhan, *"Common Risk Factors in Currency Markets"*,
  **Review of Financial Studies** 2011 — which also introduces the **dollar factor** (the average
  forward-discount level, i.e. the **dollar carry** of §8.3) as a second, distinct currency premium.
- **FX carry & global volatility.** Lukas Menkhoff, Lucio Sarno, Maik Schmeling & Andreas Schrimpf,
  *"Carry Trades and Global Foreign Exchange Volatility"*, **Journal of Finance** 2012 — the carry premium
  is compensation for doing badly when global FX volatility spikes (the crash).
- **FX momentum.** Menkhoff, Sarno, Schmeling & Schrimpf, *"Currency Momentum Strategies"*, **Journal of
  Financial Economics** 2012b — trailing FX trend is a separate, profitable currency signal.
- **The combo — carry ⊕ momentum.** Cliff Asness, Tobias Moskowitz & Lasse Pedersen, *"Value and Momentum
  Everywhere"*, **Journal of Finance** 2013 — momentum and value/carry are negatively correlated across
  asset classes, so combining them raises the Sharpe sharply. Ralph Koijen, Tobias Moskowitz, Lasse
  Pedersen & Evert Vrugt, *"Carry"*, **Journal of Financial Economics** 2018 — a unified definition of
  carry across assets and the case for combining it with momentum/trend. This **carry⊕momentum
  diversification** is the heart of Greenback's beat-7 complement.

## The honest counter — why `FRAGILE`, and the steamroller

- **Carry crashes.** Markus Brunnermeier, Stefan Nagel & Lasse Pedersen, *"Carry Trades and Currency
  Crashes"*, **NBER Macroeconomics Annual** 2008: carry returns are negatively skewed, and the crashes
  coincide with the sudden unwind of crowded, leveraged positions when funding liquidity dries up — a jump,
  correlated across all carry pairs at once. The combo dulls but does not erase this; the verdict is
  `FRAGILE`, not `INVESTABLE`.

## How this differs from — and builds on — Study 27 (Steamroller)

- **[Study 27 — Steamroller (§8.2)](../../27-steamroller/)** measured the **G10 carry premium itself**:
  is it real (yes), and can vol-targeting dodge its crash (no — the crash is a jump, not a vol build-up).
  **Greenback does NOT re-litigate that.** It takes the carry premium as given and asks the *next*
  question from the same book: the **dollar-carry tilt (§8.3)** and the **carry⊕momentum combo (§8.4)** —
  how combining the two premia *diversifies*, lifting the Sharpe above either standalone and cushioning the
  steamroller because carry and momentum pay at different times. The two studies are complementary: 27 is
  the premium and its crash; 36 is the dollar tilt and the combo that partially tames it.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference** (Newey & West, *Econometrica* 1987) on the books' means.
- **Data.** The real run uses monthly **G10 short rates from FRED** and **FX spot from yfinance**; pinned
  with [`quantlab.repro`](../../../quantlab/repro.py). The synthetic control bakes a partial-UIRP carry
  premium, a sticky two-state risk-off crash regime, and an independent autocorrelated trend for momentum.

## Caveats stated in the open (house rule)

- **Real run is PENDING one network fetch.** The carry signal needs FRED short rates, whose download
  **times out** in this environment; `examples/verify.py --fetch` populates it where FRED is reachable. The
  committed verdict rests on the fully-validated synthetic control and the literature until then —
  **Real-tape run? `PRE-REG`**.
- **Monthly horizon, USD base, spot (not forward) carry.** A stated simplification: spot FX plus the rate
  gap approximates the forward-based carry return and is the cleanly-available construction from public data.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
