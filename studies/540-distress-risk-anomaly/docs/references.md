# References & literature map — Study 540 (Distress-Risk-Anomaly)

## The claim, at full strength

- **Campbell, Hilscher & Szilagyi (2008)**, *"In Search of Distress Risk."* *Journal of Finance*
  63(6). The canonical statement of the **distress puzzle**: a dynamic logit failure model
  (profitability, leverage, equity volatility, market cap, cash, the price level and past excess
  return) sorts the cross-section, and the most-distressed firms earn *anomalously low* returns
  with a large negative alpha — the opposite of a risk premium. The measure this study proxies.
- **Dichev (1998)**, *"Is the Risk of Bankruptcy a Systematic Risk?"* *Journal of Finance* 53(3).
  The first clean statement that high-bankruptcy-risk firms (high Altman-Z / Ohlson-O distress)
  earn *lower*, not higher, returns — the empirical seed of the puzzle.
- **Griffin & Lemmon (2002)**, *"Book-to-Market Equity, Distress Risk, and Stock Returns."*
  *Journal of Finance* 57(5). The distress effect concentrates in the most-distressed,
  highest-Ohlson-O firms, especially small/illiquid names — and is *not* a value premium in
  disguise.
- **Vassalou & Xing (2004)**, *"Default Risk in Equity Returns."* *Journal of Finance* 59(2). The
  contrasting view: a Merton distance-to-default measure earns a *positive* premium in some
  cuts — the debate the puzzle sits inside, and why the sign matters.
- **Campbell, Hilscher & Szilagyi (2011)**, *"Predicting Financial Distress and the Performance of
  Distressed Stocks."* The follow-up confirming the low returns of distressed stocks out of
  sample.

## The distress measure we build

- The CHS failure probability is a logit of (among others) **NIMTA** (net income / market-adjusted
  total assets — *profitability*), **TLMTA** (total liabilities / total assets — *leverage*) and
  **SIGMA** (annualised equity volatility). This study standardises the three legs a no-key retail
  stack can build cleanly — leverage, ROA, trailing realised vol — into a composite *distress
  score* (higher = more distressed), and names the dropped legs (size, cash, price, excess return)
  as the simplification.

## Neighbours on this bench (the dedup map)

- **[Study 123 — Altman-Z](../../123-altman-z/)** — the Altman (1968) Z-**score**, a static
  five-ratio bankruptcy classifier. Study 540 is about the **return anomaly and its sign**, built
  from a CHS-style failure-probability proxy, not the Z classifier.
- **[Study 230 — Ohlson-O-score](../../230-ohlson-o-score/)** — the Ohlson (1980) O-**score**
  logit. Again a bankruptcy *score*; Study 540 tests the *distress-risk return puzzle* itself
  (the most-distressed names earning the lowest returns), on a decile/tercile sort.
- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)** /
  **[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)** — the low-risk
  anomalies. Distress overlaps high volatility, but the CHS puzzle is a *fundamental-distress*
  sort, not a pure volatility/beta sort.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the safe-minus-distressed
  bucket spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  distress labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (one-sample
  *t* ≥ 2 plus a placebo null and seed-robustness), the explicit survivorship caveat, one
  execution lag, and costs one-way × NAV with shorts paying borrow.
