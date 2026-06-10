# Results — Study 37 (Barometer): cross-asset macro momentum & inflation hedging

> ⚠️ **Real run pending a reliable FRED macro fetch.** Unlike the desk's cached studies, this one's real
> tape is **FRED macro series** (growth: `INDPRO` / `PAYEMS`; inflation: `CPIAUCSL` / `T10YIE`) plus liquid
> asset proxies — and in this environment those series are **not reliably fetchable**: the daily FRED
> series (`T10YIE`, `DGS10`) reliably time out, and even the small monthly `CPIAUCSL` succeeds only
> intermittently. So, exactly like [Study 27 (Steamroller)](../../27-steamroller/), the verdict below is
> earned on a fully-validated **synthetic control** and the macro-momentum / inflation-hedging literature,
> and the real run is **PENDING a reliable FRED macro fetch as of 2026-06-10**:
>
> ```
> python examples/verify.py --fetch     # tries FRED macro + asset proxies; writes this file when it succeeds
> ```
>
> Until then the offline core is fully validated and reproducible via
> [`examples/run_synthetic_demo.py`](../examples/run_synthetic_demo.py). This is a **pre-registration**: the
> apparatus, the null, the regime split and the mirage-line are fixed *before* the real numbers exist.

## The verdict — Signal `WEAK`/`REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`

The *trend* in fundamental macro data is a real, slow, cross-asset predictor. Two threads:

1. **Macro momentum** (Brooks & Moskowitz 2017). Going long the assets favoured by *improving* macro
   momentum — growth-up lifts pro-cyclical assets (equities, commodities), inflation-up favours real
   assets — is a documented, diversifying premium across equities, bonds, commodities and FX. It is
   **real but modest and slow**: a stand-alone Sharpe in the ~0.4–0.8 range in the literature, with long
   flat stretches, which is why the desk stamps the signal `WEAK`→`REAL` and tradability `FRAGILE`.
2. **Inflation hedging** (Neville, Draaisma, Funnell, Harvey & van Hemert 2021). Tilting toward *real*
   assets (commodities, TIPS, gold, trend) when inflation is rising protects a portfolio in exactly the
   regimes nominal stocks and bonds suffer — but it is **episodic**: those regimes are rare (the 1970s,
   2021–22), so the tilt is dead weight or a drag most of the time and only earns its keep when inflation
   actually accelerates.

Both are slow books — cost is *not* the threat (break-even costs are high). The threats are the modest
stand-alone Sharpe, the long droughts, and the episodic, regime-dependent payoff of the inflation hedge.

## What the synthetic control proves (offline, reproducible)

On a synthetic cross-asset world (5 assets: equities, nominal bonds, commodities, a TIPS/real-rate proxy,
gold) driven by two latent, persistent, regime-switching macro state variables — *growth* and *inflation*
— whose **momentum** (one-month change, lagged) predicts next-month returns through fixed signed betas
(seed 37, 50 years, gross of cost):

- **Macro momentum is real and recovered:** the macro-momentum book earns **+5.1%/yr** at Sharpe
  **+1.09**, max drawdown **−12%**, on low turnover (**5.6×/yr**). On the `macro_strength = 0` **null**
  (assets are pure noise) it collapses to Sharpe **−0.17** — proving the apparatus measures the effect,
  not itself.
- **The inflation hedge pays — and is regime-dependent:** the inflation-tilt book earns **+2.3%/yr** at
  Sharpe **+0.55** (null **−0.02**), and the [beat-7 regime split](extension.md) shows it earns **more in
  rising-inflation regimes (Sharpe +0.59, +2.5%/yr) than in falling ones (+0.46, +1.9%/yr)** — it pays
  when it is supposed to, exactly the conditional behaviour the steelman claims.
- **Cost is not the threat:** the break-even cost is **~91 bp** (macro momentum) / **~60 bp** (inflation
  tilt) per unit traded — far above realistic cross-asset costs — so a slow macro book survives costs
  comfortably; the cost sweep degrades gracefully (Sharpe 1.09 → 1.03 → 0.97 → 0.78 at 0/5/10/25 bp).

## What `--fetch` will fill in (the pre-registered real run)

When a reliable FRED macro fetch is available, the real run will report, on real cross-asset proxies and
FRED growth/inflation series: the macro-momentum book's Sharpe and HAC *t*-stat, its turnover and
break-even, the inflation-hedge book's Sharpe, and — the headline — the **regime split** (does the
inflation hedge actually pay in the historical rising-inflation episodes: the 1970s, 2008's commodity
spike, 2021–22?). The expected shape, from the literature, is a macro-momentum Sharpe of roughly 0.4–0.8
with long flat stretches and an inflation hedge that is near-zero most of the time but strongly positive
in the inflationary windows — i.e. exactly the `WEAK`/`REAL` · `FRAGILE` · `PRE-REG` verdict the synthetic
control already earns.

*Sources & literature map: [docs/references.md](references.md). Engine: [`quantlab/`](../../../quantlab/).
**Not investment advice** — research & education.*
