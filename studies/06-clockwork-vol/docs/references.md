# References & literature map — Study 06 (Clockwork-Vol)

The study sits at the intersection of three literatures: **cycle theory** (the claim), the
**spectral-significance** toolkit that adjudicates it, and the **VIX dynamics** literature
that explains what the VIX *actually* does instead of running on a clock.

## The claim — cycle theory / Hurst cycles
- **J. M. Hurst, *The Profit Magic of Stock Transaction Timing* (1970).** The origin of the
  fixed-period "nominal cycle" model traders apply to price (and, here, to the VIX): swings
  decompose into harmonically related cycles of roughly constant length whose lows can be
  projected forward. The steelman the study tests.
- **Namzes Cycles ([@Namzes_G](https://x.com/Namzes_G)), the prompting thread (May 2026).**
  The specific, datable claims: a VIX 80-day cycle low "formed May 29", nested 40-day cycles
  cresting late July, synced to a stock 20-week low and a 4-year peak. The full-strength
  version we take apart.
- **S. Benner, *Benner's Prophecies of Future Ups and Downs in Prices* (1875).** The genre's
  ancestor — fixed-period business-cycle forecasting — included for the historical pattern:
  every generation rediscovers the clock.

## The method — is a spectral peak real, or red noise?
- **Mann & Lees (1996), "Robust estimation of background noise and signal detection in
  climatic time series," *Climatic Change* 33.** The canonical robust **AR(1) red-noise
  null** against which a periodogram peak must be judged — the exact test this study runs.
- **Torrence & Compo (1998), "A Practical Guide to Wavelet Analysis," *Bull. Amer. Meteor.
  Soc.* 79.** The red-noise significance envelope and the warning that autocorrelated series
  manufacture significant-looking peaks; the source of the Monte-Carlo envelope here.
- **Schuster (1898); Fisher (1929).** The periodogram and **Fisher's g-test** for a hidden
  periodicity in noise — the classical statement that finite records grow spurious peaks.
- **Ferrara & Guégan**, work on **pseudo-cycles** in finance: apparent periodicities that
  vanish out-of-sample, the empirical backdrop to the null verdict.

## What the VIX actually does (the alternative to a clock)
- **Whaley (2009), "Understanding the VIX," *Journal of Portfolio Management*.** The index
  construction and its core behaviour — **mean reversion** and **volatility clustering**, not
  periodicity.
- **Bollerslev, Tauchen & Zhou (2009)** on the **variance risk premium**; the VIX's
  predictable component is a risk-premium/mean-reversion story, not a fixed cycle — the
  honest source of any "VIX is forecastable" intuition.
- **Cont (2001), "Empirical properties of asset returns: stylized facts."** Long-memory
  volatility (slow, persistent autocorrelation) is precisely the *red-noise* structure that
  fakes cycles to the eye — the thing the AR(1) null is built to absorb.

## In-repo
- The shared desk method: [`../../../METHODOLOGY.md`](../../../METHODOLOGY.md).
- Reproducibility stamp (as-of + fingerprint): [`quantlab/repro.py`](../../../quantlab/repro.py).
- Bootstrap Sharpe CI: [`quantlab/stats.py`](../../../quantlab/stats.py).
- The VIX as a *trigger* (the complementary question): [Study 03 — Fear-Gauge](../../03-fear-gauge/).
