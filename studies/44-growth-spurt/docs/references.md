# References & literature map — Study 44 (Growth-Spurt)

## The effect and its source

- **Cooper, M., Gulen, H., & Schill, M. (2008).** *Asset Growth and the Cross-Section of Stock
  Returns.* Journal of Finance 63(4), 1609–1651 — the foundational asset-growth-effect paper: firms
  that grow total assets fast subsequently underperform.
- **Fama, E., & French, K. (2015).** *A Five-Factor Asset Pricing Model.* Journal of Financial
  Economics 116(1) — the **investment (CMA)** factor that largely *subsumes* the asset-growth anomaly,
  the central caveat to any standalone version.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Asset Growth Effect"* (listed Sharpe `0.835` — the highest on the open list), with a QuantConnect
  implementation. Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why the caveats matter

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies.* Review of Financial Studies 33(5)
  — many anomalies, asset growth included, weaken sharply with micro-caps excluded and proper
  breakpoints; the effect concentrates in small/illiquid names.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — post-publication decay.

## Data

- **SEC EDGAR** — `us-gaap:Assets` from 10-K filings (`data.sec.gov/api/xbrl/companyconcept`), fiscal-
  year-end total assets, for current S&P 500 members. **Yahoo! Finance** — annual total returns.
  Two honest caveats baked into the verdict: the universe is *current* members (**survivorship bias**)
  and **all large-cap**, while the effect is documented to live in small/micro caps. The offline
  synthetic panel injects a known growth penalty (and a null) to exercise the machinery without the network.

*Siblings: [40 Paper-Tiger](../../40-paper-tiger/), [41 Hangover](../../41-hangover/),
[42 Last-Call](../../42-last-call/), [43 Free-Lunch](../../43-free-lunch/) — the headline-vs-benchmark bench.*
