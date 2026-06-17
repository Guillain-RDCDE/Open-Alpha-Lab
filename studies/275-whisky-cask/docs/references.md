# References & literature map — Study 275 (Whisky-Cask)

## The claim under test

The retail "cask investment" pitch — *rare whisky is an uncorrelated, low-volatility
alternative asset that beat the stock market through the 2010s* — is built almost
entirely on **appraisal-based annual indices**:

- **Knight Frank, "The Wealth Report" / Luxury Investment Index (KFLII), annual.**
  Knight Frank's rare-whisky sub-index reported whisky as the best-performing luxury
  collectible of the 2010s (on the order of +500–600% over the decade), and later
  reported it *falling* (~−9% in 2023) as the market cooled. The numbers most often
  quoted in cask-broker marketing trace back to this index.
- **Rare Whisky 101, "Apex 1000" and "Icon 100" indices, annual commentary.** Basket
  indices of the rarest collectable bottles, rebased and published once a year. RW101's
  own commentary has repeatedly cautioned that these are bottle (not cask) indices and
  that the secondary market turned down sharply after 2021.

Both are **appraisal/auction-hammer baskets reported annually** — the cadence that
manufactures the flattering low volatility this study dismantles.

## Why the headline looks so good — and why it isn't real

- **Appraisal smoothing (the central critique).** An index marked once a year and
  anchored to the prior mark is serially autocorrelated by construction, which biases
  its measured variance *downward* and its Sharpe *upward*. This is the same artifact
  that inflated reported Sharpe ratios for private real estate and other illiquid
  assets for decades. The fix — first-order unsmoothing — is **Geltner (1991)**.

- **The cost wedge.** The index is a seller-of-rare-bottles benchmark. A cask buyer
  pays a large entry markup (often 30–50%), annual bonded-warehouse storage and
  insurance, the Angels' Share (~2%/yr evaporative loss), and an exit commission. None
  of these appear in the index, and together they roughly halve the realised return.

- **Survivorship.** A "rare whisky" index selects the bottles that became sought-after
  *ex post*. Distilleries that closed, casks that spoiled, and bottlings that never
  appreciated are absent. An individual cask is far more exposed to that left tail than
  the curated index implies.

- **Mis-selling backdrop.** The UK FCA and the Advertising Standards Authority issued
  warnings about cask-investment schemes in 2023–2024, several brokers collapsed, and
  HMRC tightened guidance — real-world confirmation that the gap between "index" and
  "what a buyer keeps" is large and often abusive.

## Method lineage

- **Geltner, D. (1991).** "Smoothing in Appraisal-Based Returns." *Journal of Real
  Estate Finance and Economics*, 4(3), 327–345. The canonical first-order unsmoothing
  estimator `r_true_t = (r_t − ρ·r_{t−1}) / (1 − ρ)` used in this study's
  `geltner_unsmooth`.
- **Getmansky, M., Lo, A. W. & Makarov, I. (2004).** "An Econometric Model of Serial
  Correlation and Illiquidity in Hedge Fund Returns." *Journal of Financial Economics*,
  74(3), 529–609. Generalises the smoothing critique to any illiquid/appraised asset:
  reported smoothness inflates Sharpe and hides true volatility.
- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703–708. The HAC standard error behind `hac_tstat` for the excess-return test.
- **Asness, C., Krail, R. & Liew, J. (2001).** "Do Hedge Funds Hedge?" *Journal of
  Portfolio Management*, 28(1), 6–19. Shows how lagged/illiquid marks understate market
  exposure and overstate risk-adjusted performance — directly analogous to casks.

## Data sources

- **S&P 500 ^GSPC.** Daily close from the repo-level cache
  `_cache/^GSPC_split_only.parquet`; December/December calendar-year **price** returns,
  2009–2024 (price-only, no dividends, to match the income-free whisky index).
- **Whisky index.** Hardcoded annual appraisal-based series in `data.py`, a desk
  reconstruction of the publicly reported KFLII rare-whisky and RW101 Apex 1000
  trajectories (the underlying indices are proprietary and not redistributed here).

## Related desk studies

- **[Study 68 — All-Weather](../../68-all-weather/)** and the broader alt-asset family —
  diversification claims tested honestly.
- The appraisal-smoothing lie detector here applies one-for-one to any
  appraisal-priced "alternative asset" pitched on a suspiciously high Sharpe (fine art,
  farmland, private real estate, collectibles).
