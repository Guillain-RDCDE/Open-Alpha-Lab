# References & literature map — Study 34 (Aftershock)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §3.2 (earnings
  momentum / post-earnings-announcement drift, PEAD)** — rank names by their earnings *surprise* and hold
  a long-positive / short-negative book, because prices keep drifting in the surprise's direction after
  the announcement. *(Copyrighted; not redistributed.)*

## The claim under test — the steelman

- **Ray Ball & Philip Brown (1968), "An Empirical Evaluation of Accounting Income Numbers," *Journal of
  Accounting Research* 6(2).** The original observation that prices continue to drift in the direction of
  the earnings surprise *after* the announcement — the birth of PEAD.
- **Victor Bernard & Jacob Thomas (1989), "Post-Earnings-Announcement Drift: Delayed Price Response or
  Risk Premium?," *Journal of Accounting Research* 27.** The definitive modern treatment: sort on
  **standardised unexpected earnings (SUE)**, and the top-minus-bottom decile keeps drifting for ~60
  trading days. The drift-decay curve in this study's beat 7 is their Figure 1.
- **Bernard & Thomas (1990), "Evidence that Stock Prices Do Not Fully Reflect the Implications of Current
  Earnings for Future Earnings," *Journal of Accounting & Economics* 13(4).** The under-reaction
  mechanism: investors fail to fully incorporate the autocorrelation in earnings news.

## The honest counter — why tradability is `FRAGILE`

- **Tarun Chordia & Lakshmanan Shivakumar (2006), "Earnings and Price Momentum," *Journal of Financial
  Economics* 80(3).** Earnings momentum and price momentum are related; the earnings-surprise drift
  largely subsumes price momentum, but the tradable residual is modest.
- **Chordia, Goyal, Sadka, Sadka & Shivakumar (2009), "Liquidity and the Post-Earnings-Announcement
  Drift," *Financial Analysts Journal* 65(4).** PEAD **concentrates in illiquid, small, high-cost names**
  and is much weaker — often gone net of trading frictions — in the liquid stocks you can actually trade
  at scale. This is the core reason the desk stamps Tradability `FRAGILE`.
- **Decay / shrinkage.** Like most documented anomalies, PEAD has attenuated since publication as it was
  arbitraged (consistent with McLean & Pontiff 2016, "Does Academic Research Destroy Stock Return
  Predictability?," *Journal of Finance* 71(1)). Real and durable, but smaller than its 1980s heyday.

## The desk's own method — engine and reproducibility

- **HAC / Newey-West inference** (Newey & West, *Econometrica* 1987) on the book's mean return; **Lo
  (2002)** Sharpe inference; **White (2000)** Reality Check — the shared
  [`quantlab/`](../../../quantlab/) engine; see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
- **Reproducibility.** Headline runs are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (as-of date + content fingerprint).

## Caveats stated in the open (house rule)

- **Real run is pre-registered and pending an earnings-history fetch.** A credible PEAD cross-section
  needs *years* of reported-earnings dates + surprises per name. No free source supplies that here:
  yfinance exposes only ~6-8 reported quarters, and there is no reliable long free surprise history (the
  same data wall the desk hit for options open-interest). So the committed verdict rests on the
  fully-validated synthetic control and the long-run literature; the real measurement is wired as a stub
  in [`examples/verify.py`](../examples/verify.py) (`--fetch`) and documented as pending in
  [`docs/results.md`](results.md).
- **SUE, total-return closes, daily horizon.** The surprise is a *standardised* unexpected-earnings
  z-score (PEAD is a SUE statement, not a raw-EPS-miss one); returns are split/dividend-adjusted; the
  book trades on a one-day lag — all stated decisions, not details.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
