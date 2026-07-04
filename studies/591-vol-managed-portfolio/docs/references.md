# References & literature map — Study 591 (Vol-Managed Portfolio)

## The claim under test

- **The seminal paper.** Alan Moreira & Tyler Muir, *Volatility-Managed Portfolios*
  (2017, **Journal of Finance** 72(4), 1611–1644). Their headline: portfolios that scale a
  factor's exposure by the **inverse of its previous month's realized variance**
  (`w ∝ c/RV`) earn large positive alphas on the original factor, raise Sharpe ratios, and
  shrink exposure exactly when crashes are most likely. For the market factor they report an
  annualised alpha on the order of **~4.9%/yr** (1926–2015, no cap). The mechanism: **variance
  is highly forecastable at the monthly horizon while the conditional equity premium barely
  moves with it** — so cutting exposure after volatile months sacrifices little mean and dumps
  a lot of variance.
- **The rebuttal literature (why the desk expects a fight).** Cederburg, O'Doherty, Wang &
  Yan, *On the performance of volatility-managed portfolios* (2020, **JFE** 138(1), 95–117):
  out-of-sample and in real time, the alpha survives for the **market factor** but fails for
  most of the other factors, and direct trading-strategy implementations struggle. Barroso &
  Detzel (2021, JFE) find transaction costs and leverage constraints erode much of the gain.
  Our SPY/QQQ-clear, EFA/IWM-fail split and the net-of-borrow decertification are exactly this
  contested middle ground.
- **The mechanism's cousins.** Barroso & Santa-Clara (2015, JFE), *Momentum has its moments* —
  the same 1/RV scaling rescues momentum from its crashes. French, Schwert & Stambaugh (1987)
  and Glosten-Jagannathan-Runkle (1993) on the weak/negative contemporaneous risk-return
  trade-off — the "vol-return disconnect" that makes the strategy possible.

## What we measure, and the honest conventions

- **The rule, past-only.** `w(m+1) = min(1.5, c_m/RV_m)`, where `RV_m` is the sum of squared
  daily returns of month *m* and `c_m` is the **expanding mean** of monthly RV up to *m*.
  Moreira-Muir choose *c* ex-post to match the unmanaged portfolio's full-sample variance —
  harmless for a *t*-stat, but a look-ahead for a tradable weight; our expanding normaliser
  targets an average weight near 1 using **only past data**. Cap at 1.5× per the desk plan
  (uncapped weights reach 5–10× in calm regimes — unrealistic for retail margin).
- **Exactly one execution lag.** The month-*m+1* weight uses only month-*m* (and earlier)
  daily returns, all known at the month-*m* close where the position is set.
- **Excess-vs-excess.** Both legs subtract the same 13-week T-bill rate (^IRX/100/252 daily —
  a bank-discount shortcut worth <2 bps/yr, identical on both legs so it cancels in the race).
  Leverage above 1 finances at that rate inside `w × excess`; the **retail borrow spread**
  (1–2%/yr above bills) is charged separately on `max(w−1,0)` in the cost sweep.
- **HAC inference.** The Moreira-Muir test is a regression of managed on unmanaged monthly
  excess returns with **Newey-West (1987)** standard errors (Bartlett kernel, rule-of-thumb
  lags); we also report the appraisal ratio (Treynor & Black 1973). Sharpe races are
  excess-vs-excess.
- **Placebo.** 200 seeds of a **shuffled-RV** placebo (permute the monthly RV series, rebuild
  the entire rule per seed): destroys the timing information while keeping the weight
  distribution — the desk's random-baseline rule (never single-seed).

## Why the synthetic control has three worlds

- **NULL (`disconnect=0`).** Conditional mean ∝ conditional variance (risk fully priced,
  Merton 1973 ICAPM logic). 1/RV scaling reshuffles risk but must earn **no alpha** — and
  doesn't (mean *t* = −0.08 over 20 seeds).
- **Flat-mean (`disconnect=1`).** The textbook Moreira-Muir premise. The overlay wins on
  average but 30-year samples often can't certify it — a power lesson that contextualises the
  real tape's borderline *t*.
- **PLANTED leverage effect (`disconnect=2`).** Mean *falls* as variance rises (Black 1976
  leverage effect; the empirical post-spike return drag). The harness **must** bank this —
  and does (mean *t* = +2.09, 60% of seeds ≥ 2). *(Machinery proof; never cited as market
  evidence.)*

## Data sources used here

- **yfinance** daily auto-adjusted (total-return) closes: SPY (1993-02→2026-06, headline),
  QQQ, EFA, IWM (robustness) — cached under `_cache/vmp_prices.csv`.
- **yfinance ^IRX** — 13-week T-bill discount yield (percent), the risk-free leg — cached
  under `_cache/vmp_rf.csv`. All headline numbers pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map)

- [06-clockwork-vol](../06-clockwork-vol/) tests **fixed-period vol cycles** (can you *date*
  the next vol low?) — a spectral/timing claim, nothing to do with scaling.
- [130-vol-risk-premium](../130-vol-risk-premium/) tests the **implied-minus-realized (IV−RV)
  spread** as a premium and a timing signal — an *options-market* quantity.
- **This study** is the **Moreira-Muir SCALING strategy**: no options, no cycles — just
  realized variance from last month's daily returns deciding how much of the *same* asset you
  hold next month. The three are complementary faces of "volatility and returns".
