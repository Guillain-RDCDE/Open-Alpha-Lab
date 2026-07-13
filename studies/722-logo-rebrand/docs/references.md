# References & literature map — Study 722 (Logo-Rebrand)

## The claim under test

- **The folklore, two-sided.** A corporate **rebrand / logo change** is treated as a *signal*
  about the firm's trajectory — but the two camps read it oppositely:
  - **Renewal camp.** A fresh name or identity marks a turnaround; the market re-rates the
    stock. Design/branding practice sells rebrands as growth catalysts (Interbrand, Landor and
    the agency literature; the Meta and Alphabet renames were pitched as strategic pivots).
    *Prediction: positive post-reveal drift — buy the rebrand.*
  - **Skeptic / value camp.** A splashy rebrand is what a *floundering* firm does to distract
    from a deteriorating business — "when management is repainting the lobby, read the 10-K."
    *Prediction: negative post-reveal drift — fade the rebrand.*
- **Where it's repeated.** Business and design press cover rebrands as renewal (Fast Company,
  Bloomberg on Meta/Alphabet); investor folklore and financial-blog commentary treat logo
  churn as a distress tell. Vivid cases anchor both sides — **Google → Alphabet** (2015) and
  **Facebook → Meta** (2021) for renewal; **Weight Watchers → WW** (2018, Chapter 11 in 2025),
  **Tribune → tronc** (2016), and **Overstock → Bed Bath & Beyond** (2023) for the floundering
  read. We test **both** as directional bets on the same object: the drift after a rebrand.

## The academic anchor — name/identity changes and abnormal returns

- **Cooper, Dimitrov & Rau (2001), *A Rose.com by Any Other Name*, Journal of Finance.** The
  canonical result that a *theme-chasing* name change (adding `.com`/`.net`) earned large
  positive abnormal returns in 1998–99 **regardless of actual internet involvement** — a pure
  attention/label effect. The desk tests that specific variant in
  [Study 389 — Name-Change-Effect](../389-name-change-effect/); the present study asks the
  broader, non-theme question (renewal vs distress) for ordinary corporate rebrands 2010–2025.
- **Cooper, Khorana, Osobov, Patel & Rau (2005), *Managerial actions in response to a market
  downturn: dotcom name changes*, Journal of Corporate Finance.** Firms *dropped* `.com` after
  the bust and earned positive abnormal returns too — the name effect was a sentiment artefact
  in both directions, i.e. weak evidence for any stable "rename → future performance" signal.
- **Bosch & Hirschey (1989), *The valuation effects of corporate name changes*, Financial
  Management; Karpoff & Rankine (1994), *In search of a signaling effect: the wealth effects of
  corporate name changes*, Journal of Banking & Finance; Kot (2011), *Corporate name changes:
  price reactions and long-run performance*, Pacific-Basin Finance Journal.** The direct
  literature: announcement-window reactions to plain (non-theme) name changes are **small and
  mixed**, and long-run post-change performance shows **no reliable abnormal drift** — consistent
  with this study's null drift.
- **Investor-attention / categorisation.** Barber & Odean (2008), *All that glitters*, RFS — a
  salient label reallocates retail attention and, briefly, price. The announce pop, where it
  exists, is an attention effect, not information about the future.

## Why "renewal or floundering" is hard to certify — and why our tape is biased

- **Reverse causation by firm health.** Troubled firms disproportionately change their *name*
  (a strategic reset), while healthy incumbents merely *restyle a logo*. Any cross-sectional
  drift difference therefore encodes **prior** firm health, not a forward-looking rebrand
  signal — a selection-on-the-treatment problem, not a treatment effect.
- **Survivorship.** The worst-outcome rebrands **delisted or went private** (Twitter → X;
  Weight Watchers → WW then Chapter 11; Overstock → Bed Bath & Beyond; ViacomCBS → Paramount,
  acquired) and leave **no** clean yfinance series. Our priced sample is biased **toward**
  survivors, i.e. **against** the floundering thesis. A survivor-only drift that is *not
  negative* is a conservative refutation, and we name the bias on the Signal axis (Brown,
  Goetzmann, Ibbotson & Ross, 1992, *Survivorship bias in performance studies*, RFS).
- **Small-sample inference.** With ~26 documented rebrands (~22 priced), the cross-section of
  abnormal returns has a large standard error and heavy outlier leverage. We test each leg with
  a **one-sample t** (Welch, 1947), a **placebo / randomisation null** (Fisher's randomisation
  logic; Efron & Tibshirani, 1993, *An Introduction to the Bootstrap*), and an **outlier-drop**
  fragility curve. Event-study standard errors: MacKinlay (1997), *Event studies in economics
  and finance*, JEL.

## Method lineage (the desk's shared engine)

- **Abnormal-return event windows.** [`strategy.event_window`](../logo_rebrand/strategy.py)
  computes excess-of-SPY cumulative returns on a short **announce** leg `[+1 … +5d]` and a
  longer **drift** leg `[+6 … +126d]`, with a one-day entry lag (you act the day *after* the
  reveal headline).
- **Welch t + placebo p-value.** [`strategy.welch_t`](../logo_rebrand/strategy.py) and
  [`strategy.placebo_pvalue`](../logo_rebrand/strategy.py) — each leg's mean vs zero, and a
  20,000-draw randomisation null sized to the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_rebrands`](../logo_rebrand/data.py) plants a known renewal drift of size
  `edge` into otherwise-random windows; with `edge=0` the inference must NOT manufacture
  significance, and a large `edge` must light up the drift leg. The control runs offline.
- **Costs on the believers' trade.** [`strategy.net_of_costs`](../logo_rebrand/strategy.py)
  charges a one-way large-cap cost on the **two** crossings of a buy-the-rebrand-and-hold round
  trip.
- **Reproducibility.** [`quantlab.repro`](../../../quantlab/repro.py) pins the as-of date and
  prints a content fingerprint of the cached tape.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + ~26 rebrand tickers, 2008-01-02 → 2026-06-30,
  as-of 2026-06-30 (`fp=c12adbfe3fd2`), cached under `_cache/rebrand_prices.csv`. The rebrand
  table (tickers, announce dates, kinds) is hardcoded in
  [`data.REBRANDS`](../logo_rebrand/data.py); famously-delisted / privatised rebrands are listed
  in [`data.DELISTED`](../logo_rebrand/data.py) for the survivorship caveat. Headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: the theme-chasing cousin —
  `.com`/`Blockchain`/`AI` renames that "pop then dump." Same family (a label, not a
  fundamental), same small-sample / survivorship pathology.
- **[Study 343 — Data-Mining-Roulette](../343-data-mining-roulette/)**: the methodological
  cousin — how loud anecdotes (here: two confounded outliers) manufacture "laws" that don't
  survive a representative sample.
