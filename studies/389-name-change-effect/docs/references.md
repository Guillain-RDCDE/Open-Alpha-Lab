# References & literature map — Study 389 (Name-Change-Effect)

## The claim under test

- **The folklore.** A company that rebrands toward the hot theme — append **`.com`** in
  1999, add **`Blockchain`** in 2017, slap **`AI`** on it in 2023 — is said to get an
  instant sentiment **pop**, and then to **give it all back** once the hype cools. The
  poster children are vivid: **Long Island Iced Tea Corp → Long Blockchain Corp** (Dec 2017,
  shares ~+200–300% intraday, later delisted and an SEC/insider-trading case), **Eastman
  Kodak's "KodakCoin"** announcement (Jan 2018, ~+300% then faded), and the original
  dot-com renames (e.g. **Computer Literacy → fatbrain.com**, **K-tel → ktel.com**) that
  spiked on the rename alone.
- **Where it's repeated.** Financial media and academic event studies alike: the dot-com
  rename literature (below) found a real announcement effect; the 2017 "blockchain rename"
  episode was widely covered as a textbook hype trade. The believers' framing is two-legged
  — a *pop* on the rename and a *fade* afterward — and we test **both** legs.

## The academic anchor — dot-com renames really did pop (in 1998–2000)

- **Cooper, Dimitrov & Rau (2001), *A Rose.com by Any Other Name*, Journal of Finance.**
  The canonical study: firms that added `.com`/`.net`/`internet` to their name in 1998–99
  earned a large positive abnormal return (~+74% cumulative around the announcement) — and,
  crucially, the effect was there **regardless of the firm's actual involvement in the
  internet**. Pure name effect.
- **Cooper, Khorana, Osobov, Patel & Rau (2005), *Managerial actions in response to a market
  downturn: dotcom name changes*, Journal of Corporate Finance.** The sequel: after the
  bubble burst, firms **dropped** `.com` from their names — and *that* earned positive
  abnormal returns too. The name effect was a sentiment artefact in both directions.
- **Lee (2001) and the broader "investor-attention / categorisation" literature**
  (Barber & Odean, 2008, *All that glitters*, RFS) on how a salient label reallocates retail
  attention and, briefly, price. The pop is an attention effect, not an information effect.

## Why "and then give it back" is the hard part — and why our tape is biased against it

- **Survivorship.** The most spectacular give-backs **delisted** (Long Blockchain, UBI
  Blockchain Internet, several dot-com renames) and therefore have **no continuing yfinance
  series**. Our priced sample is the set of rebrands that *survived* — which is biased
  **toward** names that did not collapse, i.e. **against** the believers' fade. A
  survivor-only fade that is *not negative* is therefore a conservative refutation, and we
  name the bias on the Signal axis (Brown, Goetzmann, Ibbotson & Ross, 1992, *Survivorship
  bias in performance studies*, RFS).
- **Small-sample inference.** With ~25 documented rebrands (and ~21 priced), the
  cross-section of abnormal returns has a large standard error. We test each leg's mean
  against zero with a **one-sample t** (Welch, 1947) and, because the sample is tiny and
  fat-tailed, with a **placebo / randomisation null** — draw the same number of random
  non-event windows on the same tickers and ask how often chance matches the pop or the fade
  (Fisher's randomisation logic; Efron & Tibshirani, 1993, *An Introduction to the
  Bootstrap*). Event-study standard errors: MacKinlay (1997), *Event studies in economics and
  finance*, JEL.
- **Selection on a famous anecdote.** Long Blockchain is selected *because* it was extreme;
  building a "law" from the loudest cases is the classic data-snooping trap (Harvey, Liu & Zhu,
  2016, *…and the Cross-Section of Expected Returns*, RFS). A representative table of rebrands
  — not just the legends — is the honest test, and that is what the hardcoded table is.

## Method lineage (the desk's shared engine)

- **Abnormal-return event windows.**
  [`strategy.event_window`](../name_change_effect/strategy.py) computes excess-of-SPY
  cumulative returns on a short **pop** leg `[+1 … +5d]` and a longer **fade** leg
  `[+6 … +65d]`, with a one-day entry lag (you act the day *after* the rename headline).
- **Welch t + placebo p-value.** [`strategy.welch_t`](../name_change_effect/strategy.py) and
  [`strategy.placebo_pvalue`](../name_change_effect/strategy.py) — each leg's mean vs zero,
  and a 20,000-draw randomisation null sized to the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_rebrands`](../name_change_effect/data.py) plants a known pop+give-back of
  size `edge` into otherwise-random windows; with `edge=0` the inference must NOT manufacture
  significance, and a large `edge` must light up both legs. The control runs offline.
- **Costs on the believers' trade.**
  [`strategy.net_of_costs`](../name_change_effect/strategy.py) charges a one-way small-cap
  cost on the **four** crossings of a buy-the-pop / short-the-fade round trip.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + ~25 rebrand tickers, 1995-01-03 → 2026-06-18,
  cached under `_cache/rebrand_prices.csv`. The rebrand table (tickers, announce dates, waves)
  is hardcoded in [`data.REBRANDS`](../name_change_effect/data.py); famously-delisted rebrands
  are listed in [`data.DELISTED`](../name_change_effect/data.py) for the survivorship caveat.
  Headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 343 — Data-Mining-Roulette](../343-data-mining-roulette/)**: the methodological
  cousin — how loud anecdotes manufacture "laws" that don't survive a representative sample.
- **[Study 169 — Fluent-Tickers](../169-fluent-tickers/)**: the adjacent name-effect — whether
  a *pronounceable* ticker carries a premium. Same family (a label, not a fundamental).
