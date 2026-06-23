# References & literature map — Study 372 (Max-Pain Pinning)

## The claim under test

- **The folklore (max pain / pinning).** Trader lore holds that on option-expiry day the
  underlying is "pinned" to the **strike of maximum pain** — the settlement price at which
  the largest dollar amount of open interest expires *worthless* (option holders feel the
  most pain; writers pay out the least). The popular mechanism: market-makers who are short
  options hedge their gamma by buying weakness and selling strength, which mechanically drags
  the price toward the strike where their net option exposure is smallest. Max-pain
  calculators are a staple of retail options sites, and "the stock will pin at $X for
  expiry" is a standard expiry-day refrain.
- **The strong version.** Stated at full strength, the claim is a *law*: the expiry close
  reliably lands **on or adjacent to** the max-pain strike, across underlyings, often enough
  to trade (buy if max-pain is above spot, sell if below, and collect the drift into the
  close).

## What the academic literature actually finds

- **Pinning is real but small, and concentrated in single stocks.** Ni, Pearson & Poteshman
  (2005), *Stock price clustering on option expiration dates* (Journal of Financial
  Economics 78), document a statistically detectable tendency for **optionable stocks** to
  close *nearer* to strike prices on expiration Fridays than on other days — and trace part
  of it to **delta-hedge rebalancing** by market-makers and part to **stock-price
  manipulation** by option writers. The effect is **modest in magnitude** (a clustering
  tendency, not a hard pin) and is a *strike* effect, not specifically a *max-pain* effect.
- **Hedging-driven pinning theory.** Avellaneda & Lipkin (2003), *A market-induced mechanism
  for stock pinning* (Quantitative Finance 3), give the delta-hedging mechanism a formal
  model: pinning arises when a large long-gamma position is being hedged near a strike. The
  mechanism predicts pinning toward **the strike with large hedged open interest**, which can
  but need not coincide with the aggregate max-pain strike.
- **Max-pain specifically is weaker / largely folklore.** The popular "max-pain strike"
  aggregates *all* open interest into one number; the academic pinning evidence is about
  *individual* high-OI strikes and is strongest in **single-name** options at **monthly**
  expiry. In deep, liquid **index** options (SPX/SPY) the effect is heavily arbitraged and
  the close routinely sits a full percent or more from max-pain (as our snapshot shows). No
  refereed study establishes max-pain as a tradable forecast of the expiry close.

## Why our real tape is a *snapshot*, not a pinning panel

- **yfinance serves only the *current* option chain.** We can pull live call/put open
  interest per strike for upcoming expiries and compute today's **max-pain** strike, but the
  endpoint retains **no history of how past contracts expired** — there is no stored
  expiry-day close matched to that expiry's max-pain. A genuine pinning test needs exactly
  that panel (expiry-day close vs that expiry's max-pain, over many expirations), which is
  not free. So our real contribution is an explicit **cross-sectional snapshot**: 20 liquid
  underlyings on one as-of date, each with its max-pain strike and its spot-vs-max-pain gap.
  It documents *where price sits relative to max-pain right now*; it **cannot** test the
  *landing-at-expiry* claim, and we say so on the Signal axis.

## Why the decisive test is the synthetic control — the statistics

- **A snapshot can describe, but cannot certify a mechanism.** The landing claim is causal
  and temporal (price *moves toward* max-pain *into the close*). With no expiry-day panel we
  test it on a **deterministic pinning simulator** where the truth is known: a tunable
  ``pin`` knob adds a mean-reverting drag toward max-pain over the final days. With
  ``pin = 0`` there is **no** pinning, and the inference must NOT manufacture significance;
  with a large ``pin`` it must light up. This is the standard positive-control logic — a
  harness that can't bank a *planted* pin proves nothing by finding none (Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993, on resampling/placebo inference).
- **The honest baseline avoids a built-in bias.** Comparing the close's distance to max-pain
  against its distance to a *random grid strike* is biased (both cluster centrally and the
  test fires with zero pinning). We instead compare against the **spot-anchor** — the strike
  nearest where price *started* — so that pinning has to drag the close *away from its
  origin and onto max-pain* to register. A **paired Welch t** on the per-episode difference
  (Welch, 1947, *The generalization of "Student's" problem*) and a **label-shuffle placebo**
  (permute which max-pain pairs with which close) decide the Signal axis.
- **Base rates and the clustering illusion.** Strikes are a *grid*; any close is "near a
  strike" by construction, and max-pain usually sits centrally, so a naive "is the close near
  max-pain?" look will *always* say yes. The right question is whether the close is nearer
  max-pain than the geometry alone implies — the base-rate discipline of Kahneman & Tversky
  (1973), *On the psychology of prediction*.

## Method lineage (the desk's shared engine)

- **Max-pain computation.** [`data.max_pain_strike`](../max_pain_pinning/data.py) — the
  settlement price minimising total option payout, shared by the real fetch and the
  synthetic engine (so the control is self-consistent).
- **Closeness inference.** [`strategy.closeness`](../max_pain_pinning/strategy.py) and
  [`strategy.welch_t`](../max_pain_pinning/strategy.py) — per-episode distance to max-pain vs
  the spot-anchor, paired t on the difference.
- **Placebo null.** [`strategy.placebo_pvalue`](../max_pain_pinning/strategy.py) — permute
  the max-pain labels across episodes and ask how often a shuffled max-pain pins as well as
  the true one.
- **Tradability probe.** [`strategy.fade_trade`](../max_pain_pinning/strategy.py) — the
  believer's directional "fade toward max-pain" bet with a 1-day entry lag and a one-way cost.

## Data sources used here

- **yfinance** live option chains (call/put open interest per strike) + last close for 20
  liquid US underlyings (indices + large-caps), one snapshot as-of **2026-06-22**, cached
  under `_cache/mp_snapshot.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 195 — Monthly OpEx](../195-monthly-opex/)**: the calendar cousin — does the
  monthly option-expiration week itself carry a return effect? Same expiry-day plumbing,
  different question.
- **[Study 14 — Gamma Gospel](../14-gamma-gospel/)**: the dealer-gamma mechanism that the
  pinning story leans on — does aggregate dealer gamma actually steer the tape?
