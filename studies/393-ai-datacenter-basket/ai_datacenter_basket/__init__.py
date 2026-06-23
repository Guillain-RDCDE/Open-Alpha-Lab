"""Study 393 — the AI-Datacenter-Basket (riding the AI build-out by buying the picks-and-shovels).

The pitch making the rounds in 2024-2026: you don't need to pick the one AI winner — just buy
the **datacenter-and-power basket** (the chips, the cooling, the switches, the servers, the
electricity that the AI build-out has to consume) and you ride the whole capex wave. The poster
names are NVDA, VRT, ETN, CEG, VST, SMCI, ANET, DELL — equal-weight, hold, win.

The catch is the one every thematic basket carries: the names are chosen *because they already
won*. NVDA, VRT, CEG, VST were not the obvious "datacenter-and-power" basket in 2019 — they are
the realised top of a much larger field of plausible candidates (utilities, REITs, networking,
semis, industrials). So the study splits the claim in two, exactly like Study 355
(Magnificent-Seven): is the basket's excess over the market (SPY) and the tech tape (QQQ)
statistically real under autocorrelation-robust (HAC) inference — and *how much of it is the
after-the-fact selection*? The decisive instruments are an **ex-post "pick the winners" placebo**
and a **random-basket sampling distribution** drawn from the same candidate field, plus a
deterministic synthetic control whose single ``alpha_spread`` knob proves selection manufactures a
spread on a no-edge tape.

See :mod:`ai_datacenter_basket.data` (real monthly panel loader + deterministic synthetic control)
and :mod:`ai_datacenter_basket.strategy` (basket / equal-field / ex-post-winners / random-basket
returns, HAC t-stat of the spread, the selection decomposition, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
