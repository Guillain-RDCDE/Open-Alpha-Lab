"""Study 679 — Psychological Line (PSY).

The Japanese "Psychological Line" is a decades-old technical gauge: count the share of
up-closes over the last N days, express it as a percentage, and read it as a crowd-emotion
thermometer. Above 75% "everyone is already long" (sell); below 25% "everyone has already
capitulated" (buy).

``data.py`` fetches and caches the real daily tape (SPY + a basket) and builds a
deterministic synthetic control. ``strategy.py`` implements the indicator, the
zone-trigger trade ledger, the conditional-forward-return test, the parameter-robustness
grid and the random-direction placebo.
"""
