"""Study 574 — Penny-Beat: the earnings-management discontinuity.

The claim (an earnings-management cousin of Benford's Law and the Beneish M-score): firms that
*just barely* beat the consensus EPS forecast — clearing it by exactly ~1 cent — look *managed*.
The tell is a **discontinuity** in the distribution of earnings surprises: an excess spike of
observations at a surprise of +$0.01 (and a suspicious dip just below zero), the fingerprint of
firms that nudged a small miss into a small beat (Burgstahler & Dichev 1997; Degeorge, Patel &
Zeckhauser 1999). The trading claim on top: those penny-beaters, having borrowed from the future,
go on to earn **weaker** subsequent returns than the honest, decisive beaters.

This desk cannot reach a survivorship-free, point-in-time **I/B/E/S consensus + actual EPS** panel
from a no-key retail stack — the raw material of the penny-beat test is a licensed dataset. So this
study is **synthetic-only** (like [273 Lego-Returns](../273-lego-returns/),
[275 Whisky-Cask](../275-whisky-cask/), [276 Sneaker-Resale](../276-sneaker-resale/)): a
deterministic surprise-distribution generator with a single knob planting the penny-beat spike and
its return penalty, and an engine that (a) *detects* the discontinuity and (b) tests the return
penalty — with a placebo null and a seed-robust positive control. A synthetic-only study can never
earn `REAL` (that needs a robust t on a real tape); the honest ceiling here is `WEAK`.

Distinct neighbours on this bench:
- [229 Beneish M-score](../229-beneish-m-score/) — the 8-ratio *accounting* manipulation score.
- [328 Benford-Law](../328-benford-law/) — the leading-digit fraud screen on reported figures.
This study is the **earnings-surprise discontinuity** itself (the +$0.01 spike) and its return
consequence — a different tell (the shape of the surprise histogram, not a ratio or a digit).
"""
