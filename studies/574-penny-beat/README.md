# Study 574 — Penny-Beat 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do penny-beaters really pay the price later? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The literature backs it (Bhojraj et al. 2009) and the engine finds a penny-minus-decisive spread of **−4.9%** (*t* **−9.2**, placebo *p* 0.0005) — **but** with the management penalty knob set to *zero* the spread is *still* **−1.8%** (*t* −3.5): roughly **half is mechanical PEAD composition** (a +1c surprise is a smaller surprise, so it drifts less regardless of manipulation). And there is **no free real tape** (consensus-vs-actual EPS is licensed I/B/E/S) — synthetic-only ⇒ ceiling `WEAK`. |
| **Tradability** — could you short the penny-beaters? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No reachable data, a tiny per-quarter penny bucket, and shorting exactly the crowded just-beat names — with half the "edge" being PEAD you could harvest more directly. The friction cost is a footnote; the trade is a `MIRAGE`. |
| **"Discontinuity real?"** — is the +$0.01 spike there? | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Yes. The engine detects the earnings-management fingerprint cleanly: a **+606** excess of firm-quarters at exactly +1c (z **+17.7**, placebo *p* 0.0005) and the small-miss bins depleted to **48%** of their mirror — and it's flat at the null (z +0.3 with the spike off). Famous and robust (Burgstahler-Dichev 1997). |

> **In one sentence:** the penny-beat *discontinuity* — the tell-tale spike of firms clearing consensus by exactly a penny — is real, famous, and the engine catches it decisively (z +17.7); but the *stock-return* penalty on top is only half a management story (the other half is mechanical post-earnings drift, present even with zero manipulation) and lives on a licensed dataset a retail stack can't reach, so it earns `Weak` × `Mirage` with the discontinuity itself `Confirmed`.

## What we tested

The **penny-beat** claim — an earnings-management cousin of [Benford's Law (328)](../328-benford-law/)
and the [Beneish M-score (229)](../229-beneish-m-score/): firms that *just barely* beat consensus EPS
by ~1 cent look **managed** (Burgstahler & Dichev 1997; Degeorge-Patel-Zeckhauser 1999), and those
penny-beaters go on to earn **weaker** returns than decisive beaters (Bhojraj et al. 2009). Because a
survivorship-free, point-in-time **consensus-vs-actual EPS** panel is a licensed I/B/E/S product a
no-key retail stack can't reach, this is a **synthetic-only** study (like [273](../273-lego-returns/)/
[275](../275-whisky-cask/)/[276](../276-sneaker-resale/)): a deterministic surprise-distribution
generator with one knob planting the +$0.01 spike and one planting the return penalty. The engine
(a) **detects the discontinuity** (excess-mass z-score + a smooth-relabel placebo), (b) tests the
**return penalty** (penny vs decisive, a two-sample *t* + label-shuffle placebo + costs/borrow), and
(c) **decomposes it honestly** — showing ~half the naive spread is mechanical PEAD composition, not
management. A seed-robust (25-seed) positive control proves the detector catches a planted effect and
stays flat at the null; the missing real tape caps the Signal at `WEAK`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "penny beat" is, why the +$0.01 spike is a smoking gun, and why the "they pay for it later" story is only half true |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the discontinuity z + placebo, the penny-vs-decisive two-sample *t*, the PEAD-composition decomposition, the within-bin clean penalty, the robustness sweep, costs & borrow, and the seed-robust synthetic control |

The fingerprinted, as-of'd reproducible run (synthetic panel fp `1ad98df82652`, as-of 2026-06-30)
is in [docs/results.md](docs/results.md); the offline machinery runs entirely on the deterministic
synthetic world in [`penny_beat/data.py`](penny_beat/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`penny_beat/`](penny_beat/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
