# Study 621 — Share-Class Spreads 🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the one-way 1/1500 bound real on the tape? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Over 7,583 days BRK-B exceeds parity by >50 bps on only **1.77%** of days (>100 bps: 0.24%; worst **+194 bps**, March-2020 chaos) against a free tail to **−700 bps** — a **13× asymmetry** — and the structural B discount of **−35.5 bps** carries **HAC t = −12.5**, significant in every era. The unbounded GOOG/GOOGL twin is the control: symmetric excursions (0.7×), an **85-day** half-life vs **4.5** for BRK, and a "voting premium" that flipped from **+238 bps** to **−54 bps** (buybacks tilt to class C). Two named mega-cap pairs — no panel, no survivorship screen, nothing searched over. |
| **Tradability** — can you harvest the gap? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | With one honest execution lag the long-cheap/short-rich pairs trade nets **+0.21%/yr (t = 0.38)** at 2 bps and goes negative at 5+ bps (borrow paid). The paper +8.4%/yr (t = 9.4) exists only if you fill **at the signal's own close prints**. Add a ~$750k minimum A-share lot, a one-way broker-mediated conversion, and a gap averaging 35 bps — nothing deployable survives. |
| **"Has the B-discount ever been a tradable signal for A-holders?"** | ![Busted](https://img.shields.io/badge/Tradable_for_A--holders%3F-Busted-8b949e?style=flat-square) | The switch-to-cheap-B overlay shows **+4.3%/yr (t = 6.0)** at a fill-at-print convention — and **+0.1–0.4%/yr (t ≤ 1.2)** at every threshold, in every era (wild pre-2010 discounts included), once the fill waits for the first close you could actually trade. The "signal" is mostly BRK-A's stale, wide close print itself. |

> **In one sentence:** Buffett's memo is on the tape — the one-way A→B conversion really does cap BRK-B at ~1/1500th of A (violations: 1.77% of days beyond 50 bps, capped at +194 bps, vs a free discount tail to −700 bps; mean discount −35.5 bps at HAC t = −12.5) while the bridgeless GOOG/GOOGL spread wanders for months and even flipped sign — but the bound is enforced *by conversion arbitrageurs, not for you*: one honest execution lag turns every harvesting scheme into t < 1.2 dust.

## What we tested

The claim comes straight from Berkshire's own *compab* memo: any A share converts — one way —
into 1,500 B shares, so **B can never cost meaningfully more than A/1500**, while nothing stops
a B discount; GOOG vs GOOGL, with **no conversion bridge**, should show no bound at all. On the
full yfinance daily tape (BRK pair 1996-05-09 →, split-adjusted to a constant 1,500 parity;
GOOG pair 2014-04-03 →, as-of 2026-06-30) we measure bound violations by threshold, the tail
asymmetry, the discount distribution with a Newey-West **HAC t** on its mean (daily gaps are
heavily autocorrelated), era slices, and AR(1) **half-lives** of both gaps. Tradability charges
a z-score pairs trade and an A-holder class-switch overlay with exactly **one execution lag**
(signal close *t*, fill close *t+1*), one-way costs × NAV per leg and borrow on the short —
against the *diagnostic* fill-at-print convention that manufactures the paper edge. A
deterministic synthetic control (planted discount + hard one-way bound vs a symmetric unbounded
null) proves the detector faithful. Distinct from [05-twin-spread](../05-twin-spread/)
(statistical distance pairs, no conversion right) and cousins
[367-closed-end-fund-discount](../367-closed-end-fund-discount/),
[618-gbtc-premium-cycle](../618-gbtc-premium-cycle/), [620-a-h-premium](../620-a-h-premium/) —
this is the only one where the bound is a contractual option every A-holder owns.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why one Berkshire share can never cost more than 1,500 of the other (and why the reverse can happen), what the Google twins do without that leash, and why the "free money" in the discount evaporates the moment you actually have to trade — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | violation thresholds + tail asymmetry, HAC t on the mean gap by era, half-life contrast, the fill-at-print vs next-close collapse on both trading rules, costs × borrow, and the planted-bound synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`share_class_spreads/`](share_class_spreads/). The signal is the split-adjusted parity gap `1500·B/A − 1` (and `GOOGL/GOOG − 1` for the unbounded twin); the myth-check is the A-holder switch overlay under an honest one-lag fill. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
