# Study 996 — The Palindrome Portfolio 🪞

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — can a calendar pattern with no possible mechanism be made to look significant? | ![Busted](https://img.shields.io/badge/Busted-c0392b?style=flat-square) | Searching **98 meaningless calendar rules** across 6 assets — 531 tests in total — produced **25 results significant at 5%**, against 27 expected by pure luck. The best was *day is a power of two* on SPY at *t* = **+3.22**, worth +28.8% a year. On a *t*-table that is a one-in-786 event. It is nothing of the sort. Reshuffling the returns and rerunning the identical search produced a best |*t*| of **2.63** at the median and 3.42 at the 95th percentile, so the observed maximum has a shuffle *p*-value of **0.10**. Bonferroni would have required *t* > 3.91; **0** rules cleared it. The correct reading of the headline number is that it is a draw from the distribution of the *maximum* of 531 tries, whose median is 3.22 — not from a *t*-distribution, whose median is 0.67. |
| **Tradability** — what does it cost to believe one? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Selecting the ten best rules on the first half of the sample and testing them on the second half is the whole story in one table: their in-sample *t* averaged **-0.59** and their out-of-sample *t* averaged **+0.65**, with 6 of 10 keeping the same sign. Traded with costs, the single best rule turned +28.8% a year of apparent edge into -5.88% against simply holding the index. That is the price of believing a pattern with no mechanism, and it is charged in full. |

> **In one sentence:** 531 tests of hypotheses that cannot be true produced a best *t* of +3.22 — which sounds like a discovery until you notice that shuffling the returns produces 2.63 on a median attempt.

## What we tested

This study looks for a pattern that **cannot exist**. Palindromic dates —
22/02/2022 — have no mechanism: no earnings land on them, no fund rebalances to them, no
behaviour attaches to them. So every result is known in advance to be a false positive, and that
certainty is what makes them useful: they are a **calibrated ruler for data mining**.

Roughly 120 meaningless calendar rules (palindromes, prime days, Fibonacci days, "the digits add
to nineteen") are run across six assets — about 700 tests — and the winner is examined the way a
real finding would be. It has a *t* around 3.4, which a *t*-table calls a one-in-a-thousand
event. It is nothing of the sort, and the study shows why three different ways.

**The maximum, not the mean.** A researcher who tries *k* ideas and reports the best is drawing
from the distribution of the *maximum* of *k* draws, whose median after 700 tries is above 3 —
not from a *t*-distribution, whose median is 0.67. **The shuffle test** reruns the entire search
on reshuffled returns, which handles the heavy overlap between rules that Bonferroni's arithmetic
cannot ("prime day" and "multiple of 3" are not independent tests). And **the split sample**
selects winners on the first half and watches them evaporate on the second. Finally the same
search is run on a pure random walk, and the best *t* it produces is indistinguishable from the
one the real tape gave — which is the whole finding.
**Dedup:** distinct from every calendar study on the desk (**067-monday-effect**,
**552-weekend-effect**, **283-sell-in-may**, **410-santa-rally**), which test hypotheses with
plausible mechanisms; from **718-p-hacking-simulation** (simulated researchers rather than a
real tape searched with real rules); and from **860-backtest-overfitting** (parameter
optimisation within one strategy rather than a search across unrelated hypotheses).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | a genuinely significant-looking market pattern based on nothing at all, and the three ways to catch it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | a 700-test calendar scan, the distribution of the maximum t, Bonferroni and Benjamini-Hochberg, a shuffle test that respects overlap between rules, split-sample decay, and calibration against a pure random walk |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`palindrome/`](palindrome/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
