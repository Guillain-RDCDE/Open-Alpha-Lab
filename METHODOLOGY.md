# How this desk works — the methodology

*This is the house style for every study in [Open-Alpha-Lab](README.md). It exists
so that a reader landing on study #1 or study #17 — at their desk, on the couch,
whoever they are — always finds the **same shape**, asks the **same questions**,
and gets the **same kind of honest answer**. The template lives at
[`studies/_TEMPLATE/`](studies/_TEMPLATE/); this page explains the thinking behind
it.*

---

## The one idea

Most trading ideas you'll hear — *"stocks only go up overnight"*, *"sell in May"*,
*"momentum always works"* — are interesting precisely because they're *almost*
true. There's a real pattern in the data, and then a much bigger story bolted on
top. The desk's whole job is to pull those two apart:

> **Is the signal real?** — and, separately, **does it survive the real world?**

Two questions, never collapsed into one. A signal can be bulletproof statistically
and still be a `MIRAGE` once you charge it costs and try to scale it. Saying so
clearly — and showing the exact line where the dream dies — is the product.

We publish the dead-ends, not just the wins. A well-documented *"no"* is worth more
than a hand-wavy *"yes"*.

---

## The front-card and the notebooks — tell the story once

The README is a **synthetic front-card**, not a document. In ~30 lines it answers
three things and stops:

- **The verdict** — the two stamps and one honest sentence, in a box at the top. A
  reader gets the conclusion in five seconds.
- **What we tested** — the claim stated the way its *believers* state it, steelmanned,
  with a link to whoever's making it. Two to four sentences.
- **Where the proof is** — links to the two notebooks and any other deliverable.

That's it. The **full demonstration lives entirely in the notebooks** — so we never
tell the same story twice. (For years the front page *also* re-narrated the whole
teardown; that produced two documents saying the same thing. The README is now the
menu; the notebooks are the meal.)

The notebooks carry the **two-readers, one-page** contract, read two ways at once so
we never split the audience:

- **`01_for_the_curious`** runs the plain layer — no jargon, top-to-bottom, the claim
  and the stakes and the tricks that inflate the headline.
- **`02_for_the_quants`** runs the rigorous layer — HAC *t*-stats, the confounds ruled
  out and how, capacity, the Reality Check.

Same story at two altitudes — and **both walk the same seven beats below.**

---

## The seven beats

The two notebooks walk these seven beats, in this order, with these headers. The
order isn't arbitrary — it's the order an honest investigation actually happens in:
state the claim fairly, work out why it'd matter, decide how you'd be wrong, do the
work, call it, ask if it pays, then open the door for the next person. (The README
front-card surfaces only **Beat 0** and a compressed **Beat 1**; the rest is the
notebooks' job.)

> **Beat 0 — Verdict, read first.** Before any of it: the answer, in a little box.
> Two stamps and one sentence. A reader gets the conclusion in five seconds, *then*
> chooses to follow the reasoning. We don't bury the lede. *(This is the heart of the
> README front-card.)*

**1 · The Claim.** State the idea the way its *believers* state it — at full
strength, steelmanned, with a link to whoever's making it. No strawmen. If we're
going to take it apart, we take apart the strongest version.

**2 · So What?** If it were true, what would it be worth, and what would it imply?
The stakes — in money, and in what it'd mean about how markets work. This is what
earns the reader's attention before we get technical.

**3 · How We'd Know.** The falsifiable test, *announced before we run it*, so we
can't move the goalposts afterward. What data, what experiment — and crucially,
**what result would make us say "mirage."** This is where the shared 6-step
protocol (below) gets named.

**4 · The Teardown.** We run it. The findings, in narrative order — one result, one
number, one figure at a time. This is the body of the work; the charts carry it.

**5 · The Verdict.** The two stamps from the box up top, now *earned*, with the
decisive numbers in one place. Honest about what's real and what isn't — and if
it's a mirage, the exact line where it dies.

**6 · Could You Trade It?** The question most write-ups skip: if you actually
wanted to get *paid*, what would it take? The venue you'd execute on, the costs
you'd really pay, the size before market impact eats the edge, the risk you'd run —
and the candid bottom line. This is usually where a "real" signal becomes a mirage.

**7 · Going Further.** What we didn't settle, the next experiment, the variant worth
testing — and an explicit invitation: what to fork, what to challenge, what a
contributor could PR. The investigation never really closes; it hands off.

---

## The verdict rubric

So that studies are *comparable*, every verdict uses the same two axes and the same
stamps. No bespoke scoring per study — you can line them all up in one table.

| Axis | Question | Stamps |
|---|---|---|
| **Signal** | Is the effect statistically real? | `REAL` · `WEAK` · `NONE` |
| **Tradability** | Does it survive costs, capacity and scale? | `INVESTABLE` · `FRAGILE` · `MIRAGE` |

A study may add **one** extra axis when the idea demands it (e.g. *"Manipulation?
→ `NOT SUPPORTED`"* for the overnight study) — but the two core axes are always
present, always first.

What the stamps mean:

- **Signal · `REAL`** — survives autocorrelation-robust inference (HAC *t*, Lo SE)
  and, where relevant, a data-snooping correction. `WEAK` — significant raw but
  fragile to method or selection. `NONE` — indistinguishable from noise once tested
  honestly.
- **Tradability · `INVESTABLE`** — a residual edge survives realistic costs *and*
  holds meaningful capital. `FRAGILE` — survives on paper but thin, decaying, or
  tiny-capacity. `MIRAGE` — gone once you charge real costs, or it can't scale, or
  it's just beta you were always paid for.

### The visual language — verdict badges

One palette carries the verdict everywhere a reader meets it — the landing table,
each study's front-card, and the hero of both notebooks — so the same colour always
means the same thing. The stamps render as flat-square [shields.io](https://shields.io)
badges, not coloured emoji dots (which wrap badly in narrow table cells):

| Colour | Hex | Signal | Tradability | 3rd "myth-check" axis |
|---|---|---|---|---|
| 🟩 green | `2ea44f` | `REAL` | `INVESTABLE` | — |
| 🟨 amber | `dab617` | `WEAK` / `MIXED` | `FRAGILE` | — |
| 🟥 red | `c0392b` | `NONE` | `MIRAGE` | — |
| ⬜ grey | `8b949e` | — | — | `BUSTED` · `MISATTRIBUTED` · `NOT SUPPORTED` · `CONFIRMED` · `PRE-REG` |

- **Front-card / landing:** a message-only pill —
  `![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square)`.
- **Notebook hero (cell 0):** a labelled badge that names the axis —
  `![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)`.
- **Encoding:** space → `_`, literal hyphen → `--`, `?` → `%3F`
  (e.g. `Not_supported`, `Pre--reg`, `Explosive%3F`).
- **Compound stamp:** two badges plus a plain-text qualifier —
  `![Real](…/Real-2ea44f…) on the level · ![None](…/None-c0392b…) on the spike`.

Every notebook opens on the same **hero**: a tight H1 + one-line subtitle, the verdict
badge row, a 2–3 sentence lede, then a *not-investment-advice* callout and a pointer to
the companion notebook. The scaffold is in
[`studies/_TEMPLATE/notebooks/build_notebooks.py`](studies/_TEMPLATE/notebooks/build_notebooks.py).

---

## The shared protocol (the engine behind beat 4)

Beat 4 isn't improvised per study — it runs a fixed gauntlet, powered by the
reusable [`quantlab/`](quantlab/) engine so the *same* method produces the *same*
kind of number every time:

1. **Decompose / measure** the raw effect with exact identities — no fitting, no
   free parameters. *(`quantlab/decompose.py`)*
2. **Robust inference** — Newey-West (HAC) and Lo/Mertens standard errors,
   **circular block bootstrap** CIs (i.i.d. resampling destroys the volatility
   clustering the inference is supposed to respect), White (2000) Reality Check
   on the **stationary bootstrap** (Politis-Romano), as White prescribes. *Is it
   real?* *(`quantlab/analytics.py`, `stats.py`, `bayes.py`)*
3. **Critique the magnitude** — compounding, log-scale illusions, unit errors, data
   artefacts (splits/dividends), selection. *(`quantlab/diagnostics.py`)*
4. **Alpha vs beta** — how much of the "edge" is just a risk premium you were always
   paid for? *(`quantlab/stats.py`)*
5. **Execution & capacity** — cost sweeps, break-even cost, square-root market
   impact, the scale at which the edge dies. *(`quantlab/backtest.py`,
   `analytics.py`, `simulate.py`)*
6. **Verdict** — the two stamps, with the numbers behind them.

Everything is deterministic (fixed seeds), tested (`pytest`, CI on Python
3.10–3.12), and reproducible from cached or freshly-fetched data.

### The inference bar — what each stamp must show

A bench-wide audit taught us that the stamps drift upward unless the bar is
written down. So it is:

- **`REAL` is earned by the tape, not the literature.** It requires an
  autocorrelation-robust statistic (HAC *t*, Lo *t*) clearing **2** on the
  *real* data the study ran. Three decades of published support plus a
  sub-2 *t* on our sample reads **`WEAK`** — say "the literature says real;
  this tape alone can't certify it" and stamp accordingly.
- **A synthetic control is a machinery proof, never market evidence.** It shows
  the harness *can* detect the effect it plants (and every harness must pass
  that positive control — a pipeline that can't bank a planted signal proves
  nothing by finding nothing). Quoting a synthetic Sharpe in support of a
  Signal stamp is circular, full stop.
- **No conditional claim without uncertainty.** Conditional accuracies carry
  Wilson intervals; sub-period contrasts ("decayed since…", "faded after…")
  carry a block-bootstrap or exact test of the *difference*, on a split that is
  justified, not snooped. Words like *unambiguous* are banned below *t* = 2.
- **`MIXED` (amber)** is allowed when the verdict genuinely splits by regime or
  leg — but only as shorthand for a split spelled out on the front-card
  (*"Real on the level · None on the spike"*).

---

## House rules

A few non-negotiables that keep the desk honest:

- **State your data choices as decisions, not details.** Adjustment mode
  (split-only vs total-return) literally moves return between night and day — pick
  one, say which, and say why.
- **Normalise before you marvel.** Sharpe over raw return; per-calendar-hour over
  per-session. Most "shocking" gaps shrink the moment you put both sides on the
  same clock.
- **Charge costs against the *alpha*, not the gross.** The spread, paid twice a day,
  ~252 days a year, is where most paper edges go to die.
- **Selection is everywhere.** "The 25 most extreme markets" is significant by
  construction. Correct for it (Reality Check) or don't claim it.
- **No silent caps.** If a study samples, truncates, or skips, it says so in the
  open. A hidden limitation reads as "we covered everything" when we didn't.
- **Pin the as-of, publish the fingerprint.** Live data drifts and extends, so a
  number with no provenance isn't reproducible. Freeze every headline run to an
  explicit as-of date and print a content fingerprint of the inputs
  (`quantlab/repro.py`). A reader who reruns and matches the fingerprint holds your
  exact data; a mismatch flags drift *loudly*, instead of a silently different number.
  Two corollaries: **an as-of date is never in the future** (a `resample("ME")`
  label is not a pull date), and **a stamped run never includes a partial month
  or year** — drop the in-progress bar before the stats.
- **Survivorship is named on the Signal axis, not just buried in Tradability.**
  Any cross-sectional panel goes through the opt-in guard
  (`allow_survivorship_bias=True` — `quantlab/universe.py`, `hf_data.py`) and the
  caveat travels with every place the stamp appears. A current-membership panel
  can *invert* an anomaly outright (it manufactured a "significantly negative"
  premium once on this bench), and a bias can point *against* the effect as
  easily as for it — reason about its direction explicitly, in writing. A
  control that shares the bias (same-month vs other-month) can support
  *existence*; it can never certify *magnitude*.
- **One execution lag, documented exactly.** The convention is: signal known at
  the close of *t* earns the return of *t+1* — one `shift`, applied once. A
  second silent shift hiding in `book_returns` once made a one-day reversal
  study measure a two-day-old signal and call the premium dead. Docstrings state
  the *effective* lag, and calendar-known rules (turn-of-month windows) need no
  lag at all. Conservative-by-accident is still wrong.
- **Costs are counted one-way, legs are counted both, and shorts pay borrow.**
  Define turnover as one-way × NAV and say so next to every break-even. Charge
  financing on a levered book exactly once (the bench's worst self-inflicted
  wound was charging the risk-free rate twice). And when one side of a race is
  in cash part-time, compare **excess-of-cash to excess-of-cash** — a raw-Sharpe
  vs excess-Sharpe race once manufactured a verdict here.
- **Gross is labeled gross, net is labeled net,** in every table, sub-period
  included. A net-of-2bp row sold as "gross" survived three documents before an
  audit caught it. Same discipline for *price-only* vs *total-return* series —
  `^GSPC` has no dividends in it, don't call it total return.
- **Synthetic cells never wear a real-tape banner.** When the executed cells run
  the offline control and the headline numbers come from a pinned real run, the
  notebook hero says exactly that — "Beat 0 · Verdict (real tape)", numbers
  sourced to `docs/results.md` — and the real figures live in **one** place
  (a single dict at the top of `build_notebooks.py`, mirroring `results.md`).
  Hard-coded prose numbers rot: every re-run of `verify.py` triggers a notebook
  rebuild, or the numbers will drift apart (they did, seven studies' worth).
- **Friendly, human, and honest.** We write like a person explaining something they
  find genuinely cool — not like a textbook, and never like a sales deck.

---

## Definition of Done — same construction, enforced

The whole point of the house style is that every study has the **same construction**. Three
parts of that used to be silently skippable — and an audit found them broken in shipped studies
(notebooks committed without being run, test-suites never wired into CI, an empty `docs/`). So
they are no longer a matter of discipline: they are a **gate**, [`tools/check_study_completeness.py`](tools/check_study_completeness.py),
that runs first in CI and **fails the build** if a published study is missing any of them.

A study is **"published" — and held to this standard — the moment it is linked in the
[root README](README.md) studies table.** That is the desk's done-signal; until a study is in
that table it counts as work-in-progress and the gate skips it, so scaffolding commits never
turn CI red. When you add the row, the study must pass:

- [ ] **README front-card** — the ~30-line synthetic card (verdict box with shields badges, the
  *what-we-tested* paragraph, links to the two notebooks). Not a 7-beat document.
- [ ] **Both notebooks present and _executed_** — `01_for_the_curious` and `02_for_the_quants`,
  generated by `notebooks/build_notebooks.py` and then **re-run through nbconvert so every code
  cell carries its outputs and figures**. A rebuilt-but-unexecuted skeleton is a broken
  deliverable, and the gate rejects it — as it now also rejects notebooks with error
  cells or with all outputs stripped:
  `jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb 02_*.ipynb` (from the
  `notebooks/` dir — cwd matters for any `_cache/`).
- [ ] **Tests wired into CI** — if the study has a `tests/test_*.py` suite, add its step to
  [`.github/workflows/tests.yml`](.github/workflows/tests.yml). A test nobody runs guards nothing.
- [ ] **`docs/references.md`** — the literature map. `docs/` is never empty.
- [ ] **A fingerprinted results doc** — `docs/results.md` (or `docs/results_*.md`), the
  reproducible headline run with an **explicit as-of date** + content fingerprint
  (`quantlab/repro.py`). **Required for every new study** (the gate still only warns,
  for the sake of history — but as of study 50 every study on the bench has one, and
  yours will too). The doc is the single source of truth the notebooks quote from.
- [ ] **The bench map is regenerated** — if your row changes the README table, run
  `python tools/make_bench_figures.py` so [`docs/bench_map.png`](docs/bench_map.png) and
  the counts in [`docs/bench.md`](docs/bench.md) stay in sync with the table they parse.
- [ ] **Caches for the pinned run exist locally** — `_cache/` is gitignored, so third-party
  fingerprint verification goes through the release bundle
  (`tools/make_repro_bundle.py`, policy in [`docs/reproducibility.md`](docs/reproducibility.md)).
  Keep the parquets your `verify.py` pinned; they are the proof behind your fingerprint.
- [ ] **Dual-track asides** — `02_for_the_quants` carries `> 💡 In plain words` asides; the
  curious notebook may carry `> 🔬 For the quants` asides.

Run the gate locally before you publish: `python tools/check_study_completeness.py`.

---

## Reusing the template

Starting a new study? Copy [`studies/_TEMPLATE/`](studies/_TEMPLATE/) to
`studies/NN-your-idea/`. The template README is the **front-card** — fill the verdict
box, the *what-we-tested* paragraph and the links, and keep it to ~30 lines; the seven
beats themselves go in the two notebooks. Lean on `quantlab/` for the numbers so your
results line up with everyone else's. When in doubt, re-read beat 6 — *"could you
actually trade it?"* — because that's the beat that separates this desk from a blog.

---

*Part of [Open-Alpha-Lab](README.md). Not investment advice — research and
education. See [LICENSE](LICENSE).*
