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

## Two readers, one page

Every front page is written to be read **two ways at once**, so we never split the
audience or dumb anything down:

- **The plain prose is for the curious.** No jargon. You can read any study
  top-to-bottom and follow the whole story — the claim, the stakes, what we found,
  whether you could trade it — without a stats background.
- **The `> 🔬 For the quants` callouts are the rigorous layer.** Inline, right next
  to the plain sentence they back up: the HAC *t*-stat, the formula, the confound
  we ruled out and how. A pro reads these and gets the teardown; a curious reader
  skips them and loses nothing.
- **Heavy maths goes in a `<details>` fold** — there if you want it, invisible if
  you don't.

One document, two depths. That's the contract. (The narrative notebooks take this
further: `01_for_the_curious` runs the plain layer, `02_for_the_quants` runs the
deep one — but **both follow the exact same seven beats below**, so they're the
same story told at two altitudes.)

---

## The seven beats

Every front page walks these seven beats, in this order, with these headers. The
order isn't arbitrary — it's the order an honest investigation actually happens in:
state the claim fairly, work out why it'd matter, decide how you'd be wrong, do the
work, call it, ask if it pays, then open the door for the next person.

> **Beat 0 — Verdict, read first.** Before any of it: the answer, in a little box.
> Two stamps and one sentence. A reader gets the conclusion in five seconds, *then*
> chooses to follow the reasoning. We don't bury the lede.

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

---

## The shared protocol (the engine behind beat 4)

Beat 4 isn't improvised per study — it runs a fixed gauntlet, powered by the
reusable [`quantlab/`](quantlab/) engine so the *same* method produces the *same*
kind of number every time:

1. **Decompose / measure** the raw effect with exact identities — no fitting, no
   free parameters. *(`quantlab/decompose.py`)*
2. **Robust inference** — Newey-West (HAC) and Lo (2002) standard errors, bootstrap
   CIs, White (2000) Reality Check for data-snooping. *Is it real?*
   *(`quantlab/analytics.py`, `stats.py`, `bayes.py`)*
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
- **Friendly, human, and honest.** We write like a person explaining something they
  find genuinely cool — not like a textbook, and never like a sales deck.

---

## Reusing the template

Starting a new study? Copy [`studies/_TEMPLATE/`](studies/_TEMPLATE/) to
`studies/NN-your-idea/`, fill the seven beats, and keep the headers. Lean on
`quantlab/` for the numbers so your results line up with everyone else's. When in
doubt, re-read beat 6 — *"could you actually trade it?"* — because that's the beat
that separates this desk from a blog.

---

*Part of [Open-Alpha-Lab](README.md). Not investment advice — research and
education. See [LICENSE](LICENSE).*
