# References — Social-Oracle 🔮

*Sources for [Study 04](../README.md). The literature map matters here because the
effect under test — that public attention moves price briefly and then reverses — is
**well documented**, so the honest verdict turns on telling a real (but small and
late) attention bump from a tradeable edge, and on stating the survivorship the data
carries.*

## The claim under test

- **"Serenity" / @aleabitoreddit** — a viral retail-investing persona (*白毛股神*,
  "white-haired stock god") whose posts are scraped into cashtag "signals". The
  open-source repos that package her method as an installable agent skill are the
  living source of the claim:
  - `haskaomni/serenity-signal-dashboard` — scrapes posts/replies/paid posts, extracts
    `$SYMBOL`, stores to SQLite, pulls Yahoo daily charts, scores names 0–100.
  - `haskaomni/serenity-skill`, `muxuuu/serenity-skill`,
    `0xAgata-prog/serenity-skill` — the research framework distilled into a reusable
    LLM coding-agent skill ("supply-chain bottleneck → screen names → output a priority
    list").
- We test the **phenomenon** the repos assume — *a public mention is followed by a
  capturable abnormal return* — not the person. See [the claim](../README.md#1--the-claim).

> The strongest steelman: a mention front-runs price. The null: the move is already
> in the pre-event drift, or is just the name's momentum, or is gone after micro-cap
> costs and the fade.

## Data

- **The feed** — a `(timestamp, ticker[, score])` CSV, the format the scrapers above
  emit. **Not bundled**: it is third-party, scraped, and licence-encumbered, and it is
  soaked in **survivorship** (you hear about the calls that worked). The study reports
  every dropped mention rather than hide the selection — see
  [`mentions.to_events`](../social_oracle/mentions.py).
- **Prices** — per mentioned name, daily OHLC(V) from Yahoo! (cached). Volume is kept
  because micro-cap **capacity** is the decisive beat-6 question.
- **Benchmark** — a broad index (default `SPY`; `IWM`/Russell 2000 is the honest tape
  for a small-cap-heavy feed) defines the **abnormal** return name − market.

## Literature map — why the signal may be *real but late and reversing*

- **Attention-driven buying.** Retail investors are net buyers of *attention-grabbing*
  stocks; the buying pressure pushes price up temporarily and **reverses**. *Barber &
  Odean (2008), "All That Glitters: The Effect of Attention…", RFS.* — the prime
  suspect for a mention-day pop that fades.
- **"Dumb money" / sentiment reversal.** Flows chasing salience earn poor subsequent
  returns. *Frazzini & Lamont (2008), "Dumb money", JFE; Baker & Wurgler (2006),
  "Investor Sentiment and the Cross-Section of Stock Returns", JF.*
- **Social media & message-board returns.** Posting volume / sentiment predicts a
  short-lived move, often reversing, strongest in small, illiquid names. *Antweiler &
  Frank (2004), JF; Da, Engelberg & Gao (2011), "In Search of Attention", JF (Google
  search volume).*
- **Momentum as the confound.** Attention follows performance, so a mention rides an
  existing run — the reason the **momentum control** (hot-streak events) is the
  decisive test, not the random-day null alone. *Jegadeesh & Titman (1993).*
- **Event-study mechanics.** Market-adjusted cumulative abnormal returns, the standard
  short-window tool. *Campbell, Lo & MacKinlay (1997), "The Econometrics of Financial
  Markets", ch. 4; MacKinlay (1997), "Event Studies in Economics and Finance", JEL.*
- **Liquidity & impact.** Square-root market impact and the capacity of a thin-name
  strategy. *Almgren et al. (2005); Tóth et al. (2011), "Anomalous price impact".*

## Method cross-links

- Random-day null, block bootstrap, deflated Sharpe: the shared desk protocol,
  [`quantlab/`](../../../quantlab/) and the [methodology](../../../METHODOLOGY.md).
- The fade and the "sell it, don't buy it" inversion are the attention-space echo of
  [Study 03](../../03-fear-gauge/)'s variance-risk-premium punchline; the random-day-
  null + clustering-bootstrap machinery is reused from
  [Study 02](../../02-falling-knife/).

*(Add precise permalinks to the specific repos/posts, and exact citations, as a live
feed is run.)*
