"""Real-data run — the half-life of the *actual* Polymarket arbitrage gap.

Reads a manifest of resolved binary markets (slug + the two CLOB token ids), pulls each
token's `prices-history` into the local cache, assembles the gap panel `g = 1-(p_yes+p_no)`,
and runs the same gauntlet as the synthetic demo: episode detection → half-life (two
estimators) → bootstrap CI → survival curve → resolution sweep → retail capture. Writes a
reproducible `docs/results.md` (as-of date + a content fingerprint of the gap panel).

    # 1) build a manifest once from the Gamma API (resolved markets carry clobTokenIds):
    python examples/verify_real.py --discover --limit 120     # writes docs/markets_manifest.json
    # 2) fetch the price history into the cache and run:
    python examples/verify_real.py --fetch
    # later, offline, reproduce from cache only:
    python examples/verify_real.py

Network lives only behind `--discover` and `--fetch`. Without them the run is **cache-only**
— markets with no cached parquet are skipped, never silently re-downloaded. The CLOB
order-book snapshots froze ~2026-02-20, so depth (capacity) is only reconstructable before
then; the minute `prices-history` tape used here is a *coarse upper bound* on the true
half-life (see the data-mode note in `prediction_arb/data.py`).
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from prediction_arb import arbitrage, data, robustness
from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

MANIFEST = os.path.join(_STUDY, "docs", "markets_manifest.json")
OUT = os.path.join(_STUDY, "docs", "results.md")
OPEN_THRESHOLD = 0.03            # a 3-cent gap is the smallest we call an 'open' arbitrage


CLOB = "https://clob.polymarket.com/prices-history"


def _active_window(sess, token: str) -> tuple[int, int] | None:
    """The market's traded span, from its own coarse history (None if it never priced)."""
    r = sess.get(CLOB, params={"market": token, "interval": "max", "fidelity": 60}, timeout=20)
    if not r.ok:
        return None
    h = r.json().get("history", [])
    return (int(h[0]["t"]), int(h[-1]["t"])) if len(h) > 50 else None


def discover(limit: int, want: int) -> None:
    """Build the manifest: recently-closed liquid markets that **actually have minute history**.

    Top-volume *all-time* markets are mostly ancient (pre-CLOB) and return no history, so we
    order by recency, keep only markets whose YES leg prices, and stamp each with its active
    window so the fetch pulls minute data only where it traded.
    """
    import requests

    sess = requests.Session()
    resp = sess.get(
        "https://gamma-api.polymarket.com/markets",
        params={"closed": "true", "limit": limit, "order": "endDate",
                "ascending": "false", "volume_num_min": 50000},
        timeout=30,
    )
    resp.raise_for_status()
    out = []
    for m in resp.json():
        toks = m.get("clobTokenIds")
        if isinstance(toks, str):
            try:
                toks = json.loads(toks)
            except json.JSONDecodeError:
                toks = None
        if not toks or len(toks) != 2:
            continue
        win = _active_window(sess, str(toks[0]))
        if win is None:
            continue
        out.append({"slug": m.get("slug", m.get("question", "?"))[:64],
                    "yes_token": str(toks[0]), "no_token": str(toks[1]),
                    "start_ts": win[0], "end_ts": win[1]})
        if len(out) >= want:
            break
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {MANIFEST}  ({len(out)} markets with minute history)")


def fetch_all(manifest, start_ts: int, end_ts: int) -> None:
    """Populate the cache: per-token minute history over each market's active window."""
    import requests

    sess = requests.Session()
    for i, ref in enumerate(manifest):
        lo = ref.start_ts if ref.start_ts is not None else start_ts
        hi = ref.end_ts if ref.end_ts is not None else end_ts
        for tok in (ref.yes_token, ref.no_token):
            data.fetch_prices_history(tok, lo, hi, session=sess)
        print(f"  fetched {i + 1}/{len(manifest)}: {ref.slug[:48]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="rebuild the manifest (network)")
    ap.add_argument("--fetch", action="store_true", help="populate the price cache (network)")
    ap.add_argument("--limit", type=int, default=300, help="markets to scan in discovery")
    ap.add_argument("--want", type=int, default=25, help="markets-with-history to keep")
    ap.add_argument("--start-ts", type=int, default=1_711_929_600)   # 2024-04-01 fallback
    ap.add_argument("--end-ts", type=int, default=1_749_340_800)     # 2025-06-08 fallback
    args = ap.parse_args()

    if args.discover:
        discover(args.limit, args.want)

    if not os.path.exists(MANIFEST):
        print(f"no manifest at {MANIFEST} — run with --discover first.")
        return
    manifest = data.load_manifest(MANIFEST)
    print(f"manifest: {len(manifest)} markets")

    if args.fetch:
        fetch_all(manifest, args.start_ts, args.end_ts)

    gap = data.build_gap_panel(manifest)
    if gap.empty:
        print("cache is empty — run with --fetch to populate prices-history first.")
        return
    gap = as_of(gap, DEFAULT_AS_OF)            # pin the sample so it can't creep past the freeze
    print(f"gap panel: {gap.shape[1]} priced markets x {gap.shape[0]} minute-rows")

    eps = arbitrage.detect_all(gap, open_threshold=OPEN_THRESHOLD)
    s = arbitrage.summary(eps)
    boot = robustness.bootstrap_half_life(eps, n_boot=2000)
    sweep = robustness.resolution_sweep(gap)
    surv = robustness.survival_curve(eps)
    fp = fingerprint(gap.fillna(0.0), round_dp=4)

    # When the empirical half-life is below the 1-minute tape floor (no episode survives
    # long enough to be seen halving), illustrate capture against a *generous* upper bound:
    # the median episode duration, i.e. pretend the whole penny stays put that long.
    h_emp = s.get("half_life_median_min", float("nan"))
    floored = math.isnan(h_emp)
    h_used = max(s.get("median_duration_min", 1.0), 1.0) if floored else h_emp
    capture = robustness.retail_capture_table(h_used)

    print("\n[episode summary]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print("[bootstrap]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in boot.items()})
    print(f"[half-life] empirical={h_emp}  -> capture computed against "
          f"{'median-duration upper bound ' if floored else ''}{h_used:.2f} min")
    print("\n[resolution sweep]\n", sweep.round(3).to_string())
    print("\n[retail capture]\n", capture.round(4).to_string())

    _write_results(OUT, gap, s, boot, sweep, capture, surv, h_used, floored, DEFAULT_AS_OF, fp)
    print(f"\nwrote {OUT}")


def _write_results(path, gap, s, boot, sweep, capture, surv, h_used, floored, asof, fp):
    def md(df):
        return "```\n" + df.round(4).to_string() + "\n```"

    floor_note = (
        f"\n> **The empirical half-life is below the 1-minute floor.** "
        f"**{s.get('frac_below_floor', float('nan')):.0%}** of episodes never last long enough "
        f"to be *seen* halving — the gap is back under threshold by the next minute sample "
        f"(median episode duration **{s.get('median_duration_min', float('nan')):.0f} min**). "
        f"The capture table below is therefore computed against a *generous upper bound* "
        f"(the {h_used:.0f}-minute median duration, i.e. pretending the whole penny sits "
        f"there that long); the real number is worse.\n"
        if floored else "")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Results — Study 11 (Vanishing-Penny) on real Polymarket markets

*Generated by [`examples/verify_real.py`](../examples/verify_real.py). Gap panel:
**{gap.shape[1]}** priced binary markets, {gap.index.min()} → {gap.index.max()},
minute `prices-history` (`fidelity=1`), assembled cache-only from
[`docs/markets_manifest.json`](markets_manifest.json). As-of **{asof}**. Gap fingerprint
**`{fp}`**.*

> **Read the resolution caveat before the verdict.** The CLOB `prices-history` tape is
> sampled at ~1 minute and its order-book snapshots froze ~2026-02-20. A minute tape
> **cannot see the 2-second block** the real race runs on, so every half-life here is a
> *coarse upper bound* — the true close is faster than we can measure, which only sharpens
> the tradability verdict.
{floor_note}
## Episode summary
`{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} }`

## Bootstrap half-life CI
`{ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in boot.items()} }`

## Resolution sweep — episodes seen / duration vs sampling fidelity
*As the tape coarsens, the fast episodes vanish between samples: the count collapses, so
even the coarse half-life is selection-biased toward the rare slow survivors.*
{md(sweep)}

## Retail capture — fraction of the peak penny left after reacting (latency in minutes)
{md(capture)}

## Survival curve — fraction of episodes still open after t minutes (every 3rd row)
{md(surv.iloc[::3])}
""")


if __name__ == "__main__":
    main()
