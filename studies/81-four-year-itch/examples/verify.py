"""Real-tape verification — Study 81 (Four-Year-Itch). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the ^GSPC daily series back to 1928, assigns Presidential-cycle
year labels, computes per-term-year stats and the year-3-vs-rest Welch test, and prints the
headline numbers. Network is touched only with --fetch.

    python studies/81-four-year-itch/examples/verify.py            # cache-only
    python studies/81-four-year-itch/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from four_year_itch import data, strategy as st  # noqa: E402

TICKER = "^GSPC"


def main(fetch: bool) -> None:
    df = data.fetch_prices(TICKER, fetch=fetch)
    fp = data.fingerprint(df)
    start = df.index[0].date()
    end = df.index[-1].date()
    n_days = len(df)
    print(f"{TICKER}: {start} to {end}  n={n_days} days  fingerprint={fp}")
    print()

    s = st.summarize(df)

    print("=== per-term-year annual returns ===")
    print(f"{'Year':>5} | {'n':>3} | {'mean %':>7} | {'std %':>6} | {'HAC t':>7}")
    print("-" * 40)
    for ty in (1, 2, 3, 4):
        print(
            f"  Yr{ty}  | {s[f'y{ty}_n']:>3} | {s[f'y{ty}_mean']:>+7.1f} | "
            f"{s[f'y{ty}_std']:>6.1f} | {s[f'y{ty}_tstat']:>+7.2f}"
        )
    print()
    print(f"Year-3 vs rest (Yrs 1+2+4): diff = {s['y3_minus_rest_pct']:+.1f} pp, "
          f"Welch t = {s['y3_vs_rest_t']:+.2f}  (n_y3={s['n_y3']}, n_rest={s['n_rest']})")
    print()

    # Per-cycle granularity (the small-n reality check)
    ann = st.annual_returns(df)
    print("=== annual returns by term year (all cycles) ===")
    for ty in (1, 2, 3, 4):
        vals = ann.loc[ann["term_year"] == ty, "annual_ret"].values
        years = ann.index[ann["term_year"] == ty].tolist()
        positive = (vals > 0).mean()
        print(f"  Yr{ty}  positive {positive:.0%}  min={vals.min()*100:+.0f}%  "
              f"max={vals.max()*100:+.0f}%  range={', '.join(str(y) for y in years[:5])}...")
    print()

    ann_spy = None
    try:
        df_spy = data.fetch_prices("SPY", fetch=fetch)
        s_spy = st.summarize(df_spy)
        print("=== SPY (post-1993 total-return) ===")
        for ty in (1, 2, 3, 4):
            print(
                f"  Yr{ty}  n={s_spy[f'y{ty}_n']:>2}  "
                f"mean={s_spy[f'y{ty}_mean']:>+6.1f}%  "
                f"t={s_spy[f'y{ty}_tstat']:>+5.2f}"
            )
        print(f"  y3 vs rest: {s_spy['y3_minus_rest_pct']:+.1f} pp  "
              f"Welch t = {s_spy['y3_vs_rest_t']:+.2f}")
    except FileNotFoundError:
        print("SPY cache not found — run with --fetch to include SPY.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
