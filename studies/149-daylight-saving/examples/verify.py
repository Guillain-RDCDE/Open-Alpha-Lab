"""Real-tape verification — Study 149 (Daylight-Saving). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the ^GSPC daily return series, constructs DST-Monday
labels, runs the headline DST vs other-Monday comparison, the placebo, the season
split, and the era split. Network is touched only with --fetch.

    python studies/149-daylight-saving/examples/verify.py            # cache-only
    python studies/149-daylight-saving/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from daylight_saving import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    frame = data.fetch_daily(fetch=fetch)
    fp    = data.fingerprint(frame)
    print(f"^GSPC daily tape: {frame.index[0].date()} to {frame.index[-1].date()}  "
          f"n={len(frame)}  fingerprint={fp}")
    print()

    m = st.summarize(frame)

    print("=== DST Monday vs Other Mondays (headline) ===")
    print(f"Post-DST Monday:  n={m['n_dst']:4d}  mean={m['dst_mean_bps']:+6.2f}bps  "
          f"HAC t={m['dst_tstat']:+.2f}")
    print(f"Other Mondays:    n={m['n_other']:4d}  mean={m['other_mean_bps']:+6.2f}bps  "
          f"HAC t={m['other_tstat']:+.2f}")
    print(f"Difference:              {m['diff_bps']:+6.2f}bps  Welch t={m['welch_t']:+.2f}")
    print()

    print("=== Placebo (random Monday relabelling, 500 seeds) ===")
    print(f"Real gap:      {m['diff_bps']:+.2f} bps")
    print(f"Placebo mean:  {m['placebo_mean_bps']:+.2f} bps  std={m['placebo_std_bps']:.2f} bps")
    print(f"p-value (one-sided, gap <= null): {m['p_one_sided']:.3f}")
    print()

    print("=== Season split ===")
    print(f"Spring-forward Monday: n={m['spring_n']:3d}  mean={m['spring_mean_bps']:+6.2f}bps  "
          f"Welch t vs other={m['spring_t_vs_other']:+.2f}")
    print(f"Fall-back Monday:      n={m['fall_n']:3d}  mean={m['fall_mean_bps']:+6.2f}bps  "
          f"Welch t vs other={m['fall_t_vs_other']:+.2f}")
    print()

    print("=== Era split (cut 2000-01-01) ===")
    print(f"Pre-2000:  diff={m['pre_diff_bps']:+.2f}bps  Welch t={m['pre_welch_t']:+.2f}")
    print(f"Post-2000: diff={m['post_diff_bps']:+.2f}bps  Welch t={m['post_welch_t']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
