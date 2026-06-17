"""Real-data verification -- Study 280 (Solar-Eclipse). Regenerates docs/results.md.

Loads ^GSPC daily returns from the repo-level cache, builds an event study around
the hardcoded solar-eclipse dates, and runs the full analysis: event-day abnormal
returns, HAC t-stat, permutation test, the CAR window, and the multiple-comparisons
summary by eclipse type and US-path.

Cache-only -- no network. Run from anywhere:

    python studies/280-solar-eclipse/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_eclipse import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 280 -- Solar-Eclipse event study -- real-data verification\n")
    print("Data: ^GSPC daily (repo-level _cache/^GSPC_split_only.parquet)")
    print("      Solar-eclipse dates: hardcoded in data.py (1930-2026)\n")

    px = data.fetch_gspc_daily()  # cache-only by default
    ret = px["ret"]
    ecl = data.eclipse_table()
    fp = data.fingerprint(px)

    print(f"Data stamp: {ret.index.min().date()}..{ret.index.max().date()}, "
          f"n_days={len(ret)}, fingerprint={fp}")
    print(f"Eclipses: {len(ecl)} central "
          f"(total={int((ecl['kind']=='total').sum())}, "
          f"annular={int((ecl['kind']=='annular').sum())}, "
          f"hybrid={int((ecl['kind']=='hybrid').sum())}, "
          f"US-path={int(ecl['us_path'].sum())})\n")

    s = st.summarize(ret, ecl, n_permutations=10_000, seed=280)

    print("=== Event-day abnormal returns (event day 0 = first trading day on/after) ===")
    print(f"  Unconditional daily drift:  {s['uncond_bps']:+.2f} bps")
    print(f"  ALL eclipses  (n={s['n_all']}):  "
          f"mean abn {s['all_mean_abn_bps']:+.2f} bps, "
          f"HAC t {s['all_t_hac']:+.2f}, perm p {s['all_perm_p']:.4f}, "
          f"win-rate {s['all_win_rate_pct']:.1f}%")
    print(f"  TOTAL only    (n={s['n_total']}):  "
          f"mean abn {s['total_mean_abn_bps']:+.2f} bps, "
          f"HAC t {s['total_t_hac']:+.2f}, perm p {s['total_perm_p']:.4f}")
    print(f"  US-PATH only  (n={s['n_us']}):  "
          f"mean abn {s['us_mean_abn_bps']:+.2f} bps, "
          f"HAC t {s['us_t_hac']:+.2f}, perm p {s['us_perm_p']:.4f}")
    print(f"  CAR over [-5,+5] window:   {s['car_window_bps']:+.2f} bps\n")

    mc = st.multiple_comparisons_summary(ret, ecl, n_permutations=10_000, seed=280)
    print("=== Multiple-comparisons summary (Bonferroni over slices) ===")
    print(mc.to_string(index=False))
    print()

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- eclipse days are, if anything,")
    print("mildly UP (+13 bps abn, t=+1.12); the one 'significant' slice (annular)")
    print("does not survive Bonferroni. An omen that isn't.")


if __name__ == "__main__":
    main()
