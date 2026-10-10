# Reproducibility — how to verify the published numbers

Every study's headline numbers are produced from cached market-data snapshots and stamped
with **data fingerprints** (short hashes of the exact series used), quoted in each study's
`docs/results*.md`. The policy:

- **Caches are gitignored.** The `_cache/*.parquet` snapshots (~hundreds of MB, and some
  upstream licences forbid in-tree redistribution) are never committed — only their
  fingerprints are, inside the results docs.
- **A release bundle carries the snapshots.** `python tools/make_repro_bundle.py` collects
  every existing cache (root `_cache/` plus each study's `_cache/`) into
  `repro_bundle_<date>.zip`, with a `manifest.json` listing path, size and sha256 per file.
  The bundle is attached to the matching GitHub Release, so a third party can drop the
  caches in place and reproduce the published fingerprints **byte-for-byte**.
  (`--dry-run` lists what would be bundled without writing anything.)
- **Or rebuild from the vendor.** Each study's `examples/verify*.py --fetch` re-downloads
  the data and re-runs the audit. Beware **vendor drift**: Yahoo occasionally restates
  history (splits, dividends, late corrections), so a fresh fetch can yield slightly
  different fingerprints and slightly different numbers. That is expected — the release
  bundle is the byte-exact reference; a fresh fetch checks that the *conclusions* survive
  today's data.

In short: same caches → same fingerprints → same numbers, to the decimal. Fresh data →
possibly new fingerprints → the verdicts should still hold.

## Frozen tapes that ship inside packages (studies 1013–1027)

One lot reads no vendor cache at all. [`quantlab/bundled.py`](../quantlab/bundled.py) loads real
market and macro series that are distributed *inside* well-known Python packages — `arch`
(S&P 500 and NASDAQ daily OHLCV 1999–2018, VIX, WTI, Brent, Moody's AAA/BAA from 1919,
Fama-French monthly from 1926, core CPI), `statsmodels` (US quarterly macro 1959–2009, German
inflation and rates) and one pinned `skfolio` wheel (S&P 500 index and 20 large stocks daily
1990–2022, five factor ETFs). Every file is pinned by SHA-256, and a loader refuses bytes that
differ from the pin, so these tapes cannot drift the way a refreshed vendor cache can: an
upgrade that swapped a file would fail loudly instead of moving a published number.

- `arch` and `statsmodels` are already desk dependencies — their tapes are present wherever
  the test-suite runs.
- The skfolio tapes come from one wheel downloaded by exact PyPI URL and verified against its
  published hash: `python -c "from quantlab import bundled; bundled.fetch_skfolio()"`. CI does
  this before the tests; tests that need those tapes skip if they are absent.

The price of that reproducibility is the window: the tapes end where the package froze them
(2018–2022), and each study's `docs/results.md` says which window its verdict rests on.
