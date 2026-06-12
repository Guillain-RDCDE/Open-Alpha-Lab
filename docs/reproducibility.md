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
