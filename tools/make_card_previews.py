#!/usr/bin/env python
"""Card previews for the landing page — one real chart per study, as a thumbnail.

The bench's landing page ([`docs/index.html`](../docs/index.html)) lists every study as a
card: number, name, the one-line claim, two verdict stamps. That tells a reader what the
question was; it tells them nothing about *what they are about to open*. This script gives
each card a face: the study's own most representative figure, lifted straight out of its
executed notebook and shrunk to a thumbnail.

How the figure is chosen — deliberately, not at random:

* ``01_for_the_curious`` first (its charts are the ones written to be understood at a glance),
  falling back to ``02_for_the_quants``;
* among a notebook's PNG outputs, the one with the largest pixel area whose aspect ratio looks
  like a chart (between 1.2:1 and 3.6:1) — which skips colour bars, tiny legends and the
  square correlation heatmaps that read as noise at 300 px wide;
* nothing at all if the study has no executed figure. A missing preview is fine: the page
  falls back to the plain card, and ``manifest.json`` tells it which studies have one.

Notebooks on this bench run to a hundred megabytes, so the base64 blobs are pulled out with a
streaming scan rather than ``json.load`` — parsing 2 GB of JSON to find six images would take
longer than the rest of the desk's CI put together.

    python tools/make_card_previews.py                 # every published study
    python tools/make_card_previews.py 963 964 965     # just these
    python tools/make_card_previews.py --force         # rebuild existing thumbnails
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "studies")
OUT_DIR = os.path.join(ROOT, "docs", "previews")
README = os.path.join(ROOT, "README.md")

NOTEBOOKS = ("01_for_the_curious.ipynb", "02_for_the_quants.ipynb")
WIDTH = 480             # 2x the card's rendered width — sharp on a retina screen
QUALITY = 70            # WebP; ~10 kB per thumbnail at this width
MAX_CANDIDATES = 8      # images scanned per notebook before we stop reading it
MIN_BYTES = 4_000       # smaller than this is a legend or an artefact, not a chart
AR_MIN, AR_MAX = 1.2, 3.6

_MARKER = '"image/png":'


def published_studies() -> list[str]:
    """Study directory names linked in the root README — the desk's own done-signal."""
    text = open(README, encoding="utf-8").read()
    seen, out = set(), []
    for name in re.findall(r"studies/(\d+-[a-z0-9-]+)/", text):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def scan_png_blobs(path: str, limit: int = MAX_CANDIDATES) -> list[bytes]:
    """Stream a notebook and pull out up to ``limit`` PNG payloads.

    A notebook is JSON, but a *large* notebook is mostly one long base64 string per figure, so
    this walks the file with a sliding buffer and lifts each ``"image/png"`` value directly.
    Whitespace and JSON string escapes inside base64 are stripped; anything that fails to
    decode is skipped rather than raised, because a truncated blob is a reason to fall back to
    another figure, not to fail a build.
    """
    out: list[bytes] = []
    buf = ""
    grabbing = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        while len(out) < limit:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while len(out) < limit:
                if not grabbing:
                    i = buf.find(_MARKER)
                    if i < 0:
                        buf = buf[-len(_MARKER):]
                        break
                    j = buf.find('"', i + len(_MARKER))
                    if j < 0:
                        break
                    buf = buf[j + 1:]
                    grabbing = True
                    payload = ""
                k = buf.find('"')
                if k < 0:
                    payload += buf
                    buf = ""
                    break
                payload += buf[:k]
                buf = buf[k + 1:]
                grabbing = False
                raw = re.sub(r"\s|\\n", "", payload)
                try:
                    out.append(base64.b64decode(raw))
                except (binascii.Error, ValueError):
                    pass
    return out


def pick_figure(blobs: list[bytes]) -> Image.Image | None:
    """The most chart-like image in the list: biggest, with a chart's proportions."""
    best, best_area = None, -1
    fallback, fallback_area = None, -1
    for b in blobs:
        if len(b) < MIN_BYTES:
            continue
        try:
            im = Image.open(io.BytesIO(b))
            im.load()
        except Exception:
            continue
        w, h = im.size
        area = w * h
        ar = w / max(h, 1)
        if area > fallback_area:
            fallback, fallback_area = im, area
        if AR_MIN <= ar <= AR_MAX and area > best_area:
            best, best_area = im, area
    return best or fallback


def thumbnail(im: Image.Image, width: int = WIDTH) -> Image.Image:
    """Flatten onto white (notebook PNGs are transparent) and resize to ``width``."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        im = flat
    else:
        im = im.convert("RGB")
    h = max(1, round(im.height * width / im.width))
    return im.resize((width, h), Image.LANCZOS)


def build_one(name: str, force: bool = False) -> tuple[str, str]:
    """Write one study's thumbnail. Returns ``(status, detail)`` for the log."""
    num = name.split("-", 1)[0]
    out_path = os.path.join(OUT_DIR, f"{num}.webp")
    if os.path.exists(out_path) and not force:
        return "skip", "already built"
    for nb in NOTEBOOKS:
        path = os.path.join(STUDIES, name, "notebooks", nb)
        if not os.path.exists(path):
            continue
        im = pick_figure(scan_png_blobs(path))
        if im is None:
            continue
        thumb = thumbnail(im)
        os.makedirs(OUT_DIR, exist_ok=True)
        thumb.save(out_path, "WEBP", quality=QUALITY, method=5)
        kb = os.path.getsize(out_path) / 1024
        return "ok", f"{nb.split('_')[0]} {thumb.width}x{thumb.height} {kb:.0f} kB"
    return "none", "no usable figure"


def write_manifest() -> int:
    """List the study numbers that have a thumbnail, for the page to read."""
    nums = sorted(int(f[:-5]) for f in os.listdir(OUT_DIR) if f.endswith(".webp"))
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"width": WIDTH, "format": "webp", "studies": nums}, fh)
    return len(nums)


def main(argv: list[str]) -> int:
    force = "--force" in argv
    wanted = {a for a in argv if a.isdigit()}
    names = published_studies()
    if wanted:
        names = [n for n in names if n.split("-", 1)[0].lstrip("0") in
                 {w.lstrip("0") for w in wanted}]
    os.makedirs(OUT_DIR, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "none": 0}
    for i, name in enumerate(names, 1):
        status, detail = build_one(name, force)
        counts[status] += 1
        if status != "skip":
            print(f"  [{i:4d}/{len(names)}] {status:4s} {name:36s} {detail}")
        elif i % 200 == 0:
            print(f"  [{i:4d}/{len(names)}] ...")
    total = write_manifest()
    print(f"\n{counts['ok']} built, {counts['skip']} already present, "
          f"{counts['none']} without a usable figure; manifest lists {total}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
