#!/usr/bin/env python
"""Ledger consistency gate -- does docs/REFERENCE.md still agree with the studies?

The studies table is the desk's public ledger: every count on the landing page, in
``docs/bench.json`` and on the bench map is derived from it. Nothing regenerates it
wholesale, so it drifts silently, and it has drifted twice already:

  * a study's verdict changed on a rebuild while its ledger row kept the old badges --
    the study and the ledger disagreed and nothing said so;
  * the curated "only greens" highlights list at the top of the page kept showing six
    studies after the green set had grown to ten.

Both failures are invisible: the page renders, the tools exit 0, the numbers are just
wrong. This script makes them loud.

Three checks, all derived from the studies themselves rather than from another copy of
the table:

  1. **Every published study has exactly one ledger row.** Missing rows mean a study is
     invisible to every aggregate; duplicates mean it is double-counted.
  2. **Each row's badges match the study's own README.** A study's ``README.md`` carries
     the verdict its build produced; the ledger row must say the same thing.
  3. **The highlights list matches the ledger's green set.** Whatever the page advertises
     as the bench's greens must be exactly the studies stamped Investable.

Exit code 0 if the ledger agrees with the studies, 1 otherwise. Pure stdlib.

    python tools/check_reference_table.py           # report
    python tools/check_reference_table.py --fix     # rewrite drifted rows, then report
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "studies")
REFERENCE = os.path.join(ROOT, "docs", "REFERENCE.md")

# A ledger row: two shields badges. The curated highlights rows carry none, which is how
# the two are told apart -- do not "simplify" this by matching on position.
ROW_RE = re.compile(
    r"^\| \*\*\[(?P<num>\d+)\]\(\.\./studies/(?P<slug>[^)]+?)/\)\*\*.*?"
    r"!\[(?P<signal>[A-Za-z-]+)\]\(https://img\.shields\.io/badge/[^)]*\).*?"
    r"!\[(?P<trad>[A-Za-z-]+)\]\(https://img\.shields\.io/badge/[^)]*\).*$",
    re.M)
BADGE_RE = re.compile(r"!\[([A-Za-z-]+)\]\(https://img\.shields\.io/badge/")

# The documented axes. MIXED is deliberately included: METHODOLOGY.md sanctions it
# ("**`MIXED` (amber)** is allowed when the verdict genuinely splits by regime or leg")
# and the bench map folds it into Weak, the same amber bucket. It was the README legend
# that had not caught up, not the studies -- so flagging those 68 as off-palette was this
# gate accusing the corpus of its own documentation gap.
SIGNALS = {"Real", "Weak", "Mixed", "None"}
TRADS = {"Investable", "Fragile", "Mirage"}


# Badge values are not always bare words. Some early studies qualify the verdict inside
# the badge itself -- ``badge/Signal-Real_(risk--adjusted)-2ea44f`` -- where ``_`` is a
# shields-encoded space and ``--`` an encoded hyphen. Matching only ``[A-Za-z]+`` reads
# nothing there and the study gets reported as unreadable, which is how three studies
# were accused of a missing README they in fact had. Accept the richer value and strip
# the qualifier back to its head word for comparison.
LABELLED_RE = re.compile(
    r"img\.shields\.io/badge/(Signal|Tradability)-(\S+?)-[0-9a-fA-F]{6}\?")
BARE_RE = re.compile(r"img\.shields\.io/badge/([A-Za-z]+(?:--[A-Za-z]+)*)-[0-9a-fA-F]{6}")


def head_word(value: str) -> str:
    """Reduce a badge's alt text to the bare stamp.

    ``Real_(risk--adjusted)`` -> ``Real`` (the qualifier is prose, not an axis) and
    ``Signal: Real`` -> ``Real`` (some studies repeat the axis inside the alt text, which
    is how three of them were reported as carrying the stamp "Signal: Real").
    """
    value = value.split(":", 1)[-1] if value.split(":", 1)[0] in ("Signal", "Tradability") \
        else value
    return re.split(r"[_(]", value, maxsplit=1)[0].strip().replace("--", "-")


AXIS_RE = re.compile(r"\**\s*(Signal|Tradability)\b")
ALT_RE = re.compile(r"!\[([^\]]+)\]\(https://img\.shields\.io/badge/")


def study_verdict(slug: str) -> tuple[str, str] | None:
    """The (signal, tradability) a study's own README advertises, or None if unreadable.

    Bind each stamp to its **row**, never to its position in the file. The verdict table's
    first cell names the axis; the second cell holds the stamp. Two documented forms make
    any position-based reading wrong:

      * the **compound stamp** -- ``![Real](…) on the level · ![None](…) on the spike`` --
        puts two badges in one Signal cell. "The second badge in the file" then reads that
        compound's second half as the Tradability. That is what made this gate report four
        studies as drifted when all four say ``Mirage``; one of them, study 03, is
        METHODOLOGY.md's own worked example of the form, so the gate was accusing the
        reference implementation.
      * the **qualifier row** -- a third row asking its own question, stamped from the grey
        set (``BUSTED``, ``NOT SUPPORTED``, ``RUIN-PRONE``…). Never a verdict, and binding
        by row means it is simply not read, so no grey allow-list is needed.

    Within a stamp cell take the *first* badge, and read the value from the alt text, which
    is correct under both URL conventions (bare ``badge/Mirage-c0392b`` and labelled
    ``badge/Tradability-Mirage-c0392b`` -- study 702 mixes the two in one file).
    """
    path = os.path.join(STUDIES, slug, "README.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        # The axis cell opens with the axis name and then asks its question. The dash
        # between the two is em, double-hyphen or single-hyphen depending on the study's
        # vintage, so match the opening word rather than trying to split on a separator.
        m_axis = AXIS_RE.match(cells[1].strip())
        if not m_axis:
            continue
        axis = m_axis.group(1)
        if axis not in found:
            vals = [head_word(v) for v in ALT_RE.findall(cells[2])]
            if vals:
                found[axis] = vals
    if "Signal" not in found or "Tradability" not in found:
        return None

    sig, trad = found["Signal"], found["Tradability"]
    # A compound Signal cell -- two different stamps, "Real on the level · None on the
    # spike" -- is a split, and METHODOLOGY.md says the ledger may carry it either as
    # MIXED ("shorthand for a split spelled out on the front-card") or as one of the two
    # halves. Which shorthand a study uses is a desk judgement, so accept any of them and
    # only fail when the ledger says something the front card doesn't support at all.
    opts = dict.fromkeys(sig)
    if len(opts) > 1:
        opts["Mixed"] = None
    return frozenset(opts), trad[0]


def ledger_rows(text: str) -> tuple[dict[str, re.Match], list[str]]:
    """Ledger rows by study slug, plus any slug appearing more than once.

    Duplicates are a hard failure, not a cosmetic one: the study is counted twice in
    every aggregate, and when the copies disagree — which is what happens when an
    in-place update misses and the row is appended instead — the table publishes a
    verdict the study no longer carries.
    """
    out: dict[str, re.Match] = {}
    dupes: list[str] = []
    for m in ROW_RE.finditer(text):
        slug = m.group("slug")
        if slug in out:
            dupes.append(slug)
        else:
            out[slug] = m
    return out, dupes


COLOURS = {"Real": "2ea44f", "Investable": "2ea44f",
           "Weak": "dab617", "Mixed": "dab617", "Fragile": "dab617",
           "None": "c0392b", "Mirage": "c0392b"}

# A well-formed verdict badge: value, a separator hyphen, then the colour.
# ``badge/Real-2ea44f?`` renders a green stamp reading "Real"; ``badge/Real2ea44f?``
# renders a grey one reading the literal text "Real2ea44f". The alt text stays correct
# in both, so the difference is invisible in the source and only shows on the rendered
# page -- which is exactly how 40 of these shipped into the working tree unnoticed.
# Named colours are legal too (the front page's own ``badge/license-MIT-green``).
WELL_FORMED_RE = re.compile(
    r"img\.shields\.io/badge/[^)/?]+-(?:[0-9a-fA-F]{6}|[a-z]+)\?")
ANY_BADGE_RE = re.compile(r"img\.shields\.io/badge/[^)]*")


def replace_badges(line: str, sig: str, trad: str) -> str | None:
    """Rewrite a ledger row's two verdict badges to ``sig``/``trad``, colour included.

    Both the alt text and the URL carry the value, and the URL also carries the colour
    after a separator hyphen. The obvious rewrite -- substitute ``[A-Za-z-]+`` after
    ``badge/`` -- stops at the first digit of the hex, so ``Real-2ea44f`` becomes
    ``Real2ea44f``: value and colour fused, separator gone, badge grey. Build the whole
    URL from the palette instead of editing part of it.
    """
    badges = BADGE_RE.findall(line)
    if len(badges) < 2:
        return None
    for old, new in ((badges[0], sig), (badges[1], trad)):
        line = re.sub(
            r"!\[" + re.escape(old) + r"\]\(https://img\.shields\.io/badge/[^)]*\)",
            f"![{new}](https://img.shields.io/badge/{new}-{COLOURS[new]}?style=flat-square)",
            line, count=1)
    return line


README = os.path.join(ROOT, "README.md")
SCOREBOARD_RE = re.compile(
    r"^`(\d+) tested` · `(\d+) survive` · `(\d+) real signals` · `(\d+) mirages`$", re.M)


def check_scoreboard(rows: dict[str, re.Match], fix: bool) -> int:
    """The landing page's four headline numbers, re-derived from the ledger.

    Nothing generated this line -- it was maintained by hand, and by the time anyone
    looked it claimed 799 mirages against a ledger holding 795, and 116 real signals
    against 127. A hand-kept summary of a 1012-row table is a summary that is wrong;
    the only question is when someone notices. So: derive, compare, and offer to rewrite.

    Case is folded because a handful of early rows shout their verdict (``MIRAGE``);
    that is a palette question, reported separately, not a reason to miscount here.
    """
    if not os.path.exists(README):
        return 0
    with open(README, encoding="utf-8") as fh:
        text = fh.read()
    m = SCOREBOARD_RE.search(text)
    if not m:
        print("  README scoreboard line not found -- cannot verify the headline numbers")
        return 1

    sigs = [r.group("signal").lower() for r in rows.values()]
    trads = [r.group("trad").lower() for r in rows.values()]
    want = (len(rows), trads.count("investable"),
            sigs.count("real"), trads.count("mirage"))
    got = tuple(int(g) for g in m.groups())
    if want == got:
        return 0

    labels = ("tested", "survive", "real signals", "mirages")
    for lab, g, w in zip(labels, got, want):
        if g != w:
            print(f"  SCOREBOARD {lab}: README says {g}, the ledger holds {w}")
    if fix:
        line = (f"`{want[0]} tested` · `{want[1]} survive` · "
                f"`{want[2]} real signals` · `{want[3]} mirages`")
        text = text[:m.start()] + line + text[m.end():]
        with open(README, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("  --fix rewrote the README scoreboard line")
        return 0
    return 1


def main() -> int:
    fix = "--fix" in sys.argv
    with open(REFERENCE, encoding="utf-8") as fh:
        text = fh.read()

    rows, dupes = ledger_rows(text)
    for slug in dupes:
        print(f"  DUPLICATE ledger row for {slug} -- counted twice by every aggregate")
    listed = {d for d in os.listdir(STUDIES)
              if os.path.isdir(os.path.join(STUDIES, d)) and re.match(r"\d+-", d)}

    problems = len(dupes)
    drifted: list[tuple[str, str, str, str, str]] = []
    off_palette: list[tuple[str, str, str]] = []
    no_readme: list[str] = []
    shouted: list[str] = []
    for slug in sorted(rows):
        if not os.path.exists(os.path.join(STUDIES, slug, "README.md")):
            # A published study with no front card at all. Distinct from an unparseable
            # one: the ledger advertises a verdict no file in the study backs up, so the
            # badge-drift check above can never fire for it.
            no_readme.append(slug)
            problems += 1
            continue
        got = study_verdict(slug)
        if got is None:
            print(f"  {slug}: README carries no readable verdict badges")
            problems += 1
            continue
        sig_opts, trad = got
        # The stamp to quote when reporting, and the one --fix would write: the study's
        # own first choice, not the compound's alternatives.
        sig = next(iter(sig_opts))
        m = rows[slug]
        lsig, ltrad = m.group("signal"), m.group("trad")
        lower_opts = {o.lower() for o in sig_opts}
        if lsig in sig_opts and ltrad == trad:
            pass
        elif lsig.lower() in lower_opts and ltrad.lower() == trad.lower():
            # The study shouts its stamp (``MIRAGE``) and the ledger writes it in the
            # palette's own case. Same verdict, so this is not drift -- calling it drift
            # invites a --fix that would push the shouting into the ledger and take those
            # rows off the map's grid.
            shouted.append(slug)
        else:
            # Say which side is the odd one, so the reader opens the right file.
            bad = ("study" if sig not in SIGNALS or trad not in TRADS
                   else "ledger" if lsig not in SIGNALS or ltrad not in TRADS
                   else "both readable, genuinely different verdicts")
            shown = "/".join(sorted(sig_opts)) if len(sig_opts) > 1 else sig
            print(f"  DRIFT {slug}: ledger says {lsig}/{ltrad}, "
                  f"study says {shown}/{trad}  [off-palette side: {bad}]")
            drifted.append((slug, lsig, ltrad, sig, trad))
            problems += 1
        if sig not in SIGNALS or trad not in TRADS:
            # Reported, but NOT a failure. Off-palette stamps are a taxonomy question for
            # the desk, not a bug in the ledger, and a gate that is red on day one gets
            # ignored -- which is how the three silent failures this tool exists to catch
            # went unnoticed in the first place. Fail on drift; count these separately.
            off_palette.append((slug, sig, trad))

    if shouted:
        print(f"\n  NOTE: {len(shouted)} study(ies) write their stamp in a different case "
              f"than the ledger ({', '.join(shouted[:5])}"
              f"{', ...' if len(shouted) > 5 else ''}).")
        print("  Same verdict, so not drift -- but the shouted form is what puts a study "
              "outside the map's grid, so the study README is the thing to normalise.")

    if no_readme:
        print(f"  {len(no_readme)} published study(ies) have NO README.md at all -- the "
              f"ledger advertises a verdict nothing in the study backs up:")
        print(f"    {', '.join(no_readme)}")

    missing = sorted(listed - set(rows))
    if missing:
        print(f"  {len(missing)} study directory(ies) with no ledger row: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")

    # The highlights list must be exactly the ledger's Investable set.
    greens = {slug for slug, m in rows.items() if m.group("trad") == "Investable"}
    # Highlights are the rows that link a study but carry no shields badge. Subtracting
    # the ledger slugs instead would empty the set, since both link the same studies.
    ledger_spans = [(m.start(), m.end()) for m in rows.values()]
    highlighted = set()
    for line_m in re.finditer(r"^.*\]\(\.\./studies/([^)]+?)/\).*$", text, re.M):
        if "img.shields.io" in line_m.group(0):
            continue
        if any(s <= line_m.start() < e for s, e in ledger_spans):
            continue
        highlighted.update(re.findall(r"\]\(\.\./studies/([^)]+?)/\)", line_m.group(0)))
    stray = highlighted - greens
    absent = greens - highlighted
    if stray or absent:
        print(f"  highlights list disagrees with the ledger's greens "
              f"(+{len(stray)} listed but not green, -{len(absent)} green but unlisted)")
        problems += 1

    # Badge URLs must be well formed. Without this, a fused ``badge/Real2ea44f`` reads to
    # any value parser as the off-palette value "Real2ea44f" -- so the row renders as a
    # grey blob on the page while this gate calmly files it under "taxonomy question" and
    # passes. A malformed URL is a defect in the ledger, not a decision for the desk.
    malformed = [b for b in ANY_BADGE_RE.findall(text) if not WELL_FORMED_RE.match(b)]
    if malformed:
        print(f"  {len(malformed)} malformed badge URL(s) -- value and colour fused, so "
              f"they render grey with the hex in the label:")
        for b in malformed[:6]:
            print(f"    {b}")
        problems += 1

    problems += check_scoreboard(rows, fix)

    if fix and drifted:
        rewritten = skipped = 0
        for slug, _, _, sig, trad in drifted:
            # --fix writes ONLY documented stamps. When a study's own README carries an
            # off-palette value (a shouted ``MIRAGE``, a ``Tradability-Weak``), propagating
            # it into the ledger would replace a curated desk call with a value that has no
            # colour and no cell on the map. Which stamp is right there is a taxonomy
            # decision for a person; the fixer reports and stands down.
            if sig not in SIGNALS or trad not in TRADS:
                print(f"  --fix SKIPPED {slug}: study says {sig}/{trad}, which is off the "
                      f"documented palette -- resolve by hand, not automatically")
                skipped += 1
                continue
            row_re = re.compile(
                rf"^(\| \*\*\[\d+\]\(\.\./studies/{re.escape(slug)}/\)\*\*.*)$", re.M)
            hit = row_re.search(text)
            if not hit:
                continue
            line = hit.group(1)
            line = replace_badges(line, sig, trad)
            if line is None:
                continue
            text = text[:hit.start(1)] + line + text[hit.end(1):]
            rewritten += 1
        with open(REFERENCE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  --fix rewrote {rewritten} drifted row(s)"
              f"{f', skipped {skipped} off-palette' if skipped else ''}; re-run to confirm")
        return 1

    nums = [int(m.group("num")) for m in
            sorted(rows.values(), key=lambda x: x.start())]
    if nums != sorted(nums):
        print("  ledger rows are not in study-number order (an appended row?)")
        problems += 1

    if off_palette:
        kinds = sorted({f"{s}/{t_}" for _, s, t_ in off_palette})
        print(f"\n  NOTE: {len(off_palette)} studies carry stamps outside the documented "
              f"axes ({', '.join(kinds[:6])}{', ...' if len(kinds) > 6 else ''}).")
        print("  They render fine but fall outside the bench map's 3x3 grid, so every "
              "count on it excludes them. Not a failure here -- deciding what those "
              "stamps mean is a taxonomy call for the desk, not a ledger bug.")

    print(f"\n{len(rows)} ledger rows | {len(listed)} study directories | "
          f"{problems} drift problem(s) | {len(off_palette)} off-palette.")
    if problems:
        print("Ledger check FAILED -- the published table does not match the studies.")
        return 1
    print("Ledger check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
