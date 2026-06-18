"""Verify whether the top-level CutEEG_Cognitive *.mat files contain the SAME
EEG data as the EEG_All_Fixations / EEG_All_Questions files.

Compares the actual numeric arrays inside each .mat (header-independent), so a
file re-saved at a different time is still detected as identical content.

Usage:
    python scripts/compare_cognitive_vs_all.py            # scan all subjects, print summary
    python scripts/compare_cognitive_vs_all.py S13        # detailed report for one subject
"""
import sys
import hashlib
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset" / "iStar_data"


def main_array(path):
    """Return (key, ndarray) for the largest numeric variable in a .mat file."""
    m = sio.loadmat(path)
    cands = []
    for k, v in m.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        if a.dtype != object and np.issubdtype(a.dtype, np.number) and a.size:
            cands.append((k, a))
    if not cands:
        return None, None
    return max(cands, key=lambda kv: kv[1].size)


def content_hash(a):
    """Hash of array shape + values, independent of the .mat file header."""
    a = np.ascontiguousarray(a.astype(np.float64))
    h = hashlib.sha1()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def scan_subject(sdir):
    """Return list of (relpath, shape, hash, key) for every .mat under CutEEG_Cognitive."""
    root = sdir / "CutEEG_Cognitive"
    if not root.is_dir():
        return []
    rows = []
    for p in sorted(root.rglob("*.mat")):
        try:
            key, a = main_array(p)
            if a is None:
                rows.append((p.relative_to(root), None, "NO_NUMERIC_VAR", key))
            else:
                rows.append((p.relative_to(root), a.shape, content_hash(a), key))
        except Exception as e:  # noqa: BLE001
            rows.append((p.relative_to(root), None, f"ERR:{e}", None))
    return rows


def folder_of(rel):
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "(top-level)"


def detail(subject):
    sdir = DATA_DIR / subject
    rows = scan_subject(sdir)
    print(f"\n================ {subject} ================")
    by_hash = {}
    for rel, shape, h, key in rows:
        by_hash.setdefault(h, []).append((rel, shape))

    top = [(rel, shape, h) for rel, shape, h, key in rows if folder_of(rel) == "(top-level)"]
    print(f"\nTop-level cognitive files: {len(top)}")
    for rel, shape, h in sorted(top, key=lambda r: str(r[0])):
        dupes = [str(r) for r, s in by_hash.get(h, []) if str(r) != str(rel)]
        if str(h).startswith(("ERR", "NO_")):
            print(f"  {rel.name:24s} shape={shape}  {h}")
        elif dupes:
            print(f"  {rel.name:24s} shape={str(shape):14s} == {', '.join(dupes)}")
        else:
            print(f"  {rel.name:24s} shape={str(shape):14s} UNIQUE (no copy elsewhere)")

    # groups that span more than one folder
    print("\nIdentical-content groups spanning multiple folders:")
    found = False
    for h, items in by_hash.items():
        if str(h).startswith(("ERR", "NO_")):
            continue
        folders = {folder_of(rel) for rel, _ in items}
        if len(items) > 1 and len(folders) > 1:
            found = True
            shape = items[0][1]
            print(f"  shape={str(shape):14s} :: " + "  ==  ".join(str(r) for r, _ in items))
    if not found:
        print("  (none)")


def summary():
    subjects = sorted(
        [p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda s: int(s[1:]) if s[1:].isdigit() else 999,
    )
    print(f"{'Subj':5s} {'#top':>4s} {'dupOfAll':>8s} {'unique':>6s}  verdict")
    for subj in subjects:
        rows = scan_subject(DATA_DIR / subj)
        if not rows:
            print(f"{subj:5s} {'--':>4s}  (no CutEEG_Cognitive folder)")
            continue
        by_hash = {}
        for rel, shape, h, key in rows:
            if not str(h).startswith(("ERR", "NO_")):
                by_hash.setdefault(h, []).append(rel)
        top = [(rel, h) for rel, shape, h, key in rows if folder_of(rel) == "(top-level)"]
        dup, uniq = 0, 0
        for rel, h in top:
            others = [r for r in by_hash.get(h, []) if str(r) != str(rel)]
            if any(folder_of(r) != "(top-level)" for r in others):
                dup += 1
            else:
                uniq += 1
        verdict = "ALL top files duplicated in *_All_*" if uniq == 0 and dup else (
            "MIXED" if dup else "all unique")
        print(f"{subj:5s} {len(top):>4d} {dup:>8d} {uniq:>6d}  {verdict}")


def categorize(rel):
    """Label which 'other' folder a matched sibling file lives in."""
    top = rel.parts[0]
    if top == "EEG_All_Fixations":
        return "All_Fixations(old-typo)" if "AllFixaitons" in str(rel) else "All_Fixations(new)"
    return top  # EEG_All_Questions / EEG_Cognitive / EEG_Cognitive_Training


def tableall():
    """Per-file provenance of the cuteeg files we USE vs every other folder."""
    from collections import Counter

    subjects = sorted(
        [p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda s: int(s[1:]) if s[1:].isdigit() else 999,
    )
    cat_counts = Counter()
    n_top = n_unique = 0
    aq_hits = []
    table_lines = ["subject\tfile\tshape\tprovenance"]
    print(f"{'Subj':5s} {'top':>3s}  per-file provenance (counts)")
    for subj in subjects:
        rows = scan_subject(DATA_DIR / subj)
        top = [(rel, shape, h) for rel, shape, h, key in rows
               if folder_of(rel) == "(top-level)"]
        if not top:
            continue
        by_hash = {}
        for rel, shape, h, key in rows:
            if not str(h).startswith(("ERR", "NO_")):
                by_hash.setdefault(h, []).append(rel)
        sc = Counter()
        for rel, shape, h in sorted(top, key=lambda r: str(r[0])):
            others = [r for r in by_hash.get(h, []) if str(r) != str(rel)]
            sib = [r for r in others if folder_of(r) != "(top-level)"]
            n_top += 1
            if sib:
                cats = sorted({categorize(r) for r in sib})
                for c in cats:
                    cat_counts[c] += 1
                    sc[c] += 1
                prov = "; ".join(f"== {r}" for r in sib)
                if "EEG_All_Questions" in cats:
                    aq_hits.append((subj, rel.name, [str(r) for r in sib]))
            else:
                n_unique += 1
                if others:
                    prov = "UNIQUE vs other folders (internal copy: " + \
                           ", ".join(r.name for r in others) + ")"
                    sc["unique*"] += 1
                else:
                    prov = "UNIQUE"
                    sc["unique"] += 1
            table_lines.append(f"{subj}\t{rel.name}\t{shape}\t{prov}")
        print(f"{subj:5s} {len(top):>3d}  " + " ".join(f"{k}={v}" for k, v in sorted(sc.items())))

    out = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "cuteeg_provenance_table.txt"
    out.write_text("\n".join(table_lines) + "\n")

    print("\n===== AGGREGATE (all subjects) =====")
    print(f"  total cuteeg files we USE      : {n_top}")
    print(f"  identical to some OTHER folder : {n_top - n_unique}")
    print(f"  unique to cuteeg (not elsewhere): {n_unique}")
    print("  breakdown of matches by folder:")
    for k, v in cat_counts.most_common():
        print(f"    {k:26s}: {v}")
    print("\n===== CHECK 1: any cuteeg file identical to EEG_All_Questions? =====")
    if aq_hits:
        print(f"  YES - {len(aq_hits)} match(es):")
        for subj, name, m in aq_hits:
            print(f"    {subj} {name} == {m}")
    else:
        print("  NO - not one cuteeg file is identical to any EEG_All_Questions file.")
    print(f"\nFull per-file table written to: {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        tableall()
    elif len(sys.argv) > 1:
        for s in sys.argv[1:]:
            detail(s)
    else:
        summary()
