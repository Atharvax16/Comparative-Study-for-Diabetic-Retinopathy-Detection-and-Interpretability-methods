"""Append low+mid scoping cells (markdown + code) to BOTH notebook variants
so the next time you re-upload OR re-download, the scoping is baked in.

Placement: inserted BEFORE the trailing empty cell (if one exists), else at end.
Backup files are created the first time each variant is touched.
"""
import json
import shutil
from pathlib import Path


TARGETS = [
    Path("C:/Dissertation/Thesis_optimized_final (1).ipynb"),
    Path("C:/Dissertation/Thesis_optimized_final_version1.ipynb"),
]

PASTE_FILE = Path("C:/Dissertation/COLAB_PASTE_low_mid_scoping.py")


MD_SRC = '''## V6 - Low+mid restoration scoping + Phase 5 policy fix

Final scoping pass based on the observation that high-severity inputs are clinically ungradable: restoration is only attempted at **low** and **mid** severity. The raw line in the Phase 4 plots still spans all three levels so the degradation cliff is visible, but enhancer lines stop at mid. One supplementary figure shows the cold-diffusion noise-high collapse as a documented failure mode. Phase 5 `reject` route is changed from "enhance with GenAI" to "no enhancement, flag for re-acquisition" so the routing policy is consistent with the Phase 4 scoping.
'''


def make_md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def make_code_cell(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src.splitlines(keepends=True),
    }


def insert_before_empty_tail(cells, *new_cells):
    """Insert new_cells before a trailing empty cell if one exists."""
    if cells and not "".join(cells[-1].get("source", [])).strip():
        idx = len(cells) - 1
    else:
        idx = len(cells)
    for offset, c in enumerate(new_cells):
        cells.insert(idx + offset, c)
    return idx


def patch_one(path: Path, code_src: str):
    bak = path.with_suffix(".ipynb.bak3" if "(1)" in path.name else ".ipynb.bak1")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup -> {bak.name}")
    else:
        print(f"  backup already exists at {bak.name}")

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Idempotency: don't insert twice. Detect by marker string in any cell.
    marker = "V6 - Low+mid restoration scoping"
    if any(marker in "".join(c.get("source", [])) for c in nb["cells"]):
        print(f"  [skip] marker already present in {path.name} — not re-inserting")
        return

    idx = insert_before_empty_tail(
        nb["cells"],
        make_md_cell(MD_SRC),
        make_code_cell(code_src),
    )
    print(f"  inserted V6 cells at index {idx}, {idx + 1}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  wrote {path.name}  ({len(nb['cells'])} cells total)")


def main():
    code_src = PASTE_FILE.read_text(encoding="utf-8")
    for path in TARGETS:
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        print(f"\n=== {path.name} ===")
        patch_one(path, code_src)


if __name__ == "__main__":
    main()
