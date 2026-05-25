"""Dump every cell's text output (stdout + results) to plain-text files for review."""
import json
from pathlib import Path

NB = Path("C:/Dissertation/Thesis_optimized_final_version1.ipynb")
OUT = Path("C:/Dissertation/_outputs_dump")
OUT.mkdir(exist_ok=True)

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


def extract_text(out):
    """Return concatenated text from a single output dict."""
    t = out.get("output_type", "")
    if t == "stream":
        return "".join(out.get("text", []))
    if t in ("execute_result", "display_data"):
        data = out.get("data", {})
        if "text/plain" in data:
            chunk = data["text/plain"]
            return "".join(chunk) if isinstance(chunk, list) else chunk
        return ""
    if t == "error":
        return "\n".join(out.get("traceback", []))
    return ""


for i, c in enumerate(nb["cells"]):
    outs = c.get("outputs", [])
    if not outs:
        continue
    text_chunks = []
    for o in outs:
        t = extract_text(o)
        if t.strip():
            text_chunks.append(t)
    if not text_chunks:
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    first_line = src.split("\n")[0][:90]
    fname = OUT / f"cell_{i:03d}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"### CELL {i}  |  first line: {first_line}\n")
        f.write("=" * 80 + "\n\n")
        for j, t in enumerate(text_chunks):
            f.write(f"--- output[{j}] ---\n")
            f.write(t)
            if not t.endswith("\n"):
                f.write("\n")
            f.write("\n")

print(f"Wrote dumps to {OUT}/")
print("Files:")
for p in sorted(OUT.glob("cell_*.txt")):
    print(f"  {p.name}  ({p.stat().st_size} bytes)")
