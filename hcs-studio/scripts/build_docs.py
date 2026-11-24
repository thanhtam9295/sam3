"""Build the user guide as static HTML (shipped with the Windows installer).

python scripts/build_docs.py <output folder>
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

from markdown_it import MarkdownIt

DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = [
    ("index.md", "User guide"),
    ("install.md", "Installing"),
    ("troubleshooting.md", "Troubleshooting"),
    ("acceptance.md", "Acceptance tests"),
    ("processing.md", "Processing modules"),
    ("measurements.md", "Measurement columns"),
    ("annotation.md", "Plate annotation"),
    ("pharmacology.md", "Pharmacology methods"),
    ("figures.md", "Comparison and figures"),
    ("ilastik.md", "ilastik"),
    ("registry.md", "Experiment registry"),
    ("image_qc.md", "Image QC"),
    ("assays.md", "Oncology assays"),
    ("spheroids.md", "Spheroids and organoids"),
    ("kinetics.md", "Live-cell kinetics"),
    ("screening.md", "Primary screens"),
    ("cellpainting.md", "Cell Painting profiling"),
    ("validation.md", "Validation"),
]
TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · HCS Studio</title>
<style>
body {{
  margin: 0;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: #1d2330;
  background: #fff;
}}
nav {{
  padding: 10px 24px;
  border-bottom: 1px solid #d5dae2;
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
}}
nav a {{
  color: #1a5fb4;
  text-decoration: none;
}}
nav a.here {{
  font-weight: 700;
  color: #1d2330;
}}
main {{
  max-width: 980px;
  padding: 8px 24px 40px;
}}
img {{
  max-width: 100%;
  border: 1px solid #d5dae2;
  border-radius: 6px;
}}
table {{
  border-collapse: collapse;
  font-size: 13.5px;
}}
th, td {{
  border: 1px solid #d5dae2;
  padding: 4px 8px;
  vertical-align: top;
}}
code {{
  background: #f4f6f9;
  padding: 1px 4px;
  border-radius: 4px;
}}
pre {{
  background: #f4f6f9;
  padding: 10px;
  overflow-x: auto;
  border-radius: 6px;
}}
@media (prefers-color-scheme: dark) {{ body {{ background: #14171c; color: #e4e7ec; }}
  code, pre {{ background: #1d2128; }}
  nav a.here {{ color: #e4e7ec; }}
  }}
</style></head><body><nav>{nav}</nav><main>{body}</main></body></html>
"""


def build(target: Path) -> list[Path]:
    md = MarkdownIt("commonmark").enable("table")
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, title in PAGES:
        nav = " ".join(
            f'<a href="{n[:-3]}.html"{" class=here" if n == name else ""}>{html.escape(t)}</a>'
            for n, t in PAGES
        )
        body = md.render((DOCS / name).read_text(encoding="utf-8"))
        body = re.sub(r'href="([\w-]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
        page = target / f"{name[:-3]}.html"
        page.write_text(TEMPLATE.format(title=title, nav=nav, body=body), encoding="utf-8")
        written.append(page)
    shutil.copytree(DOCS / "images", target / "images", dirs_exist_ok=True)
    return written


if __name__ == "__main__":
    pages = build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist/docs"))
    print(f"Wrote {len(pages)} pages to {pages[0].parent}")
