"""
Auto-generates MkDocs reference pages for every Python module in the codebase.

This script is executed automatically by the `mkdocs-gen-files` plugin at the
start of every `mkdocs build` and `mkdocs serve` run.  It walks the source
packages, creates one virtual Markdown page per module, and writes a
SUMMARY.md that `mkdocs-literate-nav` uses to build the navigation tree.

Effect on the live site:
  - Add a function / class  → appears in the docs on the next build / hot-reload.
  - Delete a function / class → disappears automatically.
  - Add a new .py file       → a new reference page is created automatically.
  - Delete a .py file        → its reference page disappears automatically.

No docstrings are required.  mkdocstrings renders signatures and type
annotations for every symbol regardless of whether a docstring exists.
"""

import mkdocs_gen_files
from pathlib import Path

# Absolute path to the project root (one level above this DOCS/ directory).
ROOT = Path(__file__).resolve().parent.parent

# Source packages to document.  Each must have an __init__.py so Python
# can import their modules (required by mkdocstrings).
SOURCE_PACKAGES = ["src", "app", "Visualizations"]

nav = mkdocs_gen_files.Nav()

for package in SOURCE_PACKAGES:
    package_dir = ROOT / package
    if not package_dir.exists():
        continue

    for path in sorted(package_dir.rglob("*.py")):
        # Skip __init__ modules and bytecode cache folders.
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue

        # e.g.  src/backtest_engine.py  →  src/backtest_engine
        module_rel = path.relative_to(ROOT).with_suffix("")

        # Python import identifier, e.g. "src.backtest_engine"
        ident = ".".join(module_rel.parts)

        # Virtual output path inside the docs tree,
        # e.g. "reference/src/backtest_engine.md"
        doc_path = Path("reference") / module_rel.with_suffix(".md")

        # Register with literate-nav using the module path tuple as the key.
        # IMPORTANT: SUMMARY.md lives at reference/SUMMARY.md, so links inside
        # it must be relative to reference/ — i.e. just "src/backtest_engine.md",
        # NOT "reference/src/backtest_engine.md" (which would double the prefix).
        nav[module_rel.parts] = module_rel.with_suffix(".md").as_posix()

        # Write the virtual page — just a heading and the mkdocstrings directive.
        with mkdocs_gen_files.open(doc_path, "w") as fh:
            fh.write(f"# `{ident}`\n\n")
            fh.write(f"::: {ident}\n")

        # Wire the "edit this page" link back to the real source file.
        mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(ROOT))

# Write the auto-generated navigation file consumed by mkdocs-literate-nav.
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
