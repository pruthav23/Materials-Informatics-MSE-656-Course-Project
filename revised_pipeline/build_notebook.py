"""Turn pipeline.py (with '# %%' cell markers) into a self-contained notebook, inlining features.py."""
import re
import nbformat as nbf

src = open("pipeline.py").read()
feat = open("features.py").read()

# inline the feature code so the notebook needs no extra files besides the data
src = src.replace("from features import elemental_table, pair_features\n", "")
feature_cell = (
    "# %% [markdown]\n"
    "# ## Feature construction\n"
    "# Elemental properties come from pymatgen's tabulated periodic-table data; elastic constants are completed\n"
    "# and sanity-checked with isotropic-elasticity relations; mismatch descriptors are built from raw values.\n\n"
    "# %%\n" + feat.split('"""', 2)[2].lstrip()
)
src = src.replace("# %% [markdown]\n# ## 1. Load data", feature_cell + "\n# %% [markdown]\n# ## 1. Load data", 1)

cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    is_md = block.startswith(" [markdown]")
    body = block.split("\n", 1)[1].strip("\n") if "\n" in block else ""
    if is_md:
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    elif body.strip():
        cells.append(nbf.v4.new_code_cell(body))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "GBSE_revised_pipeline.ipynb")
print(f"wrote {len(cells)} cells")
