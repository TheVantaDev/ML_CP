import json
import sys
from pathlib import Path

for nb_name in ["decision_tree.ipynb", "random_forest.ipynb"]:
    nb_file = Path("03_person3_decision_tree_and_random_forest") / nb_name
    print(f"Testing syntax and cells of {nb_file} ...")
    with open(nb_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    print(f"  Cells: {len(nb_data['cells'])}")
    code_cells = [c for c in nb_data['cells'] if c['cell_type'] == 'code']
    print(f"  Code cells: {len(code_cells)}")
    for i, c in enumerate(code_cells):
        code_str = "".join(c["source"])
        compile(code_str, f"{nb_name}_cell_{i}", "exec")
    print(f"  [OK] All code cells in {nb_name} compiled successfully with 0 errors!")
