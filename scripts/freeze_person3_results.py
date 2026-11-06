"""
Quick script to freeze the RECORDED_TEST_F1 in both Person 3 notebooks
after the first successful run.
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

FREEZES = {
    "03_person3_decision_tree_and_random_forest/decision_tree.ipynb": {
        "metrics_file": "03_person3_decision_tree_and_random_forest/models/metrics_decision_tree.json",
        "old": "RECORDED_TEST_F1 = None   # e.g.  = 0.8120  (fill in after first run)",
        "new_prefix": "RECORDED_TEST_F1 = "
    },
    "03_person3_decision_tree_and_random_forest/random_forest.ipynb": {
        "metrics_file": "03_person3_decision_tree_and_random_forest/models/metrics_random_forest.json",
        "old": "RECORDED_TEST_F1 = None   # fill in after first successful run",
        "new_prefix": "RECORDED_TEST_F1 = "
    },
}

for nb_path, info in FREEZES.items():
    mf = Path(info["metrics_file"])
    if not mf.exists():
        print(f"[SKIP] {nb_path} — metrics file not found yet")
        continue
    with open(mf) as f:
        metrics = json.load(f)
    recorded_f1 = round(metrics["macro_f1"], 4)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if info["old"] in src:
            new_src = src.replace(
                info["old"],
                f'{info["new_prefix"]}{recorded_f1}   # frozen after first run'
            )
            cell["source"] = new_src
            changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"[OK] {nb_path} -> RECORDED_TEST_F1 = {recorded_f1}")
    else:
        print(f"[SKIP] {nb_path} — placeholder not found (already frozen?)")
