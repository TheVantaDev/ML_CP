import json
from pathlib import Path

for nb_path in Path(".").rglob("*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        cells = len(data.get("cells", []))
        print(f"OK: {nb_path} ({cells} cells)")
