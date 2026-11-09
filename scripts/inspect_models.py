import glob
from pathlib import Path
import joblib

print("=== CHECKING MODELS ===")
for p in sorted(Path(".").glob("0*/**/model*.joblib")):
    try:
        obj = joblib.load(p)
        n_feat = getattr(obj, "n_features_in_", None)
        classes = getattr(obj, "classes_", None)
        print(f"{p}: type={type(obj).__name__}, n_features={n_feat}, classes={classes}")
    except Exception as e:
        print(f"{p}: ERROR {e}")

print("\n=== CHECKING SCALERS ===")
for p in sorted(Path(".").glob("0*/**/scaler*.joblib")):
    try:
        obj = joblib.load(p)
        n_feat = getattr(obj, "n_features_in_", None)
        print(f"{p}: type={type(obj).__name__}, n_features={n_feat}")
    except Exception as e:
        print(f"{p}: ERROR {e}")

print("\n=== CHECKING METRICS ===")
for p in sorted(Path(".").glob("0*/**/metrics*.json")):
    print(p)
