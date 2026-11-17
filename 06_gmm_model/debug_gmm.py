import sys, json, joblib, numpy as np, librosa, glob
sys.path.insert(0, '.')
from src.features.extractor import extract_features
from pathlib import Path

sel = json.load(open('data/features/selected_features.json'))
feat_cols = sel['selected_features']

gmm_scaler = joblib.load('06_gmm_model/models/gmm_scaler.joblib')
meta = json.load(open('06_gmm_model/models/gmm_metadata.json'))
classes = meta['classes']
gmms = {c: joblib.load(f'06_gmm_model/models/gmm_{c}.joblib') for c in classes}

def predict_live(wav_path):
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    if len(y) < 22050*4:
        y = np.pad(y, (0, int(22050*4)-len(y)))
    else:
        y = y[:int(22050*4)]
    feats = extract_features(y, sr=22050, clip_id='test')
    vec = np.nan_to_num(np.array([feats.get(c, 0.0) for c in feat_cols]))
    vec_s = gmm_scaler.transform([vec])
    scores = {c: float(gmms[c].score_samples(vec_s)[0]) for c in classes}
    pred = max(scores, key=scores.get)
    return pred, scores

print("=== AMBIENT FILES ===")
for wav in glob.glob('data/processed/ambient/*.wav')[:8]:
    pred, scores = predict_live(wav)
    name = Path(wav).name[:30]
    v = scores['vessel_like']
    a = scores['ambient']
    u = scores['unknown']
    print(f"  {name:30s} -> {pred:12s} | vessel={v:8.1f} ambient={a:8.1f} unknown={u:8.1f}")

print("\n=== VESSEL FILES ===")
for wav in glob.glob('data/processed/vessel_like/*.wav')[:5]:
    pred, scores = predict_live(wav)
    name = Path(wav).name[:30]
    v = scores['vessel_like']
    a = scores['ambient']
    u = scores['unknown']
    print(f"  {name:30s} -> {pred:12s} | vessel={v:8.1f} ambient={a:8.1f} unknown={u:8.1f}")

print("\n=== UNKNOWN FILES ===")
for wav in glob.glob('data/processed/unknown/*.wav')[:5]:
    pred, scores = predict_live(wav)
    name = Path(wav).name[:30]
    v = scores['vessel_like']
    a = scores['ambient']
    u = scores['unknown']
    print(f"  {name:30s} -> {pred:12s} | vessel={v:8.1f} ambient={a:8.1f} unknown={u:8.1f}")
