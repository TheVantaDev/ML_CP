import os
import soundfile as sf
import pandas as pd

raw_dir = os.path.join('person4_boosting', 'raw_audio')
catalog = []

for root, dirs, files in os.walk(raw_dir):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.wav', '.flac', '.mp3', '.ogg']:
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, raw_dir)
            source = rel_path.split(os.sep)[0]
            try:
                info = sf.info(fpath)
                catalog.append({
                    'file_name': f,
                    'relative_path': rel_path,
                    'source': source,
                    'sample_rate': info.samplerate,
                    'channels': info.channels,
                    'duration_sec': round(info.duration, 2),
                    'samples': info.frames,
                    'format': info.format,
                    'subtype': info.subtype
                })
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")

df = pd.DataFrame(catalog)
os.makedirs(os.path.join('person4_boosting', 'data'), exist_ok=True)
out_csv = os.path.join('person4_boosting', 'data', 'audio_catalog.csv')
df.to_csv(out_csv, index=False)
print(f"Cataloged {len(df)} audio files saved to {out_csv}.")
print("\nSources count:\n", df['source'].value_counts())
print("\nSample rates:\n", df['sample_rate'].value_counts())
print("\nChannels:\n", df['channels'].value_counts())
total_hours = df['duration_sec'].sum() / 3600
print(f"\nTotal audio duration: {total_hours:.2f} hours")
