# Central configuration for the Sonobuoy Sound Detective project
from pathlib import Path

# --- Project Root --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data Directories ----------------------------------------------------------
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
FEATURES_DIR    = DATA_DIR / "features"
METADATA_DIR    = DATA_DIR / "metadata"

RAW_SHIPSEAR    = RAW_DIR / "shipsear"
RAW_DEEPSHIP    = RAW_DIR / "deepship"
RAW_NOAA        = RAW_DIR / "noaa"
RAW_SYNTHETIC   = RAW_DIR / "synthetic"

PROC_VESSEL     = PROCESSED_DIR / "vessel_like"
PROC_MARINE     = PROCESSED_DIR / "marine_life"
PROC_AMBIENT    = PROCESSED_DIR / "ambient"
PROC_UNKNOWN    = PROCESSED_DIR / "unknown"

MASTER_METADATA = METADATA_DIR / "master_metadata.csv"
LABELS_FILE     = METADATA_DIR / "labels.csv"
FEATURES_CSV    = FEATURES_DIR / "features.csv"
FEATURES_PARQ   = FEATURES_DIR / "features.parquet"
SEL_FEATURES    = FEATURES_DIR / "selected_features.json"

# --- Model Output --------------------------------------------------------------
MODELS_DIR      = PROJECT_ROOT / "models"
REPORTS_DIR     = PROJECT_ROOT / "reports"
PLOTS_DIR       = REPORTS_DIR / "plots"

# --- Class Definitions ---------------------------------------------------------
CLASSES = {
    "vessel_like": 0,
    "marine_life": 1,
    "ambient":     2,
    "unknown":     3,
}
CLASS_NAMES = list(CLASSES.keys())
NUM_CLASSES = len(CLASSES)

# --- Audio Preprocessing Config ------------------------------------------------
SAMPLE_RATE         = 22_050       # Hz — standardised sample rate
SEGMENT_DURATION    = 4.0          # seconds
OVERLAP_FRACTION    = 0.50         # 50% hop
BANDPASS_LOW_HZ     = 20           # Hz — lower edge of bandpass
BANDPASS_HIGH_HZ    = 5_000        # Hz — upper edge
BANDPASS_ORDER      = 5            # Butterworth order
MIN_SEGMENT_ENERGY  = 1e-6        # below this ? silence
MIN_DURATION_SEC    = 1.0          # discard clips shorter than this
MAX_DURATION_SEC    = 300.0        # flag clips longer than this

# --- Feature Extraction Config -------------------------------------------------
N_MFCC              = 13
N_FFT               = 2048
HOP_LENGTH          = 512
N_MELS              = 128
TOP_K_PEAKS         = 3            # top frequency peaks to capture
TONAL_PERSIST_FRAMES= 5            # frames a peak must persist to be "tonal"

# --- Quality / SNR Config ------------------------------------------------------
MIN_SNR_DB          = 3.0          # below ? poor quality flag
MAX_SILENCE_PCT     = 0.60         # above ? discard segment
MAX_CLIPPING_PCT    = 0.05         # above ? flag clipping

# --- Data Split Config ---------------------------------------------------------
TRAIN_RATIO         = 0.80
VAL_RATIO           = 0.10
TEST_RATIO          = 0.10
RANDOM_SEED         = 42

# --- Model Config --------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.60    # below ? "Review Required"
CV_FOLDS            = 5
