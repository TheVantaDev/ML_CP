# Project Proposal

## Sonobuoy Sound Detective
### Offline Underwater Acoustic Event Detection, Classical ML Classification, and Operator Decision-Support System

**Project type:** End-to-end Machine Learning + Signal Processing + Full-Stack Software System  
**ML approach:** Classical machine learning only — no Deep Learning / CNN / RNN  
**Proposed deployment:** Offline ground-station prototype  

---

## 1. Executive Summary

Sonobuoy Sound Detective is an offline software system that helps an operator review large volumes of underwater acoustic recordings collected by sonobuoys. Instead of manually listening to every recording, the system processes each audio file, identifies potentially important acoustic events, extracts handcrafted audio features, and uses a classical machine-learning model to classify the sound into a broad non-sensitive category such as vessel-like sound, marine-life sound, ambient/environmental noise, or unknown.

The project is **not only an ML classifier**. It is a complete decision-support application containing data ingestion, audio preprocessing, signal-processing algorithms, a traditional ML model, a rule-based validation engine, explainable results, a database, a web dashboard, operator feedback, and offline deployment.

> **Important scope note:** This is an academic prototype using open/public and synthetic acoustic data. It will not use classified Navy data, proprietary platform signatures, or claim operational submarine identification.

---

## 2. Problem Statement

Underwater microphones and passive acoustic sensors can record long-duration audio containing vessel noise, waves, weather-related sounds, marine-life calls, machinery-like tones, and other ambient noise. Reviewing this audio manually is slow, repetitive, and inconsistent.

The proposed system performs the first-pass analysis automatically. It helps the operator answer:

- Which audio segments may contain an acoustic event?
- At what time did the event occur?
- Is the event more likely vessel-like, marine-life, ambient noise, or unknown?
- What acoustic evidence and feature values caused the prediction?
- Should the operator accept, reject, or mark the result for future retraining?

---

## 3. Objectives

1. Build a pipeline to accept underwater audio recordings in WAV/FLAC format.
2. Clean, standardize, segment, and analyze the audio using signal-processing techniques.
3. Extract interpretable handcrafted acoustic features rather than learning features with a deep-learning model.
4. Train and compare classical ML models such as Random Forest, XGBoost, SVM, KNN, and Logistic Regression.
5. Detect tonal or high-energy candidate events before classification.
6. Add a rule-based acoustic-signature cross-check for low-confidence or contradictory predictions.
7. Build an offline dashboard for audio review, spectrogram visualization, detections, confidence, and feedback.
8. Store audit records and operator feedback for model improvement.
9. Demonstrate an end-to-end working prototype on public and synthetic datasets.

---

## 4. Scope and Safety Boundaries

### In scope

- Passive underwater audio processing.
- Broad academic labels: vessel-like, marine-life, ambient noise, unknown.
- Public/open-source recordings and simulated data.
- Classical ML classification using engineered features.
- An offline local decision-support dashboard.
- Explainable predictions and operator feedback.

### Out of scope

- Classified or defense-restricted acoustic recordings.
- Identification of military assets or real submarine classes.
- Real-time operational targeting, tracking, or autonomous decision-making.
- Claims of military-grade detection accuracy.

This scoped version is academically feasible, ethically appropriate, and can still demonstrate an engineering-grade ML system.

---

## 5. Data Strategy

Because real sonobuoy mission data is restricted, the project will use a layered data strategy.

| Data source | Purpose in project | Proposed labels | How it will be used |
|---|---|---|---|
| ShipsEar dataset | Primary vessel-noise data | Vessel-like; optionally boat/fishing/tug/motorboat subclasses when labels allow | Train and evaluate the vessel class |
| DeepShip dataset | Additional real ship-radiated audio | Cargo, tanker, tug, passenger; later merged to vessel-like for broad classification | Improve vessel diversity and test generalization |
| NOAA passive acoustic archives / SanctSound | Ambient ocean noise and, where metadata allows, biological/environmental audio | Ambient, marine-life, unknown | Add realistic background conditions and test robustness |
| Synthetic generator | Controlled audio for testing edge cases | Vessel-like, tonal signal, ambient, unknown | Generate controlled SNR, noise level, tonal frequency, and duration variations |
| Optional self-recorded non-sensitive samples | Demonstration/testing only | Ambient/unknown | Validate file ingestion and dashboard behavior |

ShipsEar contains underwater recordings from different vessel types and natural background noise. DeepShip is reported to contain 47 hours and 4 minutes of real underwater recordings from 265 ships across four vessel classes. NOAA’s passive acoustic archive provides publicly accessible raw recordings and soundscape data, including SanctSound data collected through the NOAA–U.S. Navy monitoring project. [web:52][web:56][web:43][web:50]

### 5.1 Final recommended class design

For the first version, use **four broad classes**:

- **Vessel-like sound** — ship, boat, tug, fishing-vessel, or machinery-like radiated noise.
- **Marine-life sound** — whale/dolphin/fish/bioacoustic calls when suitable public labelled audio is available.
- **Ambient/environmental noise** — waves, rain, wind, ocean background, and other non-target soundscape audio.
- **Unknown/other** — low-quality clips, mixed sound, unsupported source types, or sounds with no reliable label.

A broad class design is more realistic than claiming exact platform recognition when data is public and limited.

### 5.2 Data collection and organization

All data will be converted into a consistent local structure:

```text
project-data/
  raw/
    shipsear/
    deepship/
    noaa/
    synthetic/
  processed/
    vessel_like/
    marine_life/
    ambient/
    unknown/
  metadata/
    master_metadata.csv
    labels.csv
```

The master metadata file will keep:

| Field | Description |
|---|---|
| clip_id | Unique ID for each audio clip |
| source_dataset | ShipsEar, DeepShip, NOAA, Synthetic, etc. |
| original_file | Original filename/path |
| label | Final broad class |
| subclass | Original source class if available |
| duration_seconds | Audio duration |
| sample_rate | Original sample rate |
| processed_sample_rate | Standardized sample rate |
| noise_condition | Clean, mild, medium, high, simulated |
| split | Train, validation, or test |
| license_or_attribution | Dataset citation and usage details |

### 5.3 Dataset preparation rules

- Keep recordings from the same original source/session/group in only one split where possible, to reduce data leakage.
- Split data by source recording rather than randomly splitting nearly identical windows into both training and test sets.
- Use a stratified split where class counts allow it.
- Preserve the original source and label mapping in metadata.
- Remove corrupted, silent, duplicate, and extremely short clips.
- Do not label data manually without documenting the reason and source.

---

## 6. Complete End-to-End Architecture

```text
                         OFFLINE GROUND-STATION SYSTEM

[Public / Synthetic Audio Files]
              |
              v
[File Upload + Metadata Registration]
              |
              v
[Audio Quality Check]
- format, duration, sample rate, silence/corruption check
              |
              v
[Signal Preprocessing]
- mono conversion, resampling, filtering, normalization, segmentation
              |
              +------------------------------+
              |                              |
              v                              v
[LOFAR/Spectrogram Visualization]     [Candidate Event / Tonal Detection]
              |                              |
              +--------------+---------------+
                             |
                             v
[Handcrafted Feature Extraction]
- MFCC, spectral, energy, temporal, tonal features
                             |
                             v
[Feature Scaling + Feature Selection]
                             |
                             v
[Classical ML Ensemble / Best Model]
- Random Forest / XGBoost / SVM
                             |
                             v
[Confidence + Rule-Based Cross-Check]
                             |
                             v
[Detection Record + SQLite Database]
                             |
                             v
[Dashboard: waveform, spectrogram, predictions, evidence, feedback]
                             |
                             v
[Operator Accept / Reject / Correct Label]
                             |
                             v
[Feedback Store -> Periodic Offline Retraining]
```

---

## 7. Detailed Workflow

### Step 1: Audio ingestion

The operator uploads one audio file or a batch of files through the dashboard. The backend validates the following before analysis:

- Supported extension: WAV, FLAC, or MP3 for demonstration.
- File size and duration limits.
- Readability and valid sample rate.
- Presence of audio signal instead of a corrupt/silent file.
- Unique file hash to avoid duplicate processing.

**Output:** A registered audio-record record with a unique analysis ID.

### Step 2: Audio preprocessing

Raw audio from different datasets will vary in duration, channels, sample rate, loudness, and noise. Preprocessing ensures that every clip is comparable.

| Operation | Method/tool | Why it is required |
|---|---|---|
| Convert to mono | Librosa / SciPy | Creates a single consistent signal channel |
| Resample audio | Librosa | Standardizes all files, e.g. 22,050 Hz or 16,000 Hz |
| Remove DC offset | NumPy/SciPy | Centers the waveform around zero |
| Band-pass filtering | SciPy Butterworth filter | Retains a relevant frequency band and reduces unwanted components |
| Noise reduction | Spectral gating or simple noise-profile subtraction | Reduces steady background noise for cleaner features |
| Amplitude normalization | RMS or peak normalization | Makes loudness differences less dominant |
| Segmentation | Fixed 3–5 second windows with optional overlap | Converts long recordings into model-ready samples |
| Silence removal | Energy threshold / voice activity-style rule | Avoids classifying empty audio |

**Output:** Clean, normalized segments, each linked to the original file and timestamp.

### Step 3: Signal visualization and candidate-event detection

For every segment, the system generates a time-frequency representation (spectrogram or LOFAR-style display). This is not fed to a CNN; it is used for human interpretation and signal-analysis algorithms.

Candidate events will be identified using classical methods:

- Short-time energy thresholding to find high-energy time regions.
- Peak finding on the power spectrum to identify strong frequency peaks.
- Tonal-line persistence check to identify narrow-band components that remain across multiple frames.
- Signal-to-noise-ratio estimate to flag low-quality segments.

**Output:** Event start/end time, peak frequency, estimated SNR, and event flag.

### Step 4: Handcrafted feature extraction

Instead of deep learning, each audio segment becomes a structured numeric feature vector. The model learns patterns from these explainable features.

| Feature family | Features | Meaning |
|---|---|---|
| Cepstral features | MFCC mean, standard deviation, min, max; delta MFCC | Captures spectral-envelope/timbre characteristics |
| Energy features | RMS energy, frame energy mean/std, dynamic range | Measures sound intensity and energy variation |
| Spectral features | Spectral centroid, bandwidth, rolloff, flatness, contrast | Describes where energy exists across frequencies and how tonal/noisy it is |
| Frequency features | Dominant frequency, top peak frequencies, peak amplitude, frequency spread | Captures characteristic tonal components |
| Temporal features | Zero-crossing rate, duration, attack/decay statistics | Captures changes in waveform over time |
| Rhythm/modulation features | Tempo or modulation-energy statistics where meaningful | Helps distinguish continuous machinery-like noise from intermittent sounds |
| Tonal/discrete features | Number of persistent peaks, tonal-to-broadband ratio, peak persistence | Provides interpretable evidence for tonal/vessel-like signals |
| Quality features | Estimated SNR, silence percentage, clipping percentage | Supports confidence and quality checks |

For each frame-level feature, summary statistics such as mean, standard deviation, minimum, maximum, median, and percentiles will be calculated. The final vector may contain roughly 60–150 features depending on the final selection.

**Output example:**

```text
clip_id, mfcc_1_mean, mfcc_1_std, spectral_centroid_mean,
rolloff_mean, rms_mean, zcr_mean, dominant_frequency,
tonal_peak_count, tonal_persistence, snr_estimate, label
```

### Step 5: Feature engineering and selection

The feature table will be prepared before model training.

- Handle missing/infinite values.
- Standardize features using `StandardScaler` for SVM, KNN, and Logistic Regression.
- Keep raw tree-friendly values for Random Forest and XGBoost where appropriate.
- Remove duplicate or highly correlated features.
- Use ANOVA F-score, mutual information, recursive feature elimination, or tree-based feature importance to choose useful features.
- Save the final selected-feature list with the model so production inference uses the exact same feature order.

**Output:** Versioned feature matrix (`features.csv` or Parquet) and preprocessing pipeline.

### Step 6: Classical ML model training

The core model will use only classical ML algorithms.

| Model | Role | Reason for inclusion |
|---|---|---|
| Logistic Regression | Baseline | Simple, interpretable reference model |
| K-Nearest Neighbors | Baseline | Tests similarity-based classification |
| Support Vector Machine (SVM) | Main comparison model | Often effective for medium-sized, high-dimensional handcrafted audio features |
| Random Forest | Main model candidate | Robust to nonlinear features, provides feature importance, handles mixed feature behavior |
| XGBoost | Main model candidate | Strong tabular-data classifier and suitable for engineered feature vectors |

### Recommended final model strategy

- Train all five models with the same train/validation/test protocol.
- Tune hyperparameters using cross-validation on the training set.
- Choose the final model based on macro F1-score, per-class recall, confusion matrix, inference time, and explainability.
- Expected final choice: **Random Forest or XGBoost**.
- Preserve SVM as a strong classical baseline.

The project will explicitly state: **No CNN, RNN, LSTM, Transformer, or other deep-learning model is used.** The model works on manually extracted, explainable signal features. Classical audio ML commonly uses features such as energy, zero-crossing rate, spectral descriptors, spectrogram-derived statistics, and MFCCs before applying a classifier; recent comparative work includes Logistic Regression, SVM, KNN, Random Forest, and XGBoost on handcrafted Librosa features. [web:29][web:31][web:32]

### Step 7: Prediction and confidence scoring

For every segment, the chosen model returns:

- Predicted broad class.
- Probability/confidence for each class.
- Top contributing features.
- Data-quality indicators such as SNR and silence percentage.
- Candidate-event metadata such as dominant frequency and tonal peak count.

**Low-confidence policy:**

- If maximum class probability is below a defined threshold (for example 0.60 after validation), mark the result as **Review Required**.
- If the segment has poor SNR or no reliable event detection, return **Unknown/Low Quality** rather than forcing a vessel-like prediction.

### Step 8: Rule-based acoustic cross-check

This is the main non-ML intelligence layer. A small local rule library validates whether the model prediction agrees with observed signal properties.

Example academic rules:

| Condition | Rule response |
|---|---|
| Model says vessel-like, but SNR is very low | Mark as low-confidence; request operator review |
| Model says vessel-like, but no persistent frequency peaks are present | Add a warning: weak tonal evidence |
| Model says ambient, but high persistent tonal energy is detected | Flag as possible missed event |
| Audio has high clipping or excessive silence | Mark input as poor quality; avoid high-confidence claim |
| Model confidence is below threshold | Final label becomes Review Required / Unknown |

This rule engine does not claim to identify real Navy signatures. It only validates generic acoustic evidence and prevents overconfident predictions.

### Step 9: Database and audit trail

SQLite will be used in the prototype because it is lightweight and works completely offline. PostgreSQL can be used later if multi-user deployment is required.

Suggested tables:

| Table | Main stored information |
|---|---|
| users | User ID, username, hashed password, role |
| audio_files | File ID, hash, source, path, upload time, duration, sample rate |
| analysis_runs | Model version, pipeline version, processing time, status |
| detections | Segment timestamps, predicted class, confidence, event data, review status |
| feature_summary | Selected feature values or a serialized reference |
| feedback | Operator decision, corrected label, note, timestamp |
| audit_logs | Login, upload, analysis, result review, export attempts |
| model_registry | Model filename, feature list, metrics, dataset version, approval status |

### Step 10: User interface/dashboard

The dashboard will make the project a usable software product rather than a notebook-only model.

#### Main pages

| Page | Functions |
|---|---|
| Login page | Local authentication and role-based access |
| Upload page | Upload WAV/FLAC files; show file validation status |
| Analysis queue | Start batch analysis; show pending/completed/failed files |
| Audio analysis page | Waveform, spectrogram/LOFAR-style view, playback, detected event markers |
| Detection results page | Timestamp, predicted class, confidence, SNR, dominant frequency, review status |
| Explainability page | Feature values, feature importance, rule-engine warnings, prediction probability chart |
| Feedback page | Accept, reject, or correct prediction; add operator note |
| Model dashboard | Dataset version, model version, test metrics, confusion matrix, selected features |
| Audit page | User activity and system events |

#### Key UI output for each detection

```text
Detection ID: DET-00041
File: buoy_recording_12.wav
Time window: 00:02:15 to 00:02:20
Predicted class: Vessel-like sound
Model confidence: 0.78
Dominant frequency: 182 Hz
Estimated SNR: 14.2 dB
Tonal peaks detected: 4
Rule-engine status: Consistent — no critical warning
Operator status: Pending review
```

### Step 11: Human-in-the-loop feedback and retraining

The operator can accept, reject, or correct each prediction. Feedback is stored separately from initial labels so every correction is traceable.

Retraining will be offline and manual/approved:

1. Admin exports reviewed feedback.
2. Data is checked for quality and class balance.
3. A new feature dataset is generated.
4. The model is retrained and evaluated against a held-out test set.
5. The new model is registered with a version number.
6. The admin activates the new model only if its metrics meet the existing model or improvement criterion.

The system will never silently retrain itself from unverified feedback.

---

## 8. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming language | Python 3.x | Audio processing, ML backend, API |
| Audio loading and feature extraction | Librosa | MFCC, spectral, temporal, and audio utility functions |
| Signal processing | SciPy, NumPy | Filtering, FFT, peak detection, numerical processing |
| Classical ML | Scikit-learn | SVM, Random Forest, KNN, Logistic Regression, preprocessing, metrics |
| Gradient boosting | XGBoost | Final-model candidate for feature-vector classification |
| Data handling | Pandas | Metadata and feature tables |
| Visualization | Matplotlib / Plotly | Spectrograms, plots, model metrics |
| Backend API | FastAPI | Upload, analysis, inference, database APIs |
| Frontend option A | Streamlit | Fastest dashboard prototype for academic submission |
| Frontend option B | React + Vite | More polished full-stack web interface |
| Database | SQLite | Offline storage for prototype |
| Authentication | Local hashed passwords / JWT | Basic local role-based access |
| Packaging | Docker Compose, optional | Repeatable offline deployment |
| Model tracking | Joblib + model_registry table | Save model, scaler, feature list, metrics, version |

### Recommended implementation path

For the first functional version, use **Python + FastAPI + Streamlit + SQLite**. It is faster to build and easier to demonstrate.

If time permits, replace the Streamlit UI with **React + FastAPI** for a more polished final interface. The ML/data-processing backend remains unchanged.

---

## 9. API Flow

```text
POST /auth/login
POST /audio/upload
POST /analysis/{file_id}/start
GET  /analysis/{analysis_id}/status
GET  /detections?file_id=...
GET  /detections/{detection_id}
GET  /detections/{detection_id}/spectrogram
POST /detections/{detection_id}/feedback
GET  /models/current
GET  /dashboard/summary
```

### Processing flow after upload

```text
User uploads audio
  -> backend validates file
  -> backend saves original file and metadata
  -> preprocessing pipeline produces segments
  -> event detector evaluates each segment
  -> feature extractor produces numeric vector
  -> saved ML pipeline predicts class and probability
  -> rule engine validates confidence/feature evidence
  -> results saved in database
  -> dashboard reads and visualizes results
```

---

## 10. Evaluation Plan

The project should be evaluated as both an ML model and a software system.

### ML evaluation metrics

| Metric | Why it matters |
|---|---|
| Accuracy | Overall percentage of correct labels |
| Precision | Reliability when the system predicts a class |
| Recall | Ability to find examples of each class |
| Macro F1-score | Fair comparison when classes are imbalanced |
| Confusion matrix | Shows which classes the model confuses |
| ROC-AUC / one-vs-rest where appropriate | Measures class-separation quality |
| Calibration/confidence analysis | Checks whether predicted probabilities are trustworthy |
| Inference time per 5-second segment | Confirms practical processing speed |

### System evaluation metrics

- File-upload success rate.
- Processing time per minute of audio.
- Number of segments processed per batch.
- Dashboard response time.
- Percentage of low-confidence results correctly sent for review.
- Feedback capture and audit-log correctness.
- Ability to run after disabling internet access.

### Required experiments

1. Compare Logistic Regression, KNN, SVM, Random Forest, and XGBoost.
2. Compare feature sets: MFCC-only versus MFCC + spectral + tonal + energy features.
3. Test clean versus noise-added samples at different SNR levels.
4. Evaluate with and without the rule-based review policy.
5. Show feature importance for Random Forest/XGBoost, or permutation importance for any final model.
6. Test at least one unseen-source scenario if datasets permit, e.g. train with one dataset and test on another.

### Expected deliverables for evaluation

- Dataset summary table with samples per class.
- Class distribution chart.
- Feature-correlation plot.
- Model comparison table.
- Confusion matrix for final model.
- Precision/recall/F1 per class.
- Feature-importance chart.
- Screenshots of UI workflow.
- Offline deployment demonstration.

---

## 11. Proposed Timeline

| Week | Work items | Output |
|---|---|---|
| 1 | Finalize scope, class labels, collect datasets, create metadata format | Project dataset plan |
| 2 | Audio cleaning, resampling, segmentation, quality checks | Preprocessed audio pipeline |
| 3 | Build feature extraction and event-detection modules | Feature dataset + visualizations |
| 4 | Train and compare classical ML models | Baseline results and model selection |
| 5 | Add confidence thresholds, rules, explainability, save model pipeline | Inference service |
| 6 | Build FastAPI + SQLite + basic Streamlit dashboard | End-to-end working prototype |
| 7 | Add feedback, audit logs, offline packaging, testing | Complete integrated system |
| 8 | Evaluation, screenshots, report, demo/video preparation | Final submission package |

---

## 12. Key Innovations / Contributions

The contribution is not a claim of creating a military sonar classifier. The contribution is the design and implementation of a practical academic prototype with the following elements:

- **Classical ML instead of deep learning:** interpretable features and models such as Random Forest, XGBoost, and SVM.
- **Hybrid intelligence:** ML prediction combined with deterministic signal-quality and tonal-evidence rules.
- **Full signal-to-decision workflow:** raw audio to preprocessing, event detection, features, prediction, review, and feedback.
- **Explainable outputs:** features, confidence, dominant frequency, SNR, warnings, and feature importance are shown to the operator.
- **Offline-first deployment:** suitable for environments where internet connectivity is unavailable or undesirable.
- **Human-in-the-loop improvement:** verified feedback is saved for controlled periodic retraining.
- **Responsible data design:** public and synthetic data only; no restricted data or sensitive asset identification.

---

## 13. Expected Final Demonstration

A demonstration will show the following user journey:

1. Login locally to the dashboard.
2. Upload an underwater audio recording.
3. View waveform and LOFAR-style/spectrogram visualization.
4. Start analysis.
5. See detected time windows and potential acoustic events.
6. Open a detection record to view predicted class, probability, feature evidence, dominant frequency, and rule-engine status.
7. Accept, reject, or correct the classification.
8. View the audit record and feedback entry.
9. View model metrics and feature-importance chart.
10. Demonstrate that the software runs with internet disabled.

---

## 14. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Public underwater datasets have inconsistent labels | Map detailed source labels to four broad, clearly documented project classes |
| Insufficient marine-life examples | Start with vessel-like versus ambient/unknown; add marine-life only when adequate labelled data is obtained |
| Dataset imbalance | Use class weights, balanced sampling, macro F1-score, and careful data augmentation only on training data |
| Model overfits a dataset | Use source-aware splitting and test on unseen recordings/datasets where possible |
| High background noise | Include noise-robust features, SNR checks, synthetic noise tests, and an unknown/review state |
| Misleading high-confidence predictions | Add confidence threshold and rule-based cross-check before presenting final status |
| Scope becomes too large | Build in phases: ML pipeline first, then dashboard, then feedback/security/offline packaging |

---

## 15. Approval Requested

We request approval to develop **Sonobuoy Sound Detective: an Offline Underwater Acoustic Event Detection, Classical ML Classification, and Operator Decision-Support System** as an academic project.

The project will use public/synthetic non-sensitive audio data and will implement a complete system covering dataset preparation, signal processing, handcrafted feature extraction, traditional machine learning, rule-based validation, explainable predictions, offline dashboard deployment, and human feedback. It will not use deep-learning models or restricted/defense data.

---

## 16. References and Data Sources

1. ShipsEar — underwater vessel-noise database designed for researchers and containing vessel recordings plus natural background noise. [web:52]
2. DeepShip — public underwater acoustic dataset described as 47 hours 4 minutes of real recordings of 265 ships across four classes. [web:56]
3. NOAA SanctSound raw passive acoustic data — raw recordings from the NOAA–Navy Sanctuary Soundscapes Monitoring Project. [web:43][web:44]
4. NOAA Passive Acoustic Data Archive — public access to globally accessible raw underwater audio and data products. [web:50]
5. Classical audio feature examples — energy, zero-crossing rate, pitch/chroma, spectrograms, log-mel features, and MFCCs. [web:29]
6. Standard classical audio-ML pipeline and feature-selection workflow. [web:31]
7. Classical classifier comparison using handcrafted Librosa features, including Logistic Regression, SVM, KNN, Random Forest, and XGBoost. [web:32]
