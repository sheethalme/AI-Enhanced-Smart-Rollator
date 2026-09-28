# Fall Detection (Member 1): AI-Enhanced Smart Rollator

Binary **fall vs non-fall** classifier on 2.5 s IMU windows (accelerometer + gyroscope, 50 Hz),
trained on two public datasets pooled into one: **SisFall** (waist-worn) and **MobiFall** (phone in pocket).

**Status:** the data pipeline, a validated RandomForest baseline, and a deployable `predict_fall()` are done.
The CNN (this branch's name) and testing on real rollator data are still to do. See [To do](#to-do).

---

## Quick start: using the trained model

```python
import sys; sys.path.append("fall-detection/scripts")
from predict import predict_fall
# window: numpy array (125, 6) = 2.5 s @ 50 Hz, columns [ax, ay, az, gx, gy, gz],
#         accel in g (including gravity), gyro in deg/s
result = predict_fall(window)
print(result)   # {'fall': True, 'probability': 0.97}
```

- Model: `models/fall/fall_rf.pkl` (joblib bundle: model + feature names + config).
- Exact input contract: `models/fall/config.json` (rate, window length, channel order, units, ±2 g clip, threshold 0.5).
- `predict.py` loads the model once at import. Each call takes about 6 ms on a laptop. It raises `ValueError` on a wrong shape or NaN input.
- Smoke test: `cd scripts && python3 predict.py`
- **Install the library versions below.** A pickled scikit-learn model only loads reliably with the same scikit-learn version (1.8.0).

```
pip install -r requirements.txt   # Python 3.12+, pandas 2.3, numpy 2.x, scipy 1.16, scikit-learn 1.8.0, pyarrow, matplotlib, joblib
```

---

## What has been done

### 1. Data (Stage -1 / 0): sources and inspection
The raw data is **not in git** (about 1 GB). Download it into `data/raw/`:

| folder | dataset | source |
|---|---|---|
| `data/raw/sisfall/SisFall_dataset/` | SisFall: 38 subjects (23 young adults SA, 15 elderly SE), 4,505 trials, 200 Hz raw ADC, waist | GitHub release [meherabmehu/elderly_fall_detection `datasets-v1`](https://github.com/meherabmehu/elderly_fall_detection/releases/tag/datasets-v1) (`SisFall.zip`; unzip the nested `SisFall_dataset.zip`) |
| `data/raw/sisfall/SisFall_enhanced/` | Per-sample labels for SisFall: 0 = normal, 1 = pre-fall (ALERT), 2 = fall (Musci et al. 2018) | [bitbucket unipv_cvmlab/sisfalltemporallyannotated](https://bitbucket.org/unipv_cvmlab/sisfalltemporallyannotated/) |
| `data/raw/mobifall/` | MobiFall v2.0: 24 subjects, phone in trouser pocket, acc m/s² (~85 Hz irregular), gyro rad/s (~200 Hz) | Kaggle (MobiFall_Dataset_v2.0) |

**Do not use the Kaggle "SisFall Enhanced" download** (the "Three Classes" folder). It contains pre-cut, shuffled windows
with no subject IDs or timestamps, so subject-independent evaluation is impossible.
`scripts/00_inspect.py` documents what's in it. Stage 0 findings are in `data/processed/00_inspect_report.txt`.

### 2. Harmonize (Stage 1): one schema, 50 Hz, g and deg/s
`harmonize_sisfall.py` and `harmonize_mobifall.py` each write `data/processed/<dataset>.parquet` with the columns
`t, ax, ay, az, gx, gy, gz, amag, label, label3, dataset, subject, placement, trial`.

- **SisFall:** uses the ADXL345 accelerometer (±16 g, `g = ADC·32/2¹³`) and the ITG3200 gyro (`deg/s = ADC·4000/2¹⁶`), ignoring the MMA8451Q.
  Data is resampled from 200 to 50 Hz. Labels come from SisFall-enhanced (a 50 Hz sample takes the max label in ±10 ms, so short FALL spans survive).
  5 misfiled files (`SA15/D17_SE15_*`) have no labels and are skipped.
- **MobiFall:** acc ÷ 9.81 → g, gyro × 57.2958 → deg/s. Acc and gyro are recorded on separate clocks, so both are
  interpolated onto one 50 Hz grid over their shared time range. Two things are trimmed: the ~5 Hz stretch the phone
  records at the start of each trial, and the first 0.1 s (gyro spikes). Gaps longer than 0.1 s split the trial instead of being interpolated.
  MobiFall only labels whole trials, so a fall is labelled **±0.2 s around the impact peak**, which matches the median SisFall-enhanced FALL span. MobiFall has no pre-fall labels.
- `label` is binary (1 = fall). `label3` keeps 0 / 1 (pre-fall) / 2 (fall) for early-warning work.

### 3. Window and pool (Stage 2)
`pool.py` (uses `window.py`) cuts windows of 2.5 s = 125 samples with a 62-sample stride (50% overlap) per trial and writes
`data/processed/pooled_windows.parquet`. Window labelling rules:
- A window is a fall if it contains ≥ 10 FALL samples.
- Windows that catch only 1–9 FALL samples are dropped as ambiguous (538 dropped).
- A window is pre-fall if it contains ≥ 10 ALERT samples.

**Acceptance checks:** `pool.py` exits non-zero if any check fails. All pass:

| check | SisFall | MobiFall |
|---|---|---|
| uniform 50 Hz, every trial | ✅ | ✅ |
| gravity: still windows' median amag | 1.000 g | 1.009 g |
| fall peak amag (median) | 5.40 g | 2.56 g |
| placement | waist | pocket |
| windows (fall / pre-fall / non-fall) | 57,104 (3,721 / 2,660 / 50,723) | 8,133 (465 / 0 / 7,668) |

See `data/processed/amag_hist.png` and `pool_summary.txt`.
**Key finding:** MobiFall's phone accelerometer **saturates at ±2 g per axis**, so its falls never exceed 3.3 g, while SisFall's reach 16 g+.

### 4. Baseline model (Stage 3)
- `features.py`: 89 features per window, [±2 g clip] optional.
  - Per channel (ax..gz, amag, gmag): mean, std, min, max, RMS, energy, peak, jerk, kurtosis, skew.
  - Also: SMA (signal magnitude area) for acc and gyro, 6 correlations between axis pairs, and the dominant FFT frequency of amag.
- `train.py`: RandomForest (200 trees, balanced class weights). **Every evaluation holds out whole subjects:**
  - (a) within-pool: 5-fold GroupKFold, with subjects as the groups.
  - (b) cross-dataset: train on one dataset, test on all of the other.
- `--clip2g` clips SisFall acc to ±2 g, making it look like MobiFall's sensor.

Results (full confusion matrices in `data/processed/train_*.txt`, `results_*.csv`):

| evaluation | features | sensitivity | specificity | false-alarm rate |
|---|---|---|---|---|
| within-pool, leave-subjects-out | all, raw | 95.1% | 99.8% | 0.23% |
| within-pool, leave-subjects-out | all, clip ±2 g | 94.7% | 99.7% | 0.26% |
| SisFall → all MobiFall | all, raw | 92.0% | 98.7% | 1.32% |
| MobiFall → all SisFall | all, raw | 64.9% | 99.7% | 0.26% |
| **SisFall → all MobiFall** | **all, clip ±2 g** | **95.7%** | **96.5%** | **3.47%** |
| **MobiFall → all SisFall** | **all, clip ±2 g** | **84.2%** | **99.7%** | **0.29%** |
| SisFall → all MobiFall | orientation-invariant (23), clip | 74.4% | 99.5% | 0.52% |
| MobiFall → all SisFall | orientation-invariant (23), clip | 89.3% | 98.8% | 1.24% |

### 5. Deployable model (handoff)
- `train_final.py` refits the same RandomForest on **100% of the windows** (65,237 windows, 62 subjects) with the **clip ±2 g + all 89 features**
  setting. That was the most balanced cross-dataset result.
- `predict.py` exposes `predict_fall(window)`. It reuses the exact feature code from `features.py`, and the smoke test checks that
  the features it computes match the training features exactly.

---

## Reproduce everything

```bash
cd fall-detection/scripts
python3 00_inspect.py                 # optional: raw-data report
python3 harmonize_sisfall.py          # -> data/processed/sisfall.parquet
python3 harmonize_mobifall.py         # -> data/processed/mobifall.parquet
python3 pool.py                       # -> pooled_windows.parquet + acceptance checks + amag_hist.png
python3 features.py && python3 features.py --clip2g
python3 train.py && python3 train.py --clip2g      # evaluations (~1 min each)
python3 train_final.py                # -> models/fall/fall_rf.pkl
python3 predict.py                    # smoke test
```

```
fall-detection/
├── scripts/            common.py (constants/helpers), 00_inspect, harmonize_*, window, pool, features, train, train_final, predict
├── models/fall/        fall_rf.pkl, config.json
├── data/processed/     small reports/plots only in git (parquets are rebuilt by the scripts)
└── data/raw/           not in git; download (see above)
```

---

## To do

**Model (this branch: `m1/fall-cnn`)**
- [ ] **1D-CNN on raw windows**, input (125, 7) = the 6 axes + amag, using the same `pooled_windows.parquet`, the same subject-independent
      splits and the same metrics, so it can be compared directly with the RF baseline above. Try training with the ±2 g clip.
- [ ] **Early warning (pre-fall):** a 3-class model using `label3` (SisFall only has pre-fall labels; 2,660 windows).
- [ ] Tune the decision threshold for the rollator's acceptable false-alarm rate (currently 0.5). Report false alarms **per hour** of daily
      activity, not just per window.
- [ ] If the model has to run on the rollator's microcontroller, export it (e.g. TFLite for the CNN, m2cgen/ONNX for the RF) and check its size and latency.

**Data / validity**
- [ ] **Collect data from the actual rollator IMU.** Every training source is body-worn (waist or pocket), not mounted on a walker.
      66 of the 89 features depend on how the sensor is oriented, so check mounting orientation and axis conventions first. If they differ,
      switch to the orientation-invariant feature set (`FEATURE_SET = "invariant"` in `train_final.py`) or remap the axes.
- [ ] Confirm the rollator IMU's range and sample rate. Input must be 50 Hz, g and deg/s. The model clips acc to ±2 g, so any range ≥ 2 g works.
- [ ] Dataset limits to keep in mind:
  - Only 9 MobiFall subjects have daily-activity recordings.
  - MobiFall fall timing is estimated from the impact peak.
  - Almost all falls were performed by young adults. The only elderly subject who performed falls is SE06, a judo expert.

**Integration (Member 3: alert logic, `m3/alert-logic`)**
- [ ] Streaming wrapper: a 50 Hz ring buffer that calls `predict_fall` every 62 samples (~1.24 s) on the latest 125.
- [ ] Turn per-window predictions into events, e.g. fall alert when ≥ 2 consecutive windows exceed the threshold, then a cooldown.
- [ ] Combine with Member 2's gait features (`m2/gait-features`) for fall-risk and pre-fall alerts.

---

### Citations
- Sucerquia, López, Vargas-Bonilla. *SisFall: A Fall and Movement Dataset.* Sensors 17(1):198, 2017.
- Musci et al. *Online Fall Detection using Recurrent Neural Networks.* arXiv:1804.04976, 2018.
- Vavoulas et al. *The MobiFall Dataset.* BMI Lab, TEI of Crete.
