# Presentation-Aware Spoofing Detection (Lightweight Baseline)

This project is a **lightweight reproduction and analysis** inspired by recent work on *presentation-aware deepfake / spoofing detection*, with a focus on **codec-induced presentation mismatch (AMR)** under **CPU-only constraints**.

The goal is **not** to build a state-of-the-art model, but to **demonstrate, in a controlled and reproducible manner**, how presentation changes (e.g., AMR encoding) affect spoofing detection performance, and how much such degradation can be mitigated by presentation-aware training.

---

## 1. Problem Statement

Spoofing / deepfake detectors often assume that training and testing data share the same *presentation conditions*. In practice, speech may undergo additional transformations (e.g., telephony codecs such as **AMR-NB**), causing **presentation mismatch**.

This project investigates:

* How much AMR encoding degrades spoofing detection performance
* Whether adding AMR data during training improves robustness
* The limits of **traditional handcrafted features + linear models** under such mismatch

---

## 2. Dataset Structure

```
deepfake-presentation-aware/
├── data/
│   ├── train/
│   │   ├── base/
│   │   │   ├── bonafide/
│   │   │   └── spoof/
│   │   └── amr/
│   │       ├── bonafide/
│   │       └── spoof/
│   └── test/
│       ├── base/
│       │   ├── bonafide/
│       │   └── spoof/
│       └── amr/
│           ├── bonafide/
│           └── spoof/
```

* **Base**: original waveform presentation
* **AMR**: AMR-NB encoded speech (codec-induced presentation)
* Train and test sets are **disjoint at the audio-file level** (no leakage)

---

## 3. Feature Extraction

### 3.1 Log-Mel Spectrogram + Dynamic Features

Each utterance is represented using:

* 64-dimensional log-mel spectrogram
* First-order delta (Δ)
* Second-order delta-delta (ΔΔ)

Procedure:

1. Load waveform (16 kHz)
2. Truncate / zero-pad to 4 seconds
3. Compute log-mel spectrogram
4. Compute Δ and ΔΔ
5. Concatenate: `[static | Δ | ΔΔ]` → 192 dimensions
6. Mean pooling over time → utterance-level vector

This design follows **traditional ASVspoof baselines** and is fully CPU-compatible.

---

## 4. Model

* **Classifier**: Logistic Regression
* **Pipeline**:

  * StandardScaler
  * LogisticRegression (liblinear, class-balanced)

Rationale:

* Linear model avoids overfitting
* Ensures observed effects come from **features and presentation**, not model complexity
* Common baseline in classical spoofing detection literature

---

## 5. Experimental Setups

### 5.1 Training Conditions

* **Base**: trained only on Base presentation
* **Base + AMR**: trained on combined Base and AMR data (presentation-aware training)

### 5.2 Evaluation Conditions

* Test on **Base**
* Test on **AMR**

Performance is measured using **Equal Error Rate (EER)**.

---

## 6. Results

| Training Data | Test Base | Test AMR |
| ------------- | --------- | -------- |
| Base          | 23.00%    | 28.60%   |
| Base + AMR    | 23.20%    | 27.10%   |

### Observations

* AMR encoding causes a **large performance degradation** (~+5.6% EER)
* Adding AMR data during training **partially mitigates** this degradation
* A significant performance gap remains, indicating limited presentation robustness

---

## 7. Discussion

The results show that:

* Dynamic log-mel features are **sensitive to codec-induced presentation mismatch**
* Presentation-aware training improves robustness, but only marginally
* Handcrafted spectral features lack sufficient capacity to fully encode codec artifacts

This aligns with findings in recent literature: **presentation robustness is primarily a representation problem, not a classifier problem**.

---

## 8. Limitations

* No deep or codec-aware representations
* Mean pooling discards fine-grained temporal structure
* AMR-NB only (no other codecs or channels)

These limitations are intentional to keep the system interpretable and CPU-friendly.

---

## 9. Conclusion

This project demonstrates, using a simple and transparent pipeline, that:

* Presentation mismatch (AMR encoding) significantly degrades spoofing detection
* Presentation-aware training provides limited but consistent gains
* Traditional log-mel based features are insufficient for full robustness

The study highlights the need for **presentation-aware representations**, even when using simple classifiers.

---

## 10. Reproducibility Notes

* Fixed audio length (4 seconds)
* Deterministic file ordering
* Train/test split by file, not by frame
* No GPU required

---

## 11. References

* ASVspoof Challenge baselines
* On Deepfake Voice Detection – It’s All in the Presentation (arXiv:2509.26471)

---

*This repository is intended for educational and analytical purposes.*
