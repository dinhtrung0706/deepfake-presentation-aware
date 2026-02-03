# Presentation-Aware Spoofing Detection (Lightweight Baseline)

This project is a **lightweight reproduction and analysis** inspired by recent work on *presentation-aware deepfake / spoofing detection*, with a focus on **codec-induced presentation mismatch (AMR)** under **CPU-only constraints**.

The goal is **not** to build a state-of-the-art model, but to **demonstrate, in a controlled and reproducible manner**, how presentation changes (e.g., AMR encoding) affect spoofing detection performance, and how much such degradation can be mitigated by presentation-aware training.

---

## 1. Problem Statement

Spoofing / deepfake detectors often assume that training and testing data share the same *presentation conditions*. In practice, speech may undergo additional transformations (e.g., telephony codecs such as **AMR-NB**), causing **presentation mismatch**.

In high-security deployments, systems must operate at **low False Acceptance Rate (FAR)**, such as 1%, to minimize the risk of accepting spoofed audio. This makes **Missed Detection Rate (MDR) at FAR=1%** a critical metric alongside the commonly reported EER.

This project investigates:

* How much AMR encoding degrades spoofing detection performance (EER)
* The severity of degradation under high-security thresholds (MDR @ FAR=1%)
* Whether adding AMR data during training improves robustness
* The limits of **traditional handcrafted features + linear/non-linear models** under such mismatch

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

## 4. Models

### 4.1 Logistic Regression

* **Pipeline**:
  * StandardScaler
  * LogisticRegression (liblinear, class-balanced)

Rationale:
* Linear model avoids overfitting
* Ensures observed effects come from **features and presentation**, not model complexity
* Common baseline in classical spoofing detection literature

### 4.2 MLP (Multi-Layer Perceptron)

* **Pipeline**:
  * StandardScaler
  * MLPClassifier (hidden layers, early stopping)

Rationale:
* Provides a non-linear baseline for comparison
* Tests whether increased model capacity improves presentation robustness
* Still CPU-friendly and interpretable

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

Performance is measured using:
- **EER (Equal Error Rate)**: Error rate at balanced threshold (FPR = FNR)
- **MDR @ FAR=1%**: Missed Detection Rate when allowing only 1% false acceptance (high security setting)

### 6.1 Logistic Regression

| Training Data | Test Base (EER) | Test AMR (EER) | Test Base (MDR@1%) | Test AMR (MDR@1%) |
| ------------- | --------------- | -------------- | ------------------ | ----------------- |
| Base          | 23.00%          | 28.60%         | 65.60%             | 79.40%            |
| Base + AMR    | 23.20%          | 27.10%         | 74.40%             | 81.40%            |

### 6.2 MLP (Multi-Layer Perceptron)

| Training Data | Test Base (EER) | Test AMR (EER) | Test Base (MDR@1%) | Test AMR (MDR@1%) |
| ------------- | --------------- | -------------- | ------------------ | ----------------- |
| Base          | 22.00%          | 29.70%         | 91.00%             | 95.40%            |
| Base + AMR    | 24.00%          | 22.90%         | 66.80%             | 61.00%            |

### Observations

**Logistic Regression (EER):**
* AMR encoding causes a **large performance degradation** (~+5.6% EER)
* Adding AMR data during training **partially mitigates** this degradation (28.60% → 27.10%)
* A significant performance gap remains, indicating limited presentation robustness

**MLP (EER):**
* MLP achieves slightly better baseline performance on Base (22.00% vs 23.00%)
* However, AMR mismatch causes **even larger degradation** for MLP (~+7.7% EER)
* Presentation-aware training with MLP shows **stronger improvement** on AMR (29.70% → 22.90%)
* MLP benefits more from presentation-aware training due to its higher model capacity

**MDR at FAR=1%:**
* At strict security settings (FAR=1%), both models miss a significant portion of attacks
* MLP trained on Base only shows **extremely high MDR** (91-95%), indicating poor generalization at low FAR
* **Presentation-aware training dramatically improves MLP's MDR** (95.40% → 61.00% on AMR)
* Logistic Regression shows more stable but still high MDR, with less benefit from presentation-aware training

---

## 7. Discussion

The results show that:

* Dynamic log-mel features are **sensitive to codec-induced presentation mismatch**
* Both Logistic Regression and MLP suffer significant degradation under AMR mismatch
* **MLP with presentation-aware training** achieves the best AMR performance (22.90% EER), demonstrating that increased model capacity can better leverage diverse training data
* However, MLP's Base performance slightly degrades with mixed training (22.00% → 24.00%), suggesting a trade-off between matched and mismatched conditions
* Logistic Regression shows more stable but limited improvements across conditions

**High-Security Scenario (FAR=1%):**

* At strict security thresholds, the impact of presentation mismatch becomes **even more severe**
* MLP trained only on Base misses **91-95% of attacks** when tested under presentation mismatch
* Presentation-aware training provides **dramatic improvement** for MLP (95.40% → 61.00% MDR on AMR)
* Logistic Regression maintains more consistent but still high MDR across conditions
* The gap between EER and MDR@1% highlights that **real-world security deployments face much higher miss rates** than balanced metrics suggest

This aligns with findings in recent literature: **presentation robustness benefits from both diverse training data and sufficient model capacity**, especially under strict security requirements.

---

## 8. Limitations

* No deep or codec-aware representations
* Mean pooling discards fine-grained temporal structure
* AMR-NB only (no other codecs or channels)

These limitations are intentional to keep the system interpretable and CPU-friendly.

---

## 9. Conclusion

This project demonstrates, using simple and transparent pipelines, that:

* Presentation mismatch (AMR encoding) significantly degrades spoofing detection for both linear and non-linear models
* Presentation-aware training provides gains, with **MLP showing stronger improvement** on mismatched conditions (EER: 29.70% → 22.90%, MDR@1%: 95.40% → 61.00%)
* At high-security thresholds (FAR=1%), the impact is **much more severe** — models miss 61-95% of attacks
* Linear models (Logistic Regression) provide more stable but limited improvements across all metrics
* Traditional log-mel based features benefit from increased model capacity when combined with presentation-aware training

The study highlights the importance of both **presentation-aware training data** and **sufficient model capacity** for robust spoofing detection, particularly in **high-security deployments** where FAR must be minimized.

---

## 10. Reproducibility Notes

* Fixed audio length (4 seconds)
* Deterministic file ordering
* Train/test split by file, not by frame
* No GPU required

---

## 11. References

If you use this project or build upon it, please consider citing the following work which inspired the experimental design and analysis:

```bibtex
@misc{delgado2025deepfakevoicedetection,
  title={On Deepfake Voice Detection -- It's All in the Presentation},
  author={H{\'e}ctor Delgado and Giorgio Ramondetti and Emanuele Dalmasso and Gennady Karvitsky and Daniele Colibro and Haydar Talib},
  year={2025},
  eprint={2509.26471},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2509.26471}
}
```

* ASVspoof 2019

---

*This repository is intended for educational and analytical purposes.*
