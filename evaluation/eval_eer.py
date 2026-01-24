import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y, scores):
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


y = np.load("features_out/y_test.npy")

eer_b_base = compute_eer(y, np.load("evaluation/scores_base.npy"))
eer_b_amr = compute_eer(y, np.load("evaluation/scores_amr.npy"))
eer_baa_base = compute_eer(y, np.load("evaluation/scores_baa_base.npy"))
eer_baa_amr = compute_eer(y, np.load("evaluation/scores_baa_amr.npy"))

print(f"EER Base → Base: {eer_b_base * 100:.2f}%")
print(f"EER Base → AMR : {eer_b_amr * 100:.2f}%")
print(f"EER Base+AMR → Base: {eer_baa_base * 100:.2f}%")
print(f"EER Base+AMR → AMR : {eer_baa_amr * 100:.2f}%")