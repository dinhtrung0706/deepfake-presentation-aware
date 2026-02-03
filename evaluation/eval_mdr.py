import numpy as np
from sklearn.metrics import roc_curve


def compute_mdr_at_far(y, scores, target_far=0.01):
    """
    Compute Missed Detection Rate (MDR) at a given False Acceptance Rate (FAR).
    
    MDR = FN / (FN + TP) = FNR (False Negative Rate)
    FAR = FP / (FP + TN) = FPR (False Positive Rate)
    
    Args:
        y: Ground truth labels
        scores: Prediction scores
        target_far: Target FAR (default 1%)
    
    Returns:
        MDR at the specified FAR
    """
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find threshold where FAR (FPR) ≈ target_far
    idx = np.nanargmin(np.abs(fpr - target_far))
    
    # Return MDR (FNR) at that threshold
    return fnr[idx]


y = np.load("features_out/y_test.npy")

# Compute MDRs at FAR=1% Pipeline LR
print("MDR Results at FAR=1% (Logistic Regression):")
mdr_b_base = compute_mdr_at_far(y, np.load("evaluation/scores_base_lr.npy"))
mdr_b_amr = compute_mdr_at_far(y, np.load("evaluation/scores_amr_lr.npy"))
mdr_baa_base = compute_mdr_at_far(y, np.load("evaluation/scores_baa_base_lr.npy"))
mdr_baa_amr = compute_mdr_at_far(y, np.load("evaluation/scores_baa_amr_lr.npy"))

print(f"MDR Base → Base: {mdr_b_base * 100:.2f}%")
print(f"MDR Base → AMR : {mdr_b_amr * 100:.2f}%")
print(f"MDR Base+AMR → Base: {mdr_baa_base * 100:.2f}%")
print(f"MDR Base+AMR → AMR : {mdr_baa_amr * 100:.2f}%")

# Compute MDRs at FAR=1% Pipeline MLP
print("\nMDR Results at FAR=1% (MLP):")
mdr_b_base_mlp = compute_mdr_at_far(y, np.load("evaluation/scores_base_mlp.npy"))
mdr_b_amr_mlp = compute_mdr_at_far(y, np.load("evaluation/scores_amr_mlp.npy"))
mdr_baa_base_mlp = compute_mdr_at_far(y, np.load("evaluation/scores_baa_base_mlp.npy"))
mdr_baa_amr_mlp = compute_mdr_at_far(y, np.load("evaluation/scores_baa_amr_mlp.npy"))

print(f"MDR Base → Base (MLP): {mdr_b_base_mlp * 100:.2f}%")
print(f"MDR Base → AMR (MLP): {mdr_b_amr_mlp * 100:.2f}%")
print(f"MDR Base+AMR → Base (MLP): {mdr_baa_base_mlp * 100:.2f}%")
print(f"MDR Base+AMR → AMR (MLP): {mdr_baa_amr_mlp * 100:.2f}%")
