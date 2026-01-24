import numpy as np
import joblib

# Load model
clf_b = joblib.load("training/lr_base_model.joblib")
clf_baa = joblib.load("training/lr_base_amr_model.joblib")

# Load test data
X_test_base = np.load("features_out/X_test_base.npy")
X_test_amr = np.load("features_out/X_test_amr.npy")
y_test = np.load("features_out/y_test.npy")

# Scores (log-likelihood ratio style)
scores_b_base = clf_b.decision_function(X_test_base)
scores_b_amr = clf_b.decision_function(X_test_amr)
scores_baa_base = clf_baa.decision_function(X_test_base)
scores_baa_amr = clf_baa.decision_function(X_test_amr)


np.save("evaluation/scores_base.npy", scores_b_base)
np.save("evaluation/scores_amr.npy", scores_b_amr)
np.save("evaluation/scores_baa_base.npy", scores_baa_base)
np.save("evaluation/scores_baa_amr.npy", scores_baa_amr)

print("Evaluation done.")
