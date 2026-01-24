import numpy as np
import joblib

# Load model
clf_b_lr = joblib.load("training/lr_base_model.joblib")
clf_baa_lr = joblib.load("training/lr_base_amr_model.joblib")
clf_b_mlp = joblib.load("training/mlp_base_model.joblib")
clf_baa_mlp = joblib.load("training/mlp_base_amr_model.joblib")

# Load test data
X_test_base = np.load("features_out/X_test_base.npy")
X_test_amr = np.load("features_out/X_test_amr.npy")
y_test = np.load("features_out/y_test.npy")

# Scores (log-likelihood ratio style)
scores_b_base_lr = clf_b_lr.decision_function(X_test_base)
scores_b_amr_lr = clf_b_lr.decision_function(X_test_amr)
scores_baa_base_lr = clf_baa_lr.decision_function(X_test_base)
scores_baa_amr_lr = clf_baa_lr.decision_function(X_test_amr)
scores_b_base_mlp = clf_b_mlp.predict_proba(X_test_base)[:, 1]
scores_b_amr_mlp = clf_b_mlp.predict_proba(X_test_amr)[:, 1]
scores_baa_base_mlp = clf_baa_mlp.predict_proba(X_test_base)[:, 1]
scores_baa_amr_mlp = clf_baa_mlp.predict_proba(X_test_amr)[:, 1]


np.save("evaluation/scores_base_lr.npy", scores_b_base_lr)
np.save("evaluation/scores_amr_lr.npy", scores_b_amr_lr)
np.save("evaluation/scores_baa_base_lr.npy", scores_baa_base_lr)
np.save("evaluation/scores_baa_amr_lr.npy", scores_baa_amr_lr)
np.save("evaluation/scores_base_mlp.npy", scores_b_base_mlp)
np.save("evaluation/scores_amr_mlp.npy", scores_b_amr_mlp)
np.save("evaluation/scores_baa_base_mlp.npy", scores_baa_base_mlp)
np.save("evaluation/scores_baa_amr_mlp.npy", scores_baa_amr_mlp)

print("Evaluation done.")
