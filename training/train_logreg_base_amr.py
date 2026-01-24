import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load data
X_base = np.load("features_out/X_train_base.npy")
X_amr = np.load("features_out/X_train_amr.npy")
y = np.load("features_out/y_train.npy")

# Combine
X_train = np.vstack([X_base, X_amr])
y_train = np.concatenate([y, y])

print("Train shape:", X_train.shape)
print("Label shape:", y_train.shape)

# Pipeline
clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                solver="liblinear", max_iter=2000, class_weight="balanced"
            ),
        ),
    ]
)

# Train
clf.fit(X_train, y_train)

# Save
joblib.dump(clf, "training/lr_base_amr_model.joblib")

print("Training (Base + AMR) finished.")
