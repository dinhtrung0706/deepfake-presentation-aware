import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import joblib

# Load data
X_train = np.load("features_out/X_train_base.npy")
y_train = np.load("features_out/y_train.npy")

print("Train shape:", X_train.shape)
print("Label shape:", y_train.shape)

# Pipeline: scaling + LR

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

# Save model
joblib.dump(clf, "training/lr_base_model.joblib")
# Pipeline: scaling + MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    random_state=42,
)
clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ]
)

# Train
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "training/mlp_base_model.joblib")

print("Training (Base) finished.")
