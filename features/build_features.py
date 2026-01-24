from extract_logmel import build_dataset_w_dt
import numpy as np
import os

os.makedirs("features_out", exist_ok=True)

# ===== TRAIN =====
X_bf, y_bf = build_dataset_w_dt("data/train/base/bonafide", 0)
X_sp, y_sp = build_dataset_w_dt("data/train/base/spoof", 1)
X_train_base = np.vstack([X_bf, X_sp])
y_train = np.concatenate([y_bf, y_sp])

X_bf, y_bf = build_dataset_w_dt("data/train/amr/bonafide", 0)
X_sp, y_sp = build_dataset_w_dt("data/train/amr/spoof", 1)
X_train_amr = np.vstack([X_bf, X_sp])

# ===== TEST =====
X_bf, y_bf = build_dataset_w_dt("data/test/base/bonafide", 0)
X_sp, y_sp = build_dataset_w_dt("data/test/base/spoof", 1)
X_test_base = np.vstack([X_bf, X_sp])
y_test = np.concatenate([y_bf, y_sp])

X_bf, y_bf = build_dataset_w_dt("data/test/amr/bonafide", 0)
X_sp, y_sp = build_dataset_w_dt("data/test/amr/spoof", 1)
X_test_amr = np.vstack([X_bf, X_sp])

# ===== SAVE =====
np.save("features_out/X_train_base.npy", X_train_base)
np.save("features_out/X_train_amr.npy", X_train_amr)
np.save("features_out/X_test_base.npy", X_test_base)
np.save("features_out/X_test_amr.npy", X_test_amr)
np.save("features_out/y_train.npy", y_train)
np.save("features_out/y_test.npy", y_test)
