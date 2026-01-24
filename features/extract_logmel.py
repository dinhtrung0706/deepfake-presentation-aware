import librosa
import numpy as np
import os
from tqdm import tqdm

SR = 16000
N_MELS = 64
MAX_LEN = 4  # seconds

def extract_feature(wav_path):
    y, sr = librosa.load(wav_path, sr=SR)
    max_samples = SR * MAX_LEN
    if len(y) > max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=400, hop_length=160
    )
    logmel = librosa.power_to_db(mel)
    return np.mean(logmel, axis=1)  # mean pooling (64-dim)

def build_dataset(data_dir, label):
    X, y = [], []
    for f in tqdm(os.listdir(data_dir)):
        if not f.endswith(".wav"):
            continue
        feat = extract_feature(os.path.join(data_dir, f))
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)
