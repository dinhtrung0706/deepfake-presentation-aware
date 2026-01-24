import librosa
import numpy as np
import os
from tqdm import tqdm

SR = 16000
N_MELS = 64
MAX_LEN = 4  # seconds

def extract_logmel_delta(y, sr, n_mels=64):
    max_len = sr * MAX_LEN
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=400, hop_length=160
    )
    logmel = librosa.power_to_db(mel)

    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    return np.vstack([logmel, delta, delta2])


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

def build_dataset_w_dt(data_dir, label, sr=16000):
    X, y = [], []

    for f in tqdm(sorted(os.listdir(data_dir))):
        if not f.endswith(".wav"):
            continue

        path = os.path.join(data_dir, f)
        wav, _ = librosa.load(path, sr=sr)

        feat = extract_logmel_delta(wav, sr)  # (192, T)
        feat_mean = np.mean(feat, axis=1)  # (192,)

        X.append(feat_mean)
        y.append(label)

    return np.array(X), np.array(y)
