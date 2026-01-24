import os
import subprocess
from tqdm import tqdm

BASE_DIR = "data/base"
OUT_DIR = "data/presented_amr"

os.makedirs(f"{OUT_DIR}/bonafide", exist_ok=True)
os.makedirs(f"{OUT_DIR}/spoof", exist_ok=True)


def amr_simulate(in_wav, out_wav):
    tmp_amr = "tmp.amr"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_wav,
            "-ac",
            "1",
            "-ar",
            "8000",
            "-acodec",
            "libopencore_amrnb",
            tmp_amr,
        ]
    )

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_amr, out_wav])

    os.remove(tmp_amr)


def process_split(label):
    in_dir = os.path.join(BASE_DIR, label)
    out_dir = os.path.join(OUT_DIR, label)

    files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]

    for f in tqdm(files, desc=f"AMR {label}"):
        in_wav = os.path.join(in_dir, f)
        out_wav = os.path.join(out_dir, f)
        amr_simulate(in_wav, out_wav)


if __name__ == "__main__":
    process_split("bonafide")
    process_split("spoof")
