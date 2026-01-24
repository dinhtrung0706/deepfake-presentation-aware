import os
import subprocess
from tqdm import tqdm

SRC_ROOT = "data"
DST_SUBDIR = "amr"


def amr_simulate(in_wav, out_wav):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            in_wav,
            "-ac",
            "1",
            "-ar",
            "8000",
            "-acodec",
            "libopencore_amrnb",
            "temp.amr",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        ["ffmpeg", "-y", "-i", "temp.amr", out_wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    os.remove("temp.amr")


def process_split(split):
    for label in ["bonafide", "spoof"]:
        src_dir = f"{SRC_ROOT}/{split}/base/{label}"
        dst_dir = f"{SRC_ROOT}/{split}/amr/{label}"
        os.makedirs(dst_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir) if f.endswith(".wav")]
        for f in tqdm(files, desc=f"{split}-{label}"):
            in_wav = os.path.join(src_dir, f)
            out_wav = os.path.join(dst_dir, f)
            amr_simulate(in_wav, out_wav)


if __name__ == "__main__":
    process_split("train")
    process_split("test")
