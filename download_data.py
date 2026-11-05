"""Download the four public datasets into ./data (they are not redistributed in this repository).

    python download_data.py

D1  Students Performance in Exams (Kaggle)  https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics
D2/D3  UCI Student Performance (#320)       https://archive.ics.uci.edu/dataset/320/student+performance
D4  Student Performance Factors (Kaggle)    https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
"""

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import kagglehub

DATA = Path(__file__).resolve().parent / "data"
KAGGLE = {
    "student_performance_math": ("rkiattisak/student-performance-in-mathematics", "exams.csv"),
    "student_performance_factors": ("lainguyn123/student-performance-factors", "StudentPerformanceFactors.csv"),
}
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def kaggle():
    for folder, (handle, fname) in KAGGLE.items():
        out = DATA / folder / fname
        if out.exists():
            continue
        src = next(Path(kagglehub.dataset_download(handle)).rglob(fname))
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out)
        print("saved", out)


def uci():
    out = DATA / "uci_student_performance"
    if (out / "student-mat.csv").exists() and (out / "student-por.csv").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(UCI_URL, timeout=60).read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("student.zip")))  # the CSVs sit in a nested zip
    for fname in ("student-mat.csv", "student-por.csv"):
        (out / fname).write_bytes(inner.read(fname))
        print("saved", out / fname)


if __name__ == "__main__":
    kaggle()
    uci()
