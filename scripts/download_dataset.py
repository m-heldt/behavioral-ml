"""Download KMT once, verifying its archive without overwriting raw data."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pooch

RAW_DATASET_NAME = "raw_kmt_dataset"
ARCHIVE_SHA256 = "4fd279eab07ba65be0d3b4f79943d176abe1a36b16927f568b199d7ae5638739"


def download_dataset(data_dir="data"):
    data_dir = Path(data_dir)
    target_dir = data_dir / RAW_DATASET_NAME
    if target_dir.exists():
        if len(list(target_dir.glob("raw_kmt_user_*.json"))) == 88:
            return target_dir
        raise FileExistsError(f"{target_dir} already exists but is incomplete; inspect it before refreshing")
    data_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="kmt-download-", dir=data_dir) as temporary:
        extracted = Path(temporary) / "extracted"
        pooch.retrieve(
            url="https://data.mendeley.com/public-api/zip/fnf8b85kr6/download/1",
            known_hash=f"sha256:{ARCHIVE_SHA256}",
            fname=f"{RAW_DATASET_NAME}.zip",
            path=temporary,
            processor=pooch.Unzip(extract_dir=extracted),
            progressbar=True,
        )
        raw = extracted / "Behaviour Biometrics Dataset" / "behaviour_biometrics_dataset" / RAW_DATASET_NAME
        if len(list(raw.glob("raw_kmt_user_*.json"))) != 88:
            raise ValueError("The downloaded archive does not contain the expected 88 KMT JSON files")
        shutil.copytree(raw, target_dir)
    return target_dir


if __name__ == "__main__":
    print(download_dataset())
