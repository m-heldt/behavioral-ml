from pathlib import Path
import pooch
import shutil

RAW_DATASET_NAME = "raw_kmt_dataset"

data_dir = Path("data")
extracted_dir = data_dir / "Behaviour Biometrics Dataset" / "behaviour_biometrics_dataset" / RAW_DATASET_NAME
target_dir = data_dir / RAW_DATASET_NAME

pooch.retrieve(
    url="https://data.mendeley.com/public-api/zip/fnf8b85kr6/download/1",
    known_hash="sha256:4fd279eab07ba65be0d3b4f79943d176abe1a36b16927f568b199d7ae5638739",
    fname=f"{RAW_DATASET_NAME}.zip",
    path=data_dir,
    processor=pooch.Unzip(extract_dir=data_dir.absolute()),
    progressbar=True
)
(data_dir / f"{RAW_DATASET_NAME}.zip").unlink()


shutil.move(extracted_dir, target_dir)
shutil.rmtree(data_dir / "Behaviour Biometrics Dataset")