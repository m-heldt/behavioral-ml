import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[2] / "scripts/download_dataset.py"
    spec = importlib.util.spec_from_file_location("download_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_raw_data_is_not_downloaded_or_overwritten(tmp_path, monkeypatch, downloader):
    target = tmp_path / "raw_kmt_dataset"
    target.mkdir()
    for user in range(1, 89):
        (target / f"raw_kmt_user_{user:04d}.json").write_text("original")
    def unexpected_download(**kwargs):
        pytest.fail("An existing dataset must not trigger a download")
    monkeypatch.setattr(downloader.pooch, "retrieve", unexpected_download)
    assert downloader.download_dataset(tmp_path) == target
    assert (target / "raw_kmt_user_0001.json").read_text() == "original"


def test_incomplete_existing_dataset_is_preserved(tmp_path, downloader):
    target = tmp_path / "raw_kmt_dataset"
    target.mkdir()
    original = target / "raw_kmt_user_0001.json"
    original.write_text("original")
    with pytest.raises(FileExistsError, match="incomplete"):
        downloader.download_dataset(tmp_path)
    assert original.read_text() == "original"


def test_new_download_keeps_only_expected_data_and_verifies_hash(tmp_path, monkeypatch, downloader):
    def retrieve(**kwargs):
        assert kwargs["known_hash"] == f"sha256:{downloader.ARCHIVE_SHA256}"
        raw = (Path(kwargs["path"]) / "extracted" / "Behaviour Biometrics Dataset"
               / "behaviour_biometrics_dataset" / "raw_kmt_dataset")
        raw.mkdir(parents=True)
        for user in range(1, 89):
            (raw / f"raw_kmt_user_{user:04d}.json").write_text("{}")
    monkeypatch.setattr(downloader.pooch, "retrieve", retrieve)
    result = downloader.download_dataset(tmp_path)
    assert len(list(result.glob("*.json"))) == 88
    assert list(tmp_path.iterdir()) == [result]
