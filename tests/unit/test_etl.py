import json

import pandas as pd
import pytest

from behavioral_ml.etl import infer_typist_candidates, load_data


def test_loader_preserves_tied_events_and_optional_fields(tmp_path):
    attempt = {
        "mouse_events": [
            {"Event": "movement", "Epoch": 1., "Movement ID": 1, "Coordinates": [0, 0]},
            {"Event": "movement", "Epoch": 1., "Movement ID": 1, "Coordinates": [2, 0]},
            {"Event": "left press", "Epoch": 1., "Coordinates": [2, 0]},
            {"false_enters": 3}],
        "key_events": [{"Key": "a", "Event": "pressed", "Epoch": 1.,
                        "Input Box": "Name", "Text Changed": True}],
    }
    raw = {"true_data": {"test_1": attempt}, "false_data": {"test_1": attempt}}
    (tmp_path / "raw_kmt_user_0007.json").write_text(json.dumps(raw))
    (tmp_path / "unrelated.json").write_text("not dataset data")
    mouse, keys = load_data(tmp_path)
    assert len(mouse) == 6
    assert keys["Input Box"].eq("Name").all()
    assert keys["Text Changed"].all()
    assert mouse.attrs["attempt_metadata"][0]["false_enters"] == 3
    assert set(mouse.index.get_level_values("User ID")) == {7}


def test_loader_empty_directory_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="No raw_kmt"):
        load_data(tmp_path)


def test_attribution_keeps_all_candidates_instead_of_picking_nearest():
    index = pd.MultiIndex.from_tuples(
        [(10, True, 1, 0), (20, True, 1, 0), (30, True, 1, 0), (30, False, 1, 0)],
        names=["User ID", "Is Genuine", "Test ID", "Event ID"])
    events = pd.DataFrame({"Epoch": [100., 200., 5000., 300.]}, index=index)
    candidates = infer_typist_candidates(events)
    assert candidates.loc[(30, False, 1)] == (10, 20)
    assert candidates.loc[(30, True, 1)] == (30,)
