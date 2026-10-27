import pytest
import pandas as pd
import behavioral_ml.features.key as key

def test_transform_key_events():
    sample = pd.read_parquet("tests/fixtures/transform_in.parquet")
    expected = pd.read_parquet("tests/fixtures/transform_out.parquet")

    result = key._transform_key_events(sample)

    result["Release Only"] = result["Release Only"].astype("boolean")
    expected["Release Only"] = expected["Release Only"].astype("boolean")

    pd.testing.assert_frame_equal(result, expected)


def event_frame(events):
    index = pd.MultiIndex.from_tuples(
        [(7, True, 1, i) for i in range(len(events))],
        names=["User ID", "Is Genuine", "Test ID", "Event ID"])
    return pd.DataFrame(events, index=index, columns=["Key", "Event", "Epoch"])


def test_release_only_key_does_not_contaminate_next_press_timings():
    events = event_frame([("a", "pressed", 0.), ("a", "released", .1),
                          ("tab", "released", .2), ("b", "pressed", .4),
                          ("b", "released", .5)])
    result = key.add_key_dynamics(events)
    a = result.loc[result["Key"].eq("a")].iloc[0]
    assert a["Dwell Time"] == pytest.approx(.1)
    assert a["Flight Time"] == pytest.approx(.3)
    tab = result.loc[result["Key"].eq("tab")].iloc[0]
    assert tab[["Down", "Dwell Time", "Flight Time", "DD", "UU"]].isna().all()


def test_overlap_is_valid_and_repeats_count_once():
    events = event_frame([("shift", "pressed", 0.), ("shift", "pressed", .3),
                          ("a", "pressed", .4), ("shift", "released", .5),
                          ("a", "released", .6)])
    result = key.add_key_dynamics(events)
    shift = result.loc[result["Key"].eq("shift")].iloc[0]
    assert shift["Dwell Time"] == .5
    assert shift["Flight Time"] == pytest.approx(-.1)
    assert key.aggregate_selected_key_counts(result, {"shifts": ["shift"]}).iloc[0, 0] == 1


def test_dynamics_columns_are_stable_and_missing_releases_are_supported():
    events = event_frame([("shift", "pressed", 0.), ("a", "pressed", .1), ("a", "released", .2)])
    dynamics = key.add_key_dynamics(events)
    result = key.aggregate_selected_key_dynamics(dynamics, {"shifts": ["shift"]}, ["Dwell Time"])
    assert list(result) == ["shifts_Dwell Time_mean", "shifts_Dwell Time_iqr"]
    assert result.isna().all().all()
