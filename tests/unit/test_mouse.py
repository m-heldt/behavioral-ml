import numpy as np
import pandas as pd
import pytest

from behavioral_ml.features import mouse


def event_frame(events):
    index = pd.MultiIndex.from_tuples(
        [(5, True, trial, i, movement) for i, (trial, movement, _, _, _) in enumerate(events)],
        names=["User ID", "Is Genuine", "Test ID", "Event ID", "Movement ID"])
    return pd.DataFrame([(event, epoch, coordinates) for _, _, event, epoch, coordinates in events],
                        index=index, columns=["Event", "Epoch", "Coordinates"])


def test_tied_times_preserve_path_without_infinite_derivatives():
    events = event_frame([(1, 1, "movement", 0., [0, 0]),
                          (1, 1, "movement", 0., [3, 0]),
                          (1, 1, "movement", 1., [3, 4])])
    events.attrs["attempt_metadata"] = [{"false_enters": 3}]
    dynamics = mouse.add_mouse_kinematics(events)
    features = mouse.derive_per_movement_features(dynamics)
    assert len(dynamics) == 3
    assert features.iloc[0]["Path"] == 7
    assert features.iloc[0]["Straightness"] == pytest.approx(5 / 7)
    assert not np.isinf(dynamics[["V", "A", "Jerk"]]).any().any()
    assert events.attrs["attempt_metadata"] == [{"false_enters": 3}]
    assert "attempt_metadata" not in dynamics.attrs


def test_movement_differences_do_not_cross_attempts():
    events = event_frame([(1, 1, "movement", 0., [0, 0]), (1, 1, "movement", 1., [10, 0]),
                          (2, 1, "movement", 5., [100, 0]), (2, 1, "movement", 6., [110, 0])])
    result = mouse.add_mouse_kinematics(events)
    assert result["dS"].isna().sum() == 2
    assert (mouse.derive_per_movement_features(result)["Straightness"] == 1).all()


def test_scroll_bursts_and_missing_click_dwell_are_distinct():
    events = event_frame([(1, np.nan, "scrollup press", 0., [0, 0]),
                          (1, np.nan, "scrollup press", .2, [0, 0]),
                          (1, np.nan, "scrollup press", 1., [0, 0]),
                          (2, np.nan, "left press", 2., [0, 0]),
                          (2, np.nan, "left release", 2.1, [0, 0])])
    result = mouse.aggregate_button_features(mouse.add_button_features(events))
    assert result.loc[(5, True, 1), "Burst_count"] == 2
    assert pd.isna(result.loc[(5, True, 1), "Click_Dwell_Time_mean"])
    assert result.loc[(5, True, 2), "Click_Dwell_Time_mean"] == pytest.approx(.1)
