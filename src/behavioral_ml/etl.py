"""Read KMT event logs without deleting events that happen at the same time."""

import json
from pathlib import Path

import pandas as pd

TEST_LEVEL = ["User ID", "Is Genuine", "Test ID"]
EVENT_LEVEL = TEST_LEVEL + ["Event ID"]


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mouse and key events in recording order.

    Retain optional keyboard fields for future analyses. Undocumented
    false_enters trailer values are stored in mouse.attrs["attempt_metadata"],
    not treated as observed mouse events or used as model features.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    files = sorted(data_dir.glob("raw_kmt_user_*.json"))
    if not files:
        raise ValueError(f"No raw_kmt_user_*.json files in {data_dir}")

    mouse_data, key_data, metadata = {}, {}, []
    for path in files:
        user_id = int(path.stem.split("_")[-1])
        with path.open() as stream:
            raw = json.load(stream)
        for name, genuine in [("true_data", True), ("false_data", False)]:
            for test, attempt in sorted(raw[name].items(), key=lambda item: int(item[0].split("_")[-1])):
                test_id = int(test.split("_")[-1])
                identity = (user_id, genuine, test_id)
                for event_id, event in enumerate(attempt["mouse_events"]):
                    if event.get("Event") is not None:
                        mouse_data[(*identity, event_id)] = event
                    elif "false_enters" in event:
                        metadata.append(dict(zip(TEST_LEVEL, identity), false_enters=event["false_enters"]))
                for event_id, event in enumerate(attempt["key_events"]):
                    key_data[(*identity, event_id)] = event

    def frame(records):
        result = pd.DataFrame.from_dict(records, orient="index")
        if records:
            result.index = pd.MultiIndex.from_tuples(result.index, names=EVENT_LEVEL)
        else:
            result.index = pd.MultiIndex.from_tuples([], names=EVENT_LEVEL)
        result["Epoch"] = pd.to_numeric(result.get("Epoch", pd.Series(index=result.index, dtype=float))).astype("Float64")
        return result.drop(columns=["Timestamp"], errors="ignore")

    mouse = frame(mouse_data)
    if "Movement ID" not in mouse:
        mouse["Movement ID"] = pd.NA
    mouse["Movement ID"] = mouse["Movement ID"].astype("UInt16")
    mouse = mouse.set_index("Movement ID", append=True)
    mouse.attrs["attempt_metadata"] = metadata
    return mouse, frame(key_data)


def infer_typist_candidates(key_events, max_gap_seconds=3600):
    """Conservative candidates inferred from preceding genuine-session windows.

    Genuine trials identify their typist. For each impostor trial, retain EVERY
    other user's genuine session ending within the preceding hour, not just the
    nearest session. These are plausible identities, not ground-truth labels.
    Empty candidate tuples mean no attribution. The time-window assumption must
    be reported with any identity-disjoint experiment using this inference.
    """
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    windows = key_events.groupby(level=TEST_LEVEL)["Epoch"].agg(["min", "max"])
    genuine = windows.loc[windows.index.get_level_values("Is Genuine")]
    ends = genuine.groupby(level="User ID")["max"].max()
    result = {}
    for (owner, is_genuine, test), row in windows.iterrows():
        if is_genuine:
            candidates = (owner,)
        else:
            mask = ends.le(row["min"]) & ends.ge(row["min"] - max_gap_seconds) & (ends.index != owner)
            candidates = tuple(ends.index[mask])
        result[(owner, is_genuine, test)] = candidates
    index = pd.MultiIndex.from_tuples(result, names=TEST_LEVEL)
    return pd.Series(list(result.values()), index=index, name="typist_candidates")
