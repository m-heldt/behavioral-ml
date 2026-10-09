import json
from pathlib import Path
import pandas as pd
import numpy as np
import os


def load_data(data_dir: str | Path) -> pd.DataFrame:
    INDEX_NAMES = [
        "User ID",
        "Is Genuine",
        "Test ID",
        "Event ID",
    ]
    
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    file_list = [file.name for file in data_dir.glob("*.json")]
    file_list.sort()
    
    mouse_data, key_data = {}, {}
    
    for file in file_list:
        with open(data_dir / file, 'r') as f:
            user_data = json.load(f)
            for test_no in range(1, 11):
                mouse_true = user_data["true_data"][f"test_{test_no}"]["mouse_events"]
                for event_id, event in enumerate(mouse_true):
                    mouse_data[(int(file.split("_")[-1].split(".")[0]), True, test_no, event_id)] = event 

                mouse_false = user_data["false_data"][f"test_{test_no}"]["mouse_events"]
                for event_id, event in enumerate(mouse_false):
                    mouse_data[(int(file.split("_")[-1].split(".")[0]), False, test_no, event_id)] = event

                key_true = user_data["true_data"][f"test_{test_no}"]["key_events"]
                for event_id, event in enumerate(key_true):
                    key_data[(int(file.split("_")[-1].split(".")[0]), True, test_no, event_id)] = event 
                
                key_false = user_data["false_data"][f"test_{test_no}"]["key_events"]
                for event_id, event in enumerate(key_false):
                    key_data[(int(file.split("_")[-1].split(".")[0]), False, test_no, event_id)] = event


    mouse_df = pd.DataFrame.from_dict(mouse_data, orient="index").astype({
        "Epoch": "Float64",
        "Movement ID": "UInt16"
    })
    mouse_df.index.names = INDEX_NAMES
    mouse_df.set_index(["Movement ID"], append=True, inplace=True)

    mouse_df.drop(["false_enters", "Timestamp"], axis=1, inplace=True)
    mouse_df.dropna(subset="Event", inplace=True)

    # drop events with duplicated Epoch within each Event type group
    # this prevents inf velocity values while preserving data of other Event types
    mouse_df.drop_duplicates(subset=["Event", "Epoch"], inplace=True)
  

    key_df = pd.DataFrame.from_dict(key_data, orient="index").astype({
        "Epoch": "Float64"
    })
    key_df.index.names = INDEX_NAMES
    
    # "Text Changed" and "Input Box" columns may provide useful data
    # consider extracting features out of them in the future
    key_df.drop(["Text Changed", "Timestamp", "Input Box"], axis=1, inplace=True)

    return (mouse_df, key_df)
