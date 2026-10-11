import numpy as np
import pandas as pd

TEST_LEVEL = ["User ID", "Is Genuine", "Test ID"]
MOVEMENT_LEVEL = TEST_LEVEL + ["Movement ID"]

NUMPAD_KEYS = [
    'numpad4',
    'numpad2',
    'numpad8',
    'numpad9',
    'numpad1',
    'numpad5',
    'numpad0',
    'numpad6',
    'numpad7',
    'numpad3',
    'numpadsubstract',
    'numpaddivide',
    'numpadenter',
    'numpadadd',
    'numpaddecimal'
    ]
ARROW_KEYS = ["up", "down", "right", "left"]
CORRECTION_KEYS = ["delete", "backspace"]
SHIFT_KEYS = ["shift", "rshift"]
ENTER_KEYS = ["enter"]
TAB_KEYS = ["tab"]



def _transform_key_events(group):
    presses = []
    active_presses = {}
    for key, event, epoch in group[["Key", "Event", "Epoch"]].itertuples(index=False):
        if (event == "pressed") and (key not in active_presses):
            active_presses[key] = {
                "Key": key,
                "Down": epoch,
                "Epochs": [epoch]
            }
        elif (event == "released") and (key in active_presses):
            active_presses[key]["Up"] = epoch
            presses.append(active_presses.pop(key))
        elif (event == "pressed") and (key in active_presses):
            active_presses[key]["Epochs"].append(epoch)
        elif (event == "released") and (key not in active_presses):
            presses.append({
                "Key": key,
                "Down": epoch,
                "Epochs": [epoch],
                "Release Only": True
            })
    
    presses += active_presses.values()
    presses = sorted(presses, key=lambda x: x["Down"]) 
    presses = pd.DataFrame(presses)
    
    return presses

def transform_key_events(key_df):
    return key_df.groupby(level=TEST_LEVEL).apply(_transform_key_events)

def add_key_dynamics(key_df):
    key_df = key_df.copy()
    key_df = transform_key_events(key_df)    

    next_key_df = key_df.groupby(level=TEST_LEVEL).shift(-1).rename({
    "Key": "Next Key",
    "Down": "Next Down",
    "Up": "Next Up"
    }, axis=1)

    key_df = key_df.join(next_key_df[["Next Key", "Next Down", "Next Up"]])
    key_df["Dwell Time"] = key_df["Up"] - key_df["Down"]
    key_df["UU"] = key_df["Next Up"] - key_df["Up"]
    key_df["DD"] = key_df["Next Down"] - key_df["Down"]
    key_df["Flight Time"] = key_df["Next Down"] - key_df["Up"]
    key_df.drop(["Next Key", "Next Down", "Next Up"], axis=1, inplace=True)

    return key_df

def aggregate_basic_key_dynamics(key_df):
    key_df = key_df.copy()

    aggregated_features = key_df.groupby(level=TEST_LEVEL).agg(
    Dwell_median=("Dwell Time", "median"),
    Dwell_iqr=("Dwell Time", lambda x: x.quantile(.75) - x.quantile(.25)),
    Flight_median=("Flight Time", "median"),
    Flight_iqr=("Flight Time", lambda x: x.quantile(.75) - x.quantile(.25)),
    DD_median=("DD", "median"),
    DD_iqr=("DD", lambda x: x.quantile(.75) - x.quantile(.25)),
    UU_median=("UU", "median"),
    UU_iqr=("UU", lambda x: x.quantile(.75) - x.quantile(.25))
    )

    return aggregated_features

def aggregate_selected_key_counts(key_df, key_groups):

    key_df = key_df.copy()
    
    counts = key_df.groupby(level=TEST_LEVEL).apply(
        lambda g: pd.Series({
        f"{group}_count": g["Key"].isin(keys).sum().astype("int")
        for group, keys in key_groups.items()
        })
    )

    return counts


def aggregate_selected_key_dynamics(key_df, key_groups, features):
    iqr=lambda x: x.quantile(.75) - x.quantile(.25)

    key_df = key_df.copy()
    funcs = ["mean", iqr]
    
    counts = key_df.groupby(level=TEST_LEVEL).apply(
        lambda g: pd.Series({
        f"{group}_{feature}_{func}": g[g["Key"].isin(keys)][feature].agg(func)
        for group, keys in key_groups.items()
        for feature in features
        for func in funcs
        })
    )

    return counts
    