import numpy as np
import pandas as pd

TEST_LEVEL = ["User ID", "Is Genuine", "Test ID"]
MOVEMENT_LEVEL = TEST_LEVEL + ["Movement ID"]

def add_mouse_kinematics(df):
    df = df.copy()
    # Raw attempt metadata stays on the source table. pandas deep-copies attrs
    # into every group/Series, which is prohibitively costly for event features.
    df.attrs = {}
    df[['X', 'Y']] = df["Coordinates"].to_list()

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=MOVEMENT_LEVEL, sort=False)
    df["dX"] = g["X"].diff()
    df["dY"] = g["Y"].diff()
    df["dS"] = np.hypot(df["dX"], df["dY"])
    df["dT"] = g["Epoch"].diff()
    if df["dT"].lt(0).any():
        raise ValueError("Mouse events must be chronological within each movement")
    # Keep tied-time coordinates for path geometry. Their instantaneous rates
    # are undefined; never remove the event or divide by zero to obtain them.
    positive_dt = df["dT"].where(df["dT"].gt(0))
    df["V"] = df["dS"] / positive_dt

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=MOVEMENT_LEVEL, sort=False)
    df["A"] = g["V"].diff() / positive_dt

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=MOVEMENT_LEVEL, sort=False)
    df["Jerk"] = g["A"].diff() / positive_dt

    return df


def add_button_features(mouse_df, scroll_threshold=0.3):
    MOUSE_BUTTONS = ["left", "right", "mouse4"]
    SCROLL_TYPES = ["scrollup", "scrolldown"]

    mouse_df = mouse_df.copy()
    mouse_df.attrs = {}

    button_features = mouse_df.loc[mouse_df["Event"] != "movement"].copy()
    button_features[["Button", "Button Event"]] = button_features["Event"].str.split(" ", n=1, expand=True).reindex(columns=[0, 1])

    click_features = button_features.loc[button_features["Button"].isin(MOUSE_BUTTONS)]
    g = click_features.groupby([
        pd.Grouper(level="User ID"),
        pd.Grouper(level="Is Genuine"),
        pd.Grouper(level="Test ID"),
        "Button"
        ])
    button_features["Next Button Event"] = g["Button Event"].shift(-1)
    button_features["Next Epoch"] = g["Epoch"].shift(-1)

    valid_press = (button_features["Button Event"] == "press") & (button_features["Next Button Event"] == "release")
    button_features["Dwell Time"] = np.where(valid_press, button_features["Next Epoch"] - button_features["Epoch"], np.nan)
    
    
    scroll = button_features.loc[button_features["Button"].isin(SCROLL_TYPES) & button_features["Button Event"].eq("press")].copy()
    g = scroll.groupby(level=TEST_LEVEL)
    scroll["Interscroll Time"] = g["Epoch"].diff()

    scroll["New Burst"] = (
        scroll["Interscroll Time"].isna()
        | scroll["Interscroll Time"].gt(scroll_threshold)
    )

    scroll["Burst ID"] = (
        scroll.groupby(level=TEST_LEVEL, sort=False)["New Burst"]
        .cumsum()
    )


    return mouse_df.join(button_features[["Dwell Time"]]).join(scroll[["Burst ID"]]).fillna({"Burst ID": 0})


def derive_per_movement_features(mouse_df, min_Path=5.):
    movement_df = mouse_df.loc[mouse_df["Event"] == "movement"].copy()
    movement_df.attrs = {}
    
    displacement = movement_df.groupby(level=MOVEMENT_LEVEL).apply(
        lambda group: np.hypot(group["X"].iloc[-1] - group["X"].iloc[0], group["Y"].iloc[-1] - group["Y"].iloc[0])
    ).rename("Displacement")

    
    per_movement_features = movement_df.groupby(level=MOVEMENT_LEVEL).agg(
        Path=("dS", "sum"),
        Duration=("dT", "sum"),
        V_q10=("V", lambda x: x.quantile(.10)),
        V_median=("V", "median"),
        V_q90=("V", lambda x: x.quantile(.90)),
        A_min_q20=("A", lambda x: x[x < 0].quantile(.20)),
        A_plus_q80=("A", lambda x: x[x > 0].quantile(.80)),
        Jerk_min_q20=("Jerk", lambda x: x[x < 0].quantile(.20)),
        Jerk_plus_q80=("Jerk", lambda x: x[x > 0].quantile(.80))
    )


    per_movement_features = (
        per_movement_features
        .join(displacement, how="outer")
    )

    per_movement_features["Straightness"] = per_movement_features["Displacement"] / per_movement_features["Path"].where(per_movement_features["Path"].gt(0))
    per_movement_features["V_avg"] = per_movement_features["Path"] / per_movement_features["Duration"].where(per_movement_features["Duration"].gt(0))


    return per_movement_features[per_movement_features["Path"] >= min_Path]


def aggregate_movement_features(mouse_df):
    per_movement_features = derive_per_movement_features(mouse_df)
    
    aggregated_features = per_movement_features.groupby(level=["User ID", "Is Genuine", "Test ID"]).agg(
    Path_median = ("Path", "median"),
    Path_iqr = ("Path", lambda x: x.quantile(.75) - x.quantile(.25)),
    Duration_median = ("Duration", "median"),
    Duration_iqr = ("Duration", lambda x: x.quantile(.75) - x.quantile(.25)),
    V_real_mean=("V_avg", "mean"),
    V_real_std=("V_avg", "std"),
    V_q10_mean=("V_q10", "mean"),
    V_median_mean=("V_median", "mean"),
    V_q90_mean=("V_q90", "mean"),
    Straightness_mean=("Straightness", "mean"),
    Straightness_std=("Straightness", "std"),
    Straightness_median=("Straightness", "median"),
    A_min_mean=("A_min_q20", "mean"),
    A_plus_mean=("A_plus_q80", "mean"),
    Jerk_min_mean=("Jerk_min_q20", "mean"),
    Jerk_plus_mean=("Jerk_plus_q80", "mean"),
    Movement_count=("Path", "count")
    )

    return aggregated_features

def aggregate_movement_features_v2(mouse_df):
    per_movement_features = derive_per_movement_features(mouse_df)
    
    agg_funcs = {"mean": "mean", "std": "std", "median": "median",
                 "iqr": lambda x: x.quantile(.75) - x.quantile(.25)}

    aggs = {}
    for col in per_movement_features:
        for name, func in agg_funcs.items():
            aggs[f"{col}_{name}"] = (col, func)

    return per_movement_features.groupby(level=TEST_LEVEL).agg(**aggs)


def aggregate_button_features(mouse_df):

    aggregated_features = mouse_df.groupby(level=TEST_LEVEL).agg(
    Burst_count=("Burst ID", "max"),
    Click_Dwell_Time_mean=("Dwell Time", "mean"),
    Click_Dwell_Time_std=("Dwell Time", "std"),
    Click_count=("Dwell Time", "count")
    )
    # aggregated_features.fillna(0, inplace=True)

    return aggregated_features
