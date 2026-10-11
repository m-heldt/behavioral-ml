import numpy as np
import pandas as pd

TEST_LEVEL = ["User ID", "Is Genuine", "Test ID"]
MOVEMENT_LEVEL = TEST_LEVEL + ["Movement ID"]

def add_mouse_kinematics(df):
    df = df.copy()
    df[['X', 'Y']] = df["Coordinates"].to_list()

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=MOVEMENT_LEVEL, sort=False)
    df["dX"] = g["X"].diff()
    df["dY"] = g["Y"].diff()
    df["dS"] = np.hypot(df["dX"], df["dY"])
    df["dT"] = g["Epoch"].diff()
    df["V"] =  df["dS"] / df["dT"]

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=MOVEMENT_LEVEL, sort=False)
    df["A"] = g["V"].diff() / df["dT"]

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=MOVEMENT_LEVEL, sort=False)
    df["Jerk"] = g["A"].diff() / df["dT"]

    return df


def add_button_features(mouse_df):
    MOUSE_BUTTONS = ["left", "right", "mouse4"]
    SCROLL_TYPES = ["scrollup", "scrolldown"]
    SCROLL_THRESHOLD = 0.3

    mouse_df = mouse_df.copy()

    button_features = mouse_df.loc[(mouse_df["Event"] != "movement") & mouse_df["Button"].isin(MOUSE_BUTTONS)].copy()
    g = button_features.groupby([
        pd.Grouper(level="User ID"),
        pd.Grouper(level="Is Genuine"),
        pd.Grouper(level="Test ID"),
        "Button"
        ])
    button_features["Next Button Event"] = g["Button Event"].shift(-1)
    button_features["Next Epoch"] = g["Epoch"].shift(-1)

    valid_press = (button_features["Button Event"] == "press") & (button_features["Next Button Event"] == "release")
    button_features["Dwell Time"] = np.where(valid_press, button_features["Next Epoch"] - button_features["Epoch"], np.nan)
    
    
    scroll = mouse_df.loc[mouse_df["Button"].isin(SCROLL_TYPES) & mouse_df["Button Event"].eq("press") & (mouse_df["Event"] != "movement")].copy()
    g = scroll.groupby(level=TEST_LEVEL)
    scroll["Interscroll Time"] = g["Epoch"].diff()

    scroll["New Burst"] = (
        scroll["Interscroll Time"].isna()
        | scroll["Interscroll Time"].gt(SCROLL_THRESHOLD)
    )

    scroll["Burst ID"] = (
        scroll.groupby(level=TEST_LEVEL, sort=False)["New Burst"]
        .cumsum()
    )


    return mouse_df.join(button_features[["Dwell Time"]]).join(scroll[["Burst ID"]])


def derive_per_movement_features(mouse_df, min_Path=5.):
    movement_df = mouse_df.loc[mouse_df["Event"] == "movement"].copy()
    
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

    per_movement_features["Straightness"] = per_movement_features["Displacement"] / per_movement_features["Path"]
    per_movement_features["V_avg"] = per_movement_features["Path"] / per_movement_features["Duration"]


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
    
    iqr=lambda x: x.quantile(.75) - x.quantile(.25)
    agg_funcs = ["mean", "std", "median", iqr]

    aggs = {}
    for col in per_movement_features:
        for func in agg_funcs:
            aggs[f"{col}_{func}"] = (f"{col}", func)

    return per_movement_features.groupby(level=TEST_LEVEL).agg(**aggs)


def aggregate_button_features(mouse_df):

    aggregated_features = mouse_df.groupby(level=TEST_LEVEL).agg(
    Burst_count=("Burst ID", "max"),
    Dwell_Time_mean=("Dwell Time", "mean"),
    Dwell_Time_std=("Dwell Time", "std"),
    Click_count=("Dwell Time", "count")
    )
    # aggregated_features.fillna(0, inplace=True)

    return aggregated_features