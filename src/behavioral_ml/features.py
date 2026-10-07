import numpy as np
import pandas as pd

def add_kinematics(df):
    df = df.copy()
    df[['X', 'Y']] = df["Coordinates"].to_list()

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"], sort=False)
    df["dX"] = g["X"].diff()
    df["dY"] = g["Y"].diff()
    df["dS"] = np.hypot(df["dX"], df["dY"])
    df["dT"] = g["Epoch"].diff()
    df["V"] =  df["dS"] / df["dT"]

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"], sort=False)
    df["A"] = g["V"].diff() / df["dT"]

    df_mov = df.loc[df["Event"] == "movement"]
    g = df_mov.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"], sort=False)
    df["Jerk"] = g["A"].diff() / df["dT"]

    return df



def derive_movement_features(df):
    movement_features = df.loc[df["Event"] == "movement"]
    displacement = movement_features.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"]).apply(
        lambda group: np.hypot(group["X"].iloc[-1] - group["X"].iloc[0], group["Y"].iloc[-1] - group["Y"].iloc[0])
    ).rename("Displacement")
    
    aggregated_movement_features = movement_features.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"]).agg(
        Path=("dS", "sum"),
        Duration=("dT", "sum"),
        V_mean=("V", "mean"),
        V_std=("V", "std"),
        V_q10=("V", lambda x: x.quantile(.10)),
        V_median=("V", "median"),
        V_q90=("V", lambda x: x.quantile(.90)),
        a_min_q20=("A", lambda x: x[x < 0].quantile(.20)),
        a_plus_q80=("A", lambda x: x[x > 0].quantile(.80))
    )

    aggregated_movement_features = (
        aggregated_movement_features
        .join(displacement)
    )

    aggregated_movement_features["Straightness"] = aggregated_movement_features["Path"] / aggregated_movement_features["Displacement"]
    aggregated_movement_features["V_real_mean"] = aggregated_movement_features["Path"] / aggregated_movement_features["Duration"]


    return aggregated_movement_features

def derive_movement_features_v2(df):
    df = df.copy()
    movement_features = df.loc[df["Event"] == "movement"]
    displacement = movement_features.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"]).apply(
        lambda group: np.hypot(group["X"].iloc[-1] - group["X"].iloc[0], group["Y"].iloc[-1] - group["Y"].iloc[0])
    ).rename("Displacement")
    
    aggregated_movement_features = movement_features.groupby(level=["User ID", "Is Genuine", "Test ID", "Movement ID"]).agg(
        Path=("dS", "sum"),
        Duration=("dT", "sum"),
        V_mean=("V", "mean"),
        V_std=("V", "std"),
        V_q10=("V", lambda x: x.quantile(.10)),
        V_median=("V", "median"),
        V_q90=("V", lambda x: x.quantile(.90))
    )

    aggregated_movement_features = aggregated_movement_features.join(displacement)

    aggregated_movement_features["Straightness"] = aggregated_movement_features["Path"] / aggregated_movement_features["Displacement"]


    return aggregated_movement_features    

def derive_mouse_clicks_features(df):
    df = df.copy()
    click_features = df.loc[df["Event"] != "movement"]
    