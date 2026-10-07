from behavioral_ml.etl import load_data
from behavioral_ml.features import add_kinematics, derive_movement_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow

def main():
    # mlflow.autolog()
    
    mouse_df, key_df = load_data("raw_kmt_dataset")

    mouse_df = add_kinematics(mouse_df)

    movement_features = derive_movement_features(mouse_df)
    print(movement_features)
    # test_level_features = movement_features.groupby(level=["User ID", "Is Genuine", "Test ID"]).agg(
    #     Path_mean = ("Path", "mean"),
    #     Path_std = ("Path", "std"),
    #     Duration_mean = ("Duration", "mean"),
    #     Duration_std = ("Duration", "std"),
    #     V_mean=("V_mean", "mean"),
    #     V_std=("V_mean", "std"),
    #     V_q10=("V_mean", lambda x: x.quantile(.10)),
    #     V_median=("V_mean", "median"),
    #     V_q90=("V_mean", lambda x: x.quantile(.90))
    # )
    # print(test_level_features)

    # scores_list = []
    # for user in range(1, 89):

    #     X = test_level_features.loc[(user)]
    #     y = X.index.get_level_values("Is Genuine").astype(int)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

    #     cv = StratifiedKFold(
    #     n_splits=5,
    #     shuffle=True,
    #     random_state=42,
    #     )

    #     clf = DecisionTreeClassifier(random_state=42, )

    #     scores = cross_val_score(
    #         clf,
    #         X,
    #         y,
    #         cv=cv,
    #     )
    #     scores_list.append(scores.mean())
    #     print(scores.mean(), scores.std())

    # print(np.mean(scores_list), np.std(scores_list))


if __name__ == "__main__":
    main()