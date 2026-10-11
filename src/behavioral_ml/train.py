import pandas as pd
from sklearn.base import clone

def split_for_anomaly(X, y, enroll=5):
    
    X_enrolls = {}
    X_evals = {}
    y_evals = {}
    
    for user_id in X:
        X_enroll = X[user_id].loc[(True, 1):(True, enroll)].copy()
        X_eval_true = X[user_id].loc[(True, enroll+1):(True, 10)].copy()
        X_eval_false = X[user_id].loc[(False, 1):(False, 10)].copy()
        X_eval = pd.concat([X_eval_true, X_eval_false])
        X_enrolls[user_id] = X_enroll
        X_evals[user_id] = X_eval
        y_evals[user_id] = X_eval.index.get_level_values("Is Genuine")

    return X_enrolls, X_evals, y_evals


def split_by_user(test_level_df):
    X = {user_id: test_level_df.loc[(user_id)] for user_id in range(1, 89)}
    y = {user_id: test_level_df.loc[(user_id)].index.get_level_values("Is Genuine") for user_id in range(1, 89)}

    return X, y


def train_binary_classifier(model, cv, X, y):
    y_true = []
    y_score = []
    for user_id in range(1,89):
        for train_idx, test_idx in cv.split(X[user_id], y[user_id]):
            model_fold = clone(model)
            model_fold.fit(X[user_id].iloc[train_idx], y[user_id][train_idx])
            y_score.extend(model_fold.predict_proba(X[user_id].iloc[test_idx])[:,1]) 
            y_true.extend(y[user_id][test_idx])

    return y_true, y_score


def train_anomaly_detector(preprocessing_pipeline, distance_function, X, y):
    X_enrolls, X_evals, y_evals = split_for_anomaly(X, y, enroll=5)
    
    y_true = []
    y_score = []
    for user_id in range(1, 89):
        X_enroll = preprocessing_pipeline.fit_transform(X_enrolls[user_id])
        X_eval = preprocessing_pipeline.transform(X_evals[user_id])

        template = np.median(X_enroll, axis=0, keepdims=True)

        score = -distance_function(X_eval, template).ravel()
        
        y_true.extend(y_evals[user_id])
        y_score.extend(score)