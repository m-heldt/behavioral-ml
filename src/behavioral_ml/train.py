import pandas as pd
import numpy as np
from sklearn.base import clone

def split_for_novelty(test_level_df, enroll=5, by="user"):
    X, y = split_for_binary(test_level_df)
    
    if by == "user":
        X_enrolls = {}
        X_evals = {}
        y_evals = {}
        
        for user_id in X:
            X_enroll = X[user_id].loc[(user_id, True, 1):(user_id, True, enroll)].copy()
            X_eval_true = X[user_id].loc[(user_id, True, enroll+1):(user_id, True, 10)].copy()
            X_eval_false = X[user_id].loc[(user_id, False, 1):(user_id, False, 10)].copy()
            X_eval = pd.concat([X_eval_true, X_eval_false])
            X_enrolls[user_id] = X_enroll
            X_evals[user_id] = X_eval
            y_evals[user_id] = X_eval.index.get_level_values("Is Genuine")

        else:
            pass

    return X_enrolls, X_evals, y_evals


def get_differential_dataset(test_level_df, enroll=5, template_strategy="median", by=None):
    X, y = split_for_binary(test_level_df)

    X_evals = []
    y_evals = []
    
    for user_id in X:
        X_enroll = X[user_id].loc[(user_id, True, 1):(user_id, True, enroll)].copy()
        user_template = X_enroll.agg(template_strategy)

        X_eval_true = X[user_id].loc[(user_id, True, enroll+1):(user_id, True, 10)].copy()
        X_eval_false = X[user_id].loc[(user_id, False, 1):(user_id, False, 10)].copy()
        X_eval = pd.concat([X_eval_true, X_eval_false])
        X_eval -= user_template

        X_evals.append(X_eval)

    X_evals = pd.concat(X_evals, axis=0)
    y_evals = X_evals.index.get_level_values("Is Genuine")

    if by == "user":
        return split_for_binary(X_evals)

    return X_evals, y_evals


def split_for_binary(test_level_df, by="user"):
    if by == "user":
        X = {user_id: test_level_df.loc[[user_id]] for user_id in range(1, 89)}
        y = {user_id: test_level_df.loc[[user_id]].index.get_level_values("Is Genuine") for user_id in range(1, 89)}
    else:
        X = test_level_df.copy()
        y = test_level_df.index.get_level_values("Is Genuine")

    return X, y


def train_binary_classifier(model, cv, X, y, metrics=None):
    y_true = {}
    y_score = {}
    results = {}
    for user_id in range(1,89):
        user_true = []
        user_score = []
        user_results = {}
        accuracy_scores = []
        
        for train_idx, test_idx in cv.split(X[user_id], y[user_id]):
            model_fold = clone(model)
            model_fold.fit(X[user_id].iloc[train_idx], y[user_id][train_idx])
            
            score = model_fold.predict_proba(X[user_id].iloc[test_idx])[:,1]
            true = y[user_id][test_idx]
            accuracy_scores.append(model_fold.score(X[user_id].iloc[test_idx], y[user_id][test_idx]))

            user_score.extend(score)
            user_true.extend(true)

        y_score[user_id] = user_score 
        y_true[user_id] = user_true
        user_results["accuracy"] = np.mean(accuracy_scores)
        
        if metrics != None:
            for m in metrics:
                user_results[m] = metrics[m](user_true, user_score)
        
        results[user_id] = user_results

    return y_true, y_score, results


def train_anomaly_detector(preprocessing_pipeline, distance_function, X_enrolls, X_evals, y_evals, metrics=None):
    
    
    y_true = []
    y_score = []
    for user_id in range(1, 89):
        X_enroll = preprocessing_pipeline.fit_transform(X_enrolls[user_id])
        X_eval = preprocessing_pipeline.transform(X_evals[user_id])

        template = np.median(X_enroll, axis=0, keepdims=True)

        score = -distance_function(X_eval, template).ravel()
        
        y_true.extend(y_evals[user_id])
        y_score.extend(score)

    return y_true, y_score