import json

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest

from behavioral_ml import train, tuning


@pytest.fixture
def tuning_data():
    rows = [(user, genuine, test) for user in [10, 20, 30, 40]
            for genuine in [False, True] for test in range(1, 7)]
    index = pd.MultiIndex.from_tuples(rows, names=train.TEST_LEVEL)
    features = pd.DataFrame({"timing": [user + test / 10 + (0 if genuine else 5)
                                         for user, genuine, test in rows]}, index=index)
    partner = {10: 20, 20: 10, 30: 40, 40: 30}
    candidates = pd.Series([(user,) if genuine else (partner[user],)
                            for user, genuine, _ in rows], index=index)
    candidates.loc[(10, False, 1)] = (20, 30)
    candidates.loc[(10, False, 2)] = ()
    return features, candidates


def test_identity_split_is_deterministic_and_excludes_every_cross_partition_typist(tuning_data):
    features, candidates = tuning_data
    split, audit = tuning.make_identity_split(features, candidates, 3, .5, 42)
    repeated, _ = tuning.make_identity_split(features.sample(frac=1, random_state=8), candidates, 3, .5, 42)
    assert split == repeated
    assert split["development_users"] == [10, 20]
    assert split["evaluation_users"] == [30, 40]
    assert [10, False, 1] not in split["development_probes"]
    assert [10, False, 2] not in split["development_probes"]
    for phase in ["development", "evaluation"]:
        people = set(split[f"{phase}_users"])
        for identity in split[f"{phase}_probes"]:
            assert set(candidates.loc[tuple(identity)]) <= people
        assert split["coverage"][phase]["n_users_with_both_classes"] == 2
    reasons = audit.set_index(train.TEST_LEVEL)["reason"]
    assert reasons.loc[(10, False, 1)] == "cross_split_typist"
    assert reasons.loc[(10, False, 2)] == "unknown_typist"
    assert audit.loc[audit.role.eq("enrollment"), "Is Genuine"].all()


def test_split_refuses_missing_candidates_or_no_impostor_coverage(tuning_data):
    features, candidates = tuning_data
    with pytest.raises(ValueError, match="cover every attempt"):
        tuning.make_identity_split(features, candidates.iloc[:-1], 3, .5)
    unknown = candidates.copy()
    unknown.loc[~unknown.index.get_level_values("Is Genuine")] = pd.Series(
        [()] * 24, index=unknown.loc[~unknown.index.get_level_values("Is Genuine")].index)
    with pytest.raises(ValueError, match="two-class probes"):
        tuning.make_identity_split(features, unknown, 3, .5)


def test_tuning_never_selects_on_evaluation_and_freezes_before_scoring(tuning_data, tmp_path, monkeypatch):
    features, candidates = tuning_data
    original = tuning.score_partition
    phases = []

    def audited_score(detector, frame, split, phase, **kwargs):
        phases.append(phase)
        assert set(frame.index.get_level_values("User ID")) == set(split[f"{phase}_users"])
        assert (tmp_path / "selection.json").exists() == (phase == "evaluation")
        return original(detector, frame, split, phase, **kwargs)

    monkeypatch.setattr(tuning, "score_partition", audited_score)
    results, selection = tuning.run_tuning({"keys": features}, candidates, tmp_path,
                                          n_enroll=3, n_trials=2, evaluation_fraction=.5)
    assert phases == ["development"] * 4 + ["evaluation"] * 2
    assert len(results) == 2
    assert selection["selected_experiment"] in results
    assert sum(scores.attrs["selected_on_development"] for scores in results.values()) == 1
    for name, scores in results.items():
        assert set(scores.index.get_level_values("User ID")) == {30, 40}
        assert scores.attrs["n_enroll"] == 3
        models = joblib.load(tmp_path / "models" / f"{name}.joblib")
        assert set(models) == {30, 40}
        for user, model in models.items():
            frame = features.loc[scores.index[scores.index.get_level_values("User ID") == user]]
            np.testing.assert_allclose(model.decision_function(frame), scores.loc[frame.index])
            assert model.steps[0][1].statistics_[0] == pytest.approx(user + .2)
    before = optuna.get_all_study_summaries(storage=f"sqlite:///{tmp_path / 'optuna.db'}")
    phases.clear()
    again, frozen = tuning.run_tuning({"keys": features}, candidates, tmp_path,
                                     n_enroll=3, n_trials=2, evaluation_fraction=.5)
    after = optuna.get_all_study_summaries(storage=f"sqlite:///{tmp_path / 'optuna.db'}")
    assert phases == ["evaluation"] * 2
    assert [study.n_trials for study in before] == [study.n_trials for study in after] == [2, 2]
    assert frozen == selection
    for name in results:
        pd.testing.assert_series_equal(results[name], again[name])
    with pytest.raises(ValueError, match="new output directory"):
        tuning.run_tuning({"keys": features}, candidates, tmp_path,
                          n_enroll=3, n_trials=3, evaluation_fraction=.5)
    configuration = json.loads((tmp_path / "evaluation/configuration.json").read_text())
    assert configuration["optuna"] == optuna.__version__


def test_partition_scoring_refuses_evaluation_people_in_development_input(tuning_data):
    features, candidates = tuning_data
    split, _ = tuning.make_identity_split(features, candidates, 3, .5)
    with pytest.raises(ValueError, match="only the declared partition"):
        tuning.score_partition(train.TemplateDetector(), features, split, "development")


def test_interrupted_search_resumes_completed_trials_without_evaluating_early(tuning_data, tmp_path, monkeypatch):
    features, candidates = tuning_data
    original = tuning.score_partition
    calls = 0

    def interrupt_second_trial(detector, frame, split, phase, **kwargs):
        nonlocal calls
        assert phase == "development"
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(detector, frame, split, phase, **kwargs)

    monkeypatch.setattr(tuning, "score_partition", interrupt_second_trial)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        tuning.run_tuning({"keys": features}, candidates, tmp_path, n_enroll=3,
                          n_trials=3, evaluation_fraction=.5, detector_names=("knn",))
    assert not (tmp_path / "selection.json").exists()
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    summary = optuna.get_all_study_summaries(storage=storage)[0]
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    first_params = study.trials[0].params
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "FAIL"]
    monkeypatch.setattr(tuning, "score_partition", original)
    tuning.run_tuning({"keys": features}, candidates, tmp_path, n_enroll=3,
                      n_trials=3, evaluation_fraction=.5, detector_names=("knn",))
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    assert study.trials[0].params == first_params
    assert sum(trial.state.name == "COMPLETE" for trial in study.trials) == 3
    assert (tmp_path / "selection.json").exists()
