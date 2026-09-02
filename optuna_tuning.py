#optuna_tuning.py

"""Optuna ile LightGBM, CatBoost ve XGBoost hiperparametre optimizasyonu
(zaman bazli rolling CV).

Kullanim:
    from optuna_tuning import run_study
    study_lgb = run_study(train_df, model_type="lgb", n_trials=50, n_splits=5)
    study_cat = run_study(train_df, model_type="cat", n_trials=50, n_splits=5)
    study_xgb = run_study(train_df, model_type="xgb", n_trials=50, n_splits=5)
"""
import numpy as np
import optuna

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, RANDOM_STATE
from cv import rolling_time_splits
from features import align_categories
from validation import rmsle
from models_zoo import fit_lgbm, predict_lgbm, fit_catboost, predict_catboost, fit_xgboost, predict_xgboost

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _lgb_trial_params(trial):
    return {
        "objective": "regression",
        "metric": "rmse",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lgb_fit_predict(X_tr, y_tr, X_va, y_va, params):
    model = fit_lgbm(
        X_tr, y_tr, X_va, y_va,
        categorical_features=CATEGORICAL_FEATURES, params=params,
    )
    return predict_lgbm(model, X_va)


def _cat_trial_params(trial):
    return {
        "loss_function": "RMSE",
        "iterations": 3000,
        "random_seed": RANDOM_STATE,
        "early_stopping_rounds": 100,
        "verbose": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }


def _cat_fit_predict(X_tr, y_tr, X_va, y_va, params):
    model = fit_catboost(X_tr, y_tr, X_va, y_va, cat_features=CATEGORICAL_FEATURES, params=params)
    return predict_catboost(model, X_va, CATEGORICAL_FEATURES)


def _xgb_trial_params(trial):
    return {
        "objective": "reg:squarederror",
        "n_estimators": 3000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 100,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgb_fit_predict(X_tr, y_tr, X_va, y_va, params):
    model = fit_xgboost(X_tr, y_tr, X_va, y_va, params)
    return predict_xgboost(model, X_va)


MODEL_REGISTRY = {
    "lgb": (_lgb_trial_params, _lgb_fit_predict),
    "cat": (_cat_trial_params, _cat_fit_predict),
    "xgb": (_xgb_trial_params, _xgb_fit_predict),
}


def make_objective(train_df, model_type="lgb", n_splits=3, valid_days=60):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model_type: {model_type!r}. Secenekler: {list(MODEL_REGISTRY)}")
    trial_params_fn, fit_predict_fn = MODEL_REGISTRY[model_type]

    folds = rolling_time_splits(train_df, n_splits=n_splits, valid_days=valid_days)

    def objective(trial):
        params = trial_params_fn(trial)

        fold_scores = []
        for tr_part, va_part in folds:
            tr_part, va_part, _ = align_categories(
                tr_part, va_part, va_part.iloc[0:0].copy(), CATEGORICAL_FEATURES
            )
            X_tr = tr_part[FEATURES]
            y_tr = np.log1p(tr_part[TARGET].clip(lower=0))
            X_va = va_part[FEATURES]
            y_va = np.log1p(va_part[TARGET].clip(lower=0))

            pred = fit_predict_fn(X_tr, y_tr, X_va, y_va, params)
            fold_scores.append(rmsle(va_part[TARGET].values, pred))

            trial.report(float(np.mean(fold_scores)), step=len(fold_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def run_study(train_df, model_type="lgb", n_trials=60, n_splits=3, valid_days=60, study_name=None):
    study_name = study_name or f"{model_type}_rmsle"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner, study_name=study_name
    )
    study.optimize(
        make_objective(train_df, model_type, n_splits, valid_days),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study
