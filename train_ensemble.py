"""Optuna ile ayri ayri tune edilmis LightGBM + CatBoost + XGBoost ensemble.

Akis:
  1. Veri yukle, ozellikleri hazirla (mevcut data.py / features.py)
  2. Her uc model icin AYRI Optuna calismasi (rolling CV)
  3. En iyi parametrelerle tek bir temporal split uzerinde validation tahmini uret
  4. Validation setinde RMSLE'yi minimize eden agirliklarla blend et
  5. Full train uzerinde ayni ucluyu (validation-stage'de bulunan best_iteration
     sayisiyla, early stopping olmadan) yeniden egit, test tahminini blend et,
     submission.csv yaz

Not: calistirmadan once  pip install optuna catboost xgboost  gerekir.
"""
import gc

import numpy as np
import pandas as pd

from config import (
    TRAIN_PATH, TEST_PATH, TARGET, ENTITY_COL, DATE_COL, OUTPUT_DIR,
    VALID_DAYS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
from data import load_data, add_location_hierarchy
from features import prepare_features, align_categories
from validation import temporal_split, rmsle
from models_zoo import (
    fit_lgbm, predict_lgbm, fit_catboost, predict_catboost, fit_xgboost, predict_xgboost,
    DEFAULT_CATBOOST_PARAMS, DEFAULT_XGB_PARAMS, DEFAULT_LGB_PARAMS,
)
from ensemble import optimize_weights, blend
from optuna_tuning import run_study

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_dataset():
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    train = add_location_hierarchy(train)
    test = add_location_hierarchy(test)
    train = prepare_features(train)
    test = prepare_features(test)
    return train, test


def tune_all_models(train, n_trials, n_splits, valid_days):
    """Her model icin ayri Optuna calismasi yapar, tune edilmis parametre
    sozluklerini dondurur (validation-stage icin, early stopping AKTIF)."""
    print("Optuna araniyor (LightGBM)...")
    study_lgb = run_study(train, model_type="lgb", n_trials=n_trials, n_splits=n_splits, valid_days=valid_days)
    print(f"  LightGBM en iyi CV RMSLE: {study_lgb.best_value:.6f}  params: {study_lgb.best_params}")

    print("Optuna araniyor (CatBoost)...")
    study_cat = run_study(train, model_type="cat", n_trials=n_trials, n_splits=n_splits, valid_days=valid_days)
    print(f"  CatBoost en iyi CV RMSLE: {study_cat.best_value:.6f}  params: {study_cat.best_params}")

    print("Optuna araniyor (XGBoost)...")
    study_xgb = run_study(train, model_type="xgb", n_trials=n_trials, n_splits=n_splits, valid_days=valid_days)
    print(f"  XGBoost en iyi CV RMSLE: {study_xgb.best_value:.6f}  params: {study_xgb.best_params}")

    lgb_params = {**DEFAULT_LGB_PARAMS, **study_lgb.best_params, "n_estimators": 3000}
    cat_params = {**DEFAULT_CATBOOST_PARAMS, **study_cat.best_params}
    xgb_params = {**DEFAULT_XGB_PARAMS, **study_xgb.best_params, "early_stopping_rounds": 100}

    return lgb_params, cat_params, xgb_params


def train_all_models(train_part, valid_part, lgb_params, cat_params, xgb_params):
    train_part, valid_part, _ = align_categories(
        train_part, valid_part, valid_part.iloc[0:0].copy(), CATEGORICAL_FEATURES
    )

    X_train, X_valid = train_part[FEATURES], valid_part[FEATURES]
    y_train = np.log1p(train_part[TARGET].clip(lower=0))
    y_valid = np.log1p(valid_part[TARGET].clip(lower=0))

    preds = {}

    lgb_model = fit_lgbm(
        X_train, y_train, X_valid, y_valid,
        categorical_features=CATEGORICAL_FEATURES, params=lgb_params,
    )
    preds["lgb"] = predict_lgbm(lgb_model, X_valid)

    cat_model = fit_catboost(
        X_train, y_train, X_valid, y_valid,
        cat_features=CATEGORICAL_FEATURES, params=cat_params,
    )
    preds["cat"] = predict_catboost(cat_model, X_valid, CATEGORICAL_FEATURES)

    xgb_model = fit_xgboost(X_train, y_train, X_valid, y_valid, xgb_params)
    preds["xgb"] = predict_xgboost(xgb_model, X_valid)

    models = {"lgb": lgb_model, "cat": cat_model, "xgb": xgb_model}
    return models, preds, valid_part[TARGET].values


def main(n_optuna_trials=1, n_splits=3):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_dataset()

    # 1) Her uc model icin ayri Optuna calismasi
    lgb_params, cat_params, xgb_params = tune_all_models(
        train, n_trials=n_optuna_trials, n_splits=n_splits, valid_days=VALID_DAYS
    )

    # 2) Tek temporal split uzerinde uc modeli egit (early stopping ile en iyi iterasyon sayisini da ogreniyoruz)
    train_part, valid_part, cutoff = temporal_split(train, VALID_DAYS)
    print(f"Validation cutoff: {cutoff.date()}")

    models, valid_preds, y_valid_true = train_all_models(
        train_part, valid_part, lgb_params, cat_params, xgb_params
    )

    for name, pred in valid_preds.items():
        print(f"  {name} RMSLE: {rmsle(y_valid_true, pred):.6f}")

    # 3) Agirliklari validation setinde optimize et
    weights, blended_score = optimize_weights(y_valid_true, valid_preds)
    print("Ensemble agirliklari:", weights)
    print(f"Ensemble validation RMSLE: {blended_score:.6f}")

    valid_output = valid_part[[ENTITY_COL, DATE_COL, TARGET]].copy()
    for name, pred in valid_preds.items():
        valid_output[f"tahmin_{name}"] = pred
    valid_output["tahmin_ensemble"] = blend(valid_preds, weights)
    valid_output.to_csv(OUTPUT_DIR / "validation_predictions_ensemble.csv", index=False)

    lgb_best_iter = getattr(models["lgb"], "best_iteration_", None)
    cat_best_iter = models["cat"].get_best_iteration()
    xgb_best_iter = getattr(models["xgb"], "best_iteration", None)

    del models, valid_preds, train_part, valid_part
    gc.collect()

    all_train, _, test = align_categories(
        train, test.iloc[0:0].copy(), test, CATEGORICAL_FEATURES
    )
    del train
    gc.collect()

    X_full = all_train[FEATURES]
    X_test = test[FEATURES]
    y_full = np.log1p(all_train[TARGET].clip(lower=0))

    final_lgb_params = {**lgb_params}
    if lgb_best_iter:
        final_lgb_params["n_estimators"] = lgb_best_iter
    final_lgb = fit_lgbm(
        X_full, y_full, None, None, 
        categorical_features=CATEGORICAL_FEATURES, params=final_lgb_params,
        use_gpu=False,
    )

    final_cat_params = {**cat_params, "early_stopping_rounds": None}
    if cat_best_iter:
        final_cat_params["iterations"] = cat_best_iter + 1
    final_cat = fit_catboost(
        X_full, y_full, None, None,
        cat_features=CATEGORICAL_FEATURES, params=final_cat_params,
        use_gpu=False,
    )
    gc.collect()

    final_xgb_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    if xgb_best_iter is not None:
        final_xgb_params["n_estimators"] = xgb_best_iter + 1
    final_xgb = fit_xgboost(
        X_full, y_full, None, None, 
        final_xgb_params, 
        use_gpu=False,
    )
    gc.collect()

    test_preds = {
        "lgb": predict_lgbm(final_lgb, X_test),
        "cat": predict_catboost(final_cat, X_test, CATEGORICAL_FEATURES),
        "xgb": predict_xgboost(final_xgb, X_test),
    }
    test_ensemble = blend(test_preds, weights)

    submission = test[["id"]].copy()
    submission[TARGET] = test_ensemble
    submission.to_csv(OUTPUT_DIR / "submission_ensemble.csv", index=False)
    print(f"Submission kaydedildi: {OUTPUT_DIR / 'submission_ensemble.csv'}")


if __name__ == "__main__":
    main()
