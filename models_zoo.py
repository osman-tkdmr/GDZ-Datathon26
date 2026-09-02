#models_zoo.py
"""L ensemble icin lgbm CatBoost ve XGBoost sarmalayicilari.

GPU varsa (nvidia-smi calisiyorsa) otomatik denenir; kutuphane GPU
derlemesini desteklemiyorsa veya calisma zamaninda hata verirse otomatik
CPU'ya duser (bkz. gpu_utils.py -- sonuc onbelleklenir, tekrar tekrar
denenmez).

Not: kategorik sutunlar (features.align_categories sonrasi) pandas
'category' dtype'inda olmali. CatBoost bunu string'e cevirip kullanir,
XGBoost >= 1.6 enable_categorical=True ile native destekler.
"""
import numpy as np

from gpu_utils import should_try_gpu, mark_gpu_result, cat_device_params, xgb_device_params, lgb_device_params

def fit_lgbm(X_train,y_train,X_valid,y_valid,categorical_features,params,use_gpu=True):
    import lightgbm as lgb
    def _fit(p):
        model = lgb.LGBMRegressor(**p)
        fit_kwargs = {}
        # Doğrulama seti varsa erken durdurma ve değerlendirme metrikleri
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_X"] = X_valid,
            fit_kwargs["eval_y"] = y_valid,
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ]
        # Kategorik değişkenler varsa ekle
        if categorical_features:
            fit_kwargs["categorical_feature"] = categorical_features
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    # GPU denenmesi uygunsa dene
    if should_try_gpu("lgb", use_gpu):
        try:
            model = _fit({**params, **lgb_device_params()})
            mark_gpu_result("lgb", True)
            return model
        except Exception as exc:
            mark_gpu_result("lgb", False)
            print(f"[model.py] LightGBM GPU kullanilamadi ({exc}); CPU'ya gecildi (bundan sonra hep CPU kullanilacak).")

    return _fit(params)


def predict_lgbm(model, X):
    pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)


def fit_catboost(X_train, y_train, X_valid, y_valid, cat_features, params, use_gpu=True):
    from catboost import CatBoostRegressor, Pool

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = (
        Pool(X_valid, y_valid, cat_features=cat_features)
        if X_valid is not None
        else None
    )

    def _fit(p):
        model = CatBoostRegressor(**p)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=valid_pool is not None,
            verbose=False,
        )
        return model

    if should_try_gpu("cat", use_gpu):
        try:
            model = _fit({**params, **cat_device_params()})
            mark_gpu_result("cat", True)
            return model
        except Exception as exc:
            mark_gpu_result("cat", False)
            print(f"[models_zoo.py] CatBoost GPU kullanilamadi ({exc}); CPU'ya gecildi (bundan sonra hep CPU kullanilacak).")

    return _fit({**params, "task_type": "CPU"})


def predict_catboost(model, X, cat_features):
    from catboost import Pool

    pred_log = model.predict(Pool(X, cat_features=cat_features))
    return np.clip(np.expm1(pred_log), 0, None)


def fit_xgboost(X_train, y_train, X_valid, y_valid, params, use_gpu=True):
    import xgboost as xgb

    def _fit(p):
        model = xgb.XGBRegressor(**p, enable_categorical=True)
        fit_kwargs = {}
        if X_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    if should_try_gpu("xgb", use_gpu):
        try:
            model = _fit({**params, **xgb_device_params()})
            mark_gpu_result("xgb", True)
            return model
        except Exception as exc:
            mark_gpu_result("xgb", False)
            print(f"[models_zoo.py] XGBoost GPU kullanilamadi ({exc}); CPU'ya gecildi (bundan sonra hep CPU kullanilacak).")

    return _fit({**params, "tree_method": "hist", "device": "cpu"})


def predict_xgboost(model, X):
    pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)

DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 1500,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
DEFAULT_CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "early_stopping_rounds": 100,
    "verbose": False,
}

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
