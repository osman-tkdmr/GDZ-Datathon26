import numpy as np
import lightgbm as lgb


def fit_model(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    categorical_features=None,
    params=None,
):
    model_params = params.copy()

    model = lgb.LGBMRegressor(**model_params)

    fit_kwargs = {
        "categorical_feature": categorical_features or [],
    }

    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_X"] = X_valid,
        fit_kwargs["eval_y"] = y_valid,
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ]

    model.fit(X_train, y_train, **fit_kwargs)

    return model


def predict_log_model(model, X):
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    return np.clip(pred, 0, None)
