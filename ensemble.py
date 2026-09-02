#ensemble.py
"""Birden fazla modelin tahminini RMSLE'yi minimize edecek sekilde birlestirir."""
import numpy as np
from scipy.optimize import minimize

from validation import rmsle


def optimize_weights(y_true, pred_dict):
    """pred_dict: {model_adi: tahmin_array}.

    Validation seti uzerinde RMSLE'yi minimize eden (toplami 1, negatif olmayan)
    agirliklari bulur.
    """
    names = list(pred_dict.keys())
    preds = np.column_stack([pred_dict[n] for n in names])

    def loss(w):
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        blended = preds @ w
        return rmsle(y_true, blended)

    x0 = np.ones(len(names)) / len(names)
    bounds = [(0, 1)] * len(names)
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=cons)

    raw_weights = np.clip(result.x, 0, None)
    total = raw_weights.sum() or 1.0
    weights = dict(zip(names, raw_weights / total))
    return weights, float(result.fun)


def blend(pred_dict, weights):
    first = next(iter(pred_dict.values()))
    total = np.zeros_like(first, dtype=float)
    for name, w in weights.items():
        total += w * pred_dict[name]
    return total


def stack_with_ridge(y_true, pred_dict, alpha=1.0):
    """Log-tahminler uzerinde pozitif katsayili Ridge meta-model (stacking)."""
    from sklearn.linear_model import Ridge

    names = list(pred_dict.keys())
    X = np.column_stack([np.log1p(pred_dict[n]) for n in names])
    y = np.log1p(np.clip(y_true, 0, None))

    meta = Ridge(alpha=alpha, positive=True)
    meta.fit(X, y)
    return meta, names


def predict_stack(meta, names, pred_dict):
    X = np.column_stack([np.log1p(pred_dict[n]) for n in names])
    pred_log = meta.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)


def seed_bag_predictions(fit_fn, predict_fn, seeds, *fit_args, **fit_kwargs):
    """Ayni modeli farkli seed'lerle egitip tahminleri ortalar (varyans azaltma)."""
    preds = []
    for seed in seeds:
        fit_kwargs["params"] = {**fit_kwargs.get("params", {}), "random_state": seed}
        model = fit_fn(*fit_args, **fit_kwargs)
        preds.append(predict_fn(model))
    return np.mean(preds, axis=0)
