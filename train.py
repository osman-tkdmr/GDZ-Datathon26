from config import (
    TRAIN_PATH,
    TEST_PATH,
    TARGET,
    ENTITY_COL,
    DATE_COL,
    OUTPUT_DIR,
    VALID_DAYS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    LGB_PARAMS,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import load_data, add_location_hierarchy
from features import prepare_features, align_categories
from validation import temporal_split, rmsle
from model import fit_model, predict_log_model


def save_feature_importance(model, features, output_dir, top_n=25):
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    plot_data = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.32)))
    ax.barh(plot_data["feature"], plot_data["importance"], color="#176b87")
    ax.set_title("Özellik Önemleri (LightGBM gain)")
    ax.set_xlabel("Toplam gain")
    ax.set_ylabel("Özellik")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_data(TRAIN_PATH, TEST_PATH)

    train = add_location_hierarchy(train)
    test = add_location_hierarchy(test)

    train = prepare_features(train)
    test = prepare_features(test)

    train_part, valid_part, cutoff = temporal_split(train, VALID_DAYS)

    train_part, valid_part, test = align_categories(
        train_part,
        valid_part,
        test,
        CATEGORICAL_FEATURES,
    )

    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    print(features)

    X_train = train_part[features]
    X_valid = valid_part[features]

    # RMSLE ile hizalı hedef
    y_train = __import__("numpy").log1p(train_part[TARGET].clip(lower=0))
    y_valid = __import__("numpy").log1p(valid_part[TARGET].clip(lower=0))

    model = fit_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_features=CATEGORICAL_FEATURES,
        params=LGB_PARAMS,
    )

    valid_pred = predict_log_model(model, X_valid)
    score = rmsle(valid_part[TARGET].values, valid_pred)

    print("=" * 60)
    print(f"Validation cutoff : {cutoff.date()}")
    print(f"Train rows        : {len(train_part):,}")
    print(f"Validation rows   : {len(valid_part):,}")
    print(f"Best iteration    : {getattr(model, 'best_iteration_', None)}")
    print(f"Validation RMSLE  : {score:.6f}")
    print("=" * 60)

    # Validation tahminlerini sakla
    valid_output = valid_part[[ENTITY_COL, DATE_COL, TARGET]].copy()
    valid_output["tahmin"] = valid_pred
    valid_output.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    # Full train üzerinde tekrar eğit
    all_train, _, test = align_categories(
        train,
        test.iloc[0:0].copy(),
        test,
        CATEGORICAL_FEATURES,
    )

    X_full = all_train[features]
    X_test = test[features]
    y_full = __import__("numpy").log1p(all_train[TARGET].clip(lower=0))

    full_params = LGB_PARAMS.copy()
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter:
        full_params["n_estimators"] = best_iter

    final_model = fit_model(
        X_full,
        y_full,
        categorical_features=CATEGORICAL_FEATURES,
        params=full_params,
    )

    save_feature_importance(final_model, features, OUTPUT_DIR)

    test_pred = predict_log_model(final_model, X_test)

    submission = test[["id"]].copy()
    submission[TARGET] = test_pred
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    print(f"Submission saved : {OUTPUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
