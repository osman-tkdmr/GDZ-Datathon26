import numpy as np
import pandas as pd


def add_calendar_features(df):
    df = df.copy()
    dt = df["tarih"]

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["dayofweek"] = dt.dt.dayofweek.astype("int8")
    df["dayofyear"] = dt.dt.dayofyear.astype("int16")
    df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    df["is_month_start"] = dt.dt.is_month_start.astype("int8")
    df["is_month_end"] = dt.dt.is_month_end.astype("int8")

    return df


def add_numeric_features(df):
    df = df.copy()
    df["guc"] = pd.to_numeric(df["guc"], errors="coerce").fillna(0)
    df["guc_log"] = np.log1p(df["guc"].clip(lower=0))
    return df


def prepare_features(df):
    df = add_calendar_features(df)
    df = add_numeric_features(df)

    for col in ["tanim", "lokasyon", "il", "bolge", "ilce"]:
        df[col] = df[col].fillna("BILINMIYOR").astype(str)

    return df


def align_categories(train_df, valid_df, test_df, categorical_cols):
    for col in categorical_cols:
        categories = pd.Index(
            pd.concat(
                [
                    train_df[col].astype(str),
                    valid_df[col].astype(str),
                    test_df[col].astype(str),
                ],
                ignore_index=True,
            ).unique()
        )

        train_df[col] = pd.Categorical(train_df[col], categories=categories)
        valid_df[col] = pd.Categorical(valid_df[col], categories=categories)
        test_df[col] = pd.Categorical(test_df[col], categories=categories)

    return train_df, valid_df, test_df
