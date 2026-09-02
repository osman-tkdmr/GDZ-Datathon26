#validation.py
import numpy as np
import pandas as pd


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    return np.sqrt(
        np.mean(
            (np.log1p(y_pred) - np.log1p(np.clip(y_true, 0, None))) ** 2
        )
    )


def temporal_split(df, valid_days=60):
    cutoff = df["tarih"].max() - pd.Timedelta(days=valid_days - 1)

    train_mask = df["tarih"] < cutoff
    valid_mask = df["tarih"] >= cutoff

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    return train_df, valid_df, cutoff
