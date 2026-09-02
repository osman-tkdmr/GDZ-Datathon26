#cv.py
"""Zaman bazli (rolling-origin) cross validation bolme fonksiyonlari."""
import pandas as pd


def rolling_time_splits(df, n_splits=3, valid_days=60):
    """Son tarihten geriye dogru n_splits adet (train, valid) fold'u uretir.

    Her fold, bir onceki fold'un train setinden valid_days kadar daha az veri
    icerir (rolling-origin). Donen liste kronolojik siradadir (en eski fold ilk).
    """
    df = df.sort_values("tarih")
    max_date = df["tarih"].max()
    folds = []

    for i in range(n_splits):
        valid_end = max_date - pd.Timedelta(days=i * valid_days)
        valid_start = valid_end - pd.Timedelta(days=valid_days - 1)

        train_mask = df["tarih"] < valid_start
        valid_mask = (df["tarih"] >= valid_start) & (df["tarih"] <= valid_end)

        tr = df.loc[train_mask].copy()
        va = df.loc[valid_mask].copy()

        if len(tr) == 0 or len(va) == 0:
            continue

        folds.append((tr, va))

    return list(reversed(folds))
