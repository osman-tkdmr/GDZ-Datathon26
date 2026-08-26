import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["tarih"] = pd.to_datetime(train["tarih"])
    test["tarih"] = pd.to_datetime(test["tarih"])

    return train, test


def add_location_hierarchy(df):
    df = df.copy()

    parts = df["lokasyon"].fillna("BILINMIYOR").astype(str).str.split(">")

    df["il"] = parts.str[0].str.strip()
    df["bolge"] = parts.str[1].str.strip() if parts.str.len().max() >= 2 else "BILINMIYOR"
    df["ilce"] = parts.str[2].str.strip() if parts.str.len().max() >= 3 else "BILINMIYOR"

    for col in ["il", "bolge", "ilce"]:
        df[col] = df[col].fillna("BILINMIYOR").astype(str)

    return df
