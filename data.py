import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["tarih"] = pd.to_datetime(train["tarih"])
    test["tarih"] = pd.to_datetime(test["tarih"])

    return train, test


def add_location_hierarchy(df):
    df = df.copy()

    df["il"] = df["lokasyon"].str.split(">").str[0]
    df["bolge"] = df["lokasyon"].apply(lambda x: x.split(">")[1] if len(x.split(">")) == 3 else None)
    df["ilce"] = df["lokasyon"].apply(lambda x: x.split(">")[1] if len(x.split(">")) == 2 else (x.split(">")[2]))

    for col in ["il", "bolge", "ilce"]:
        df[col] = df[col].fillna("BILINMIYOR").astype(str)

    return df
