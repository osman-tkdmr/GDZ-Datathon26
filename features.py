import numpy as np
import pandas as pd
from config import ILCE_NUFUS

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

import pandas as pd


def ilce_features(df: pd.DataFrame):
  # Excel sayfalarını oku
  izmir_nufus = pd.read_excel(ILCE_NUFUS, sheet_name=0).rename(
      columns={"İlçe": "ilce", "2023": "nufus23", "2024": "nufus24", "2025": "nufus25"}
  )
  manisa_nufus = pd.read_excel(ILCE_NUFUS, sheet_name=1).rename(
      columns={"İlçe": "ilce", "2023": "nufus23", "2024": "nufus24", "2025": "nufus25"}
  )

  # İlgli şehirleri ayır ve verileri birleştir
  izmir_df = df[df["il"] == "İZMİR"].merge(izmir_nufus, on="ilce", how="left")
  manisa_df = df[df["il"] == "MANİSA"].merge(manisa_nufus, on="ilce", how="left")
  diger_df = df[~df["il"].isin(["İZMİR", "MANİSA"])]

  # Hepsini tekrar birleştir
  return pd.concat([izmir_df, manisa_df, diger_df], ignore_index=True)
import pandas as pd


def ilce_trafo_aded(df: pd.DataFrame):
  # Her satır için ilgili ilçenin benzersiz trafo sayısını hesaplar ve yeni sütuna yazar
  df["ilce_trafo_sayisi"] = df.groupby("ilce")["tanim"].transform("nunique")
  return df

def nufus_bol_trafo_aded(df:pd.DataFrame):
    df["trafo_per_nufus23"] = df["nufus23"]/df["ilce_trafo_sayisi"]
    df["trafo_per_nufus24"] = df["nufus24"]/df["ilce_trafo_sayisi"]
    df["trafo_per_nufus25"] = df["nufus25"]/df["ilce_trafo_sayisi"]
    return df



def prepare_features(df):
    df = add_calendar_features(df)
    df = add_numeric_features(df)
    df = ilce_features(df)
    df = ilce_trafo_aded(df)
    df = nufus_bol_trafo_aded(df)


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

