import numpy as np
import pandas as pd

import json
import time
from pathlib import Path

import holidays as tr_holidays_lib

from geopy.geocoders import Nominatim

from urllib.parse import urlencode
from urllib.request import urlopen, Request

import pvlib.location as pvlocation



from config import ILCE_NUFUS, AREA_PATH, ENTITY_COL, DATE_COL, COORD_CACHE_PATH, WEATHER_CACHE_PATH

WEATHER_BASELINE_HDD = 18.0
WEATHER_BASELINE_CDD = 22.0
OPEN_METEO_MODEL = "era5_land"

SOLAR_FEATURE_COLS = ["gunes_toplam_ghi_whm2", "gun_uzunlugu_saat", "ogle_zenit_derece", "has_solar"]
WEATHER_FEATURE_COLS = ["temp_mean", "temp_max", "humidity_mean", "humidity_max", "hdd_18", "cdd_22"]

def turkce_to_ingilizce(text):
    tr_chars = "ğĞüÜşŞıİöÖçÇ"
    en_chars = "gGuUsSiIoOcC"
    ceviri_tablosu = str.maketrans(tr_chars, en_chars)
    return text.translate(ceviri_tablosu)

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

    # Percentile: gunluk tekrarlar dagilimi agirliklandirmasin diye
    # once benzersiz transformer guc tablosu uzerinden hesaplanir.
    entity_power = (
        df[["il", "ilce", ENTITY_COL, "guc"]]
        .drop_duplicates(["il", "ilce", ENTITY_COL])
        .copy()
    )
    entity_power["guc_percentile"] = (
        entity_power["guc"].rank(method="average", pct=True)
        .astype("float32")
    )
    entity_power["guc_percentile_bin"] = (
        np.floor(entity_power["guc_percentile"] * 10)
        .clip(0, 9)
        .astype("int8")
    )
    df = df.drop(columns=["guc_percentile", "guc_percentile_bin"], errors="ignore")
    df = df.merge(
        entity_power[[ENTITY_COL, "guc_percentile", "guc_percentile_bin"]],
        on=ENTITY_COL,
        how="left",
    )

    # Her transformatorun ilcesindeki toplam kurulu guc icindeki pay.
    # Gunluk tekrar eden satirlari toplamdan cikarmak icin entity-guc tablosu
    # kullanilir; boylece tarih kapsamindaki satir sayisi payi bozmaz.
    entity_power = (
        df[["il", "ilce", ENTITY_COL, "guc"]]
        .drop_duplicates(["il", "ilce", ENTITY_COL])
    )
    district_total = (
        entity_power
        .groupby(["il", "ilce"])["guc"]
        .sum()
        .rename("ilce_total_guc")
        .reset_index()
    )
    df = df.merge(district_total, on=["il", "ilce"], how="left")
    df["guc_ilce_share"] = np.where(
        df["ilce_total_guc"] > 0,
        df["guc"] / df["ilce_total_guc"],
        0.0,
    ).astype("float32")
    df.drop(columns=["ilce_total_guc"], inplace=True)
    return df


def ilce_features(df: pd.DataFrame):
    df = df.copy()
    df["ilce"] = df["ilce"].apply(lambda x: turkce_to_ingilizce(x).upper())

    izmir_nufus = pd.read_excel(ILCE_NUFUS, sheet_name=0).rename(
        columns={"İlçe": "ilce", "2023": "nufus23", "2024": "nufus24", "2025": "nufus25"}
    )
    izmir_nufus["ilce"] = izmir_nufus["ilce"].apply(lambda x: turkce_to_ingilizce(x).upper())
    
    manisa_nufus = pd.read_excel(ILCE_NUFUS, sheet_name=1).rename(
        columns={"İlçe": "ilce", "2023": "nufus23", "2024": "nufus24", "2025": "nufus25"}
    )
    manisa_nufus["ilce"] = manisa_nufus["ilce"].apply(lambda x: turkce_to_ingilizce(x).upper())

    izmir_df = df[df["il"] == "IZMIR"].merge(izmir_nufus, on="ilce", how="left")
    manisa_df = df[df["il"] == "MANISA"].merge(manisa_nufus, on="ilce", how="left")
    diger_df = df[~df["il"].isin(["IZMIR", "MANISA"])]
    df = pd.concat([izmir_df, manisa_df, diger_df], ignore_index=True)

    df["ilce_trafo_sayisi"] = df.groupby("ilce")["tanim"].transform("nunique")

    df["trafo_per_nufus23"] = df["nufus23"]/df["ilce_trafo_sayisi"]
    df["trafo_per_nufus24"] = df["nufus24"]/df["ilce_trafo_sayisi"]
    df["trafo_per_nufus25"] = df["nufus25"]/df["ilce_trafo_sayisi"]

    area_frames = []
    province_by_sheet = {"iy": "IZMIR", "my": "MANISA", 0: "IZMIR", 1: "MANISA"}
    for sheet in [0, 1]:
        area = pd.read_excel(AREA_PATH, sheet_name=sheet).rename(
            columns={"İlçe": "ilce", "Yüz Ölçümü (km²)": "yuzolcumu_km2"}
        )
        area["ilce"] = area["ilce"].astype(str).str.strip().str.upper().apply(turkce_to_ingilizce)
        area["il"] = province_by_sheet[sheet]
        area_frames.append(area)
    area_df = pd.concat(area_frames, ignore_index=True).drop_duplicates(["il", "ilce"])
    df = df.merge(area_df, on=["il", "ilce"], how="left")

    for y in [23, 24, 25]:
        df[f"trafo_per_nufus{y}"] = np.where(
            df["ilce_trafo_sayisi"] > 0,
            df[f"nufus{y}"] / df["ilce_trafo_sayisi"],
            0.0,
        )
        df[f"nufus_density{y}"] = np.where(
            df["yuzolcumu_km2"] > 0,
            df[f"nufus{y}"] / df["yuzolcumu_km2"],
            0.0,
        )

    return df

def _mark_intervals(dates, intervals):
    result = pd.Series(False, index=dates.index)
    norm = dates.dt.normalize()
    for start_date, end_date in intervals:
        result |= norm.between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
    return result.astype("int8")


def _meb_break_intervals(years):
    # MEB'in resmi 2023-24, 2024-25, 2025-26 ve 2026-27 takvimleri.
    # Yaz tatili, bir egitim-ogretim yilinin ders bitimi ile sonraki
    # egitim-ogretim yilinin baslangici arasidir.
    midterm = [
        ("2023-11-13", "2023-11-17"), ("2024-04-08", "2024-04-12"),
        ("2024-11-11", "2024-11-15"), ("2025-03-31", "2025-04-04"),
        ("2025-11-10", "2025-11-14"), ("2026-03-16", "2026-03-20"),
        ("2026-11-16", "2026-11-20"), ("2027-03-08", "2027-03-12"),
    ]
    semester = [
        ("2024-01-22", "2024-02-02"),
        ("2025-01-20", "2025-01-31"),
        ("2026-01-19", "2026-01-30"),
        ("2027-01-25", "2027-02-05"),
    ]
    summer = [
        ("2024-06-15", "2024-09-08"),
        ("2025-06-21", "2025-09-07"),
        ("2026-06-27", "2026-09-13"),
        ("2027-06-26", "2027-09-12"),
    ]
    return midterm, semester, summer

def add_holiday_features(df):
    df = df.copy()
    dt_norm = df[DATE_COL].dt.normalize()

    years = sorted(dt_norm.dt.year.unique().tolist())
    years_query = years + [years[-1] + 1]
    tr_holidays = tr_holidays_lib.Turkey(years=years_query)
    holiday_items = list(tr_holidays.items())
    holiday_dates_all = pd.to_datetime([d for d, _ in holiday_items])

    def _dates_matching(substr):
        return pd.to_datetime([d for d, name in holiday_items if substr in name])

    fixed_map = {
        "is_yilbasi": "Yılbaşı",
        "is_egemenlik_cocuk": "Ulusal Egemenlik",
        "is_emek_gunu": "Emek ve Dayanışma",
        "is_genclik_spor": "Atatürk'ü Anma",
        "is_zafer_gunu": "Zafer Bayramı",
        "is_cumhuriyet_gunu": "Cumhuriyet Bayramı",
        "is_demokrasi_gunu": "Demokrasi ve Millî Birlik",
    }
    for col, substr in fixed_map.items():
        df[col] = dt_norm.isin(_dates_matching(substr)).astype("int8")

    # holidays kutuphanesi bayram gunlerini verir; burada Ramazan ayinin
    # tamamini temsil eden ay araliklari ayrica tanimlaniyor.
    ramadan_intervals = [
        ("2023-03-23", "2023-04-20"),
        ("2024-03-11", "2024-04-09"),
        ("2025-03-01", "2025-03-30"),
        ("2026-02-19", "2026-03-19"),
        ("2027-02-08", "2027-03-08"),
    ]
    df["is_ramadan"] = _mark_intervals(dt_norm, ramadan_intervals)
    df["is_kurban"] = dt_norm.isin(_dates_matching("Kurban")).astype("int8")
    df["is_any_holiday"] = dt_norm.isin(holiday_dates_all).astype("int8")

    midterm, semester, summer = _meb_break_intervals(years)
    df["is_midterm_break"] = _mark_intervals(dt_norm, midterm)
    df["is_semester_break"] = _mark_intervals(dt_norm, semester)
    df["is_summer_break"] = _mark_intervals(dt_norm, summer)

    fixed_cols = list(fixed_map.keys())
    df["is_fixed_holiday"] = df[fixed_cols].any(axis=1).astype("int8")

    unique_dates = pd.Series(dt_norm.unique())
    holiday_arr = np.sort(pd.DatetimeIndex(holiday_dates_all).unique().values.astype("datetime64[D]"))

    def _min_days(date):
        diffs = np.abs((holiday_arr - np.datetime64(date, "D")).astype(int))
        return int(diffs.min()) if len(diffs) else 365

    days_map = pd.Series([_min_days(d) for d in unique_dates], index=unique_dates)
    df["days_to_holiday"] = dt_norm.map(days_map).astype("int16")

    # Tatil oncesi/sonrasi: entity bazinda shift.
    # Önce her entity-tarih icin gunluk flag olusturuluyor, sonra +1/-1 gun kaydiriliyor.
    base_cols = [
        "is_any_holiday", "is_ramadan", "is_kurban",
        "is_summer_break", "is_semester_break", "is_midterm_break",
    ]
    entity_dates = (
        df[[ENTITY_COL, DATE_COL] + base_cols]
        .assign(_date=lambda x: x[DATE_COL].dt.normalize())
        .groupby([ENTITY_COL, "_date"], as_index=False)[base_cols]
        .max()
        .sort_values([ENTITY_COL, "_date"])
    )
    for col in base_cols:
        prev_map = entity_dates.set_index([ENTITY_COL, "_date"])[col]
        prev_key = pd.MultiIndex.from_arrays([
            df[ENTITY_COL].to_numpy(),
            (df[DATE_COL].dt.normalize() - pd.Timedelta(days=1)).to_numpy(),
        ])
        next_key = pd.MultiIndex.from_arrays([
            df[ENTITY_COL].to_numpy(),
            (df[DATE_COL].dt.normalize() + pd.Timedelta(days=1)).to_numpy(),
        ])
        df[f"{col}_prev1"] = prev_map.reindex(prev_key).fillna(0).to_numpy().astype("int8")
        df[f"{col}_next1"] = prev_map.reindex(next_key).fillna(0).to_numpy().astype("int8")

    # Genel tatil oncesi/sonrasi aliaslari: FEATURES listesindeki isimlerle uyumlu.
    df["is_holiday_prev1"] = df["is_any_holiday_prev1"].astype("int8")
    df["is_holiday_next1"] = df["is_any_holiday_next1"].astype("int8")

    return df

def _geocode_ilceler(ilce_adlari, cache_path=COORD_CACHE_PATH, sleep_seconds=1.1):
    """Ilce adlarini (enlem, boylam, rakim) ile eslestirir. Onbellekte olmayanlar
    icin Nominatim'e (rate-limit'e uyarak) gider, sonucu diske yazar. Diskte
    zaten TUM ilceler varsa hic internete gitmez."""
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    eksikler = [x for x in ilce_adlari if x not in cache]
    if eksikler:
        geolocator = Nominatim(user_agent="gdz_datathon_solar_features")
        for ilce in eksikler:
            try:
                konum = geolocator.geocode(f"{ilce}, Turkey", timeout=10)
                if konum is None:
                    print(f"[gunes] Konum bulunamadi: {ilce} -- atlaniyor.")
                    continue
                try:
                    alt = pvlocation.lookup_altitude(konum.latitude, konum.longitude)
                except Exception:
                    alt = 100.0
                cache[ilce] = {"lat": konum.latitude, "lon": konum.longitude, "alt": float(alt)}
                print(f"[gunes] Geocode edildi: {ilce} -> {cache[ilce]}")
            except Exception as exc:
                print(f"[gunes] Geocode hatasi ({ilce}): {exc}")
            time.sleep(sleep_seconds)  # Nominatim kullanim kurallarina uymak icin

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    return cache


def _daily_solar_features_for_one_location(lat, lon, alt, date_index, tz="Europe/Istanbul"):
    """Tek bir (lat, lon) icin, verilen tum gunlerde GUNLUK ozet gunes
    ozelliklerini TEK pvlib cagrisiyla (saatlik cozunurlukte) hesaplar."""
    loc = pvlocation.Location(lat, lon, tz=tz, altitude=alt)

    start = pd.Timestamp(date_index.min()) - pd.Timedelta(hours=1)
    end = pd.Timestamp(date_index.max()) + pd.Timedelta(days=1)
    times = pd.date_range(start=start, end=end, freq="h", tz=tz)

    clearsky = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)

    df = pd.DataFrame({
        "ghi": clearsky["ghi"].values,
        "zenith": solpos["apparent_zenith"].values,
    }, index=times)
    df["gun"] = df.index.tz_localize(None).normalize()
    df["gunduz_mi"] = df["zenith"] < 90  # gunes ufkun ustunde

    gunluk = df.groupby("gun").agg(
        gunes_toplam_ghi_whm2=("ghi", "sum"),          # gunluk toplam clearsky isinim (Wh/m2)
        gun_uzunlugu_saat=("gunduz_mi", "sum"),         # gunduz olan saat sayisi (kaba yaklasim)
        ogle_zenit_derece=("zenith", "min"),            # gunun en dusuk zeniti = ogle konumu
    )
    return gunluk


def build_solar_feature_table(ilce_adlari, date_index, tz="Europe/Istanbul", cache_path=COORD_CACHE_PATH):
    """Verilen tum ilceler icin, verilen tum tarih araliginda gunluk gunes
    ozelliklerini iceren bir DataFrame dondurur."""
    ilce_adlari = sorted(set(ilce_adlari) - {"BILINMIYOR"})
    coords = _geocode_ilceler(ilce_adlari, cache_path=cache_path)

    tablolar = []
    for ilce in ilce_adlari:
        if ilce not in coords:
            continue
        c = coords[ilce]
        gunluk = _daily_solar_features_for_one_location(c["lat"], c["lon"], c["alt"], date_index, tz=tz)
        gunluk = gunluk.reset_index().rename(columns={"gun": "tarih"})
        gunluk["ilce"] = ilce
        tablolar.append(gunluk)

    if not tablolar:
        return pd.DataFrame(columns=["ilce", "tarih"] + SOLAR_FEATURE_COLS[:-1])

    return pd.concat(tablolar, ignore_index=True)


def add_solar_features(df, date_col=DATE_COL, ilce_col="ilce", cache_path=COORD_CACHE_PATH):
    """df'e (train ya da test) gunes ozelliklerini merge eder. Koordinati
    bulunamayan / BILINMIYOR ilcesi icin NaN kalir (has_solar=0)."""
    solar_table = build_solar_feature_table(df[ilce_col].unique(), df[date_col], cache_path=cache_path)
    solar_table["tarih"] = pd.to_datetime(solar_table["tarih"])

    df = df.copy()
    df["_tarih_norm"] = df[date_col].dt.normalize()
    df = df.merge(
        solar_table, left_on=[ilce_col, "_tarih_norm"], right_on=["ilce", "tarih"],
        how="left", suffixes=("", "_solar"),
    )
    df = df.drop(columns=["_tarih_norm", "tarih_solar", "ilce_solar"], errors="ignore")
    df["has_solar"] = df["gunes_toplam_ghi_whm2"].notna().astype("int8")
    return df

def _load_coord_cache(cache_path=COORD_CACHE_PATH):
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_location_coords(ilce, il, cache_path=COORD_CACHE_PATH):
    """Ilce|il anahtariyla koordinat alir. Eski cache formatini da destekler."""
    cache = _load_coord_cache(cache_path)
    key = f"{il}|{ilce}"
    if key in cache:
        return cache[key]
    if ilce in cache:
        return cache[ilce]
    # Mevcut _geocode_ilceler fonksiyonu ilce bazli cache doldurabilir.
    coords = _geocode_ilceler([ilce], cache_path=cache_path)
    return coords.get(ilce)


def _open_meteo_daily(ilce, il, start_date, end_date, cache_path=WEATHER_CACHE_PATH):
    coord = _get_location_coords(ilce, il)
    if coord is None:
        return pd.DataFrame()

    params = {
        "latitude": coord["lat"],
        "longitude": coord["lon"],
        "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_max",
        ]),
        "timezone": "Europe/Istanbul",
        "temperature_unit": "celsius",
        "models": OPEN_METEO_MODEL,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "GDZ-Datathon/1.0"})
    with urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    daily = payload.get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame()

    out = pd.DataFrame({
        "il": il,
        "ilce": ilce,
        "tarih": pd.to_datetime(daily["time"]),
        "temp_mean": daily.get("temperature_2m_mean"),
        "temp_max": daily.get("temperature_2m_max"),
        "humidity_mean": daily.get("relative_humidity_2m_mean"),
        "humidity_max": daily.get("relative_humidity_2m_max"),
    })
    out["hdd_18"] = np.maximum(WEATHER_BASELINE_HDD - out["temp_mean"], 0.0)
    out["cdd_22"] = np.maximum(out["temp_mean"] - WEATHER_BASELINE_CDD, 0.0)
    return out

def add_weather_features(df, cache_path=WEATHER_CACHE_PATH):
    """Her ilce icin Open-Meteo archive'dan gunluk hava verisi ceker.

    Cikti: ortalama/max sicaklik, ortalama/max nem, HDD18 ve CDD22.
    """
    df = df.copy()
    if df.empty:
        return df

    start_date = df[DATE_COL].min().normalize()
    end_date = df[DATE_COL].max().normalize()

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.DataFrame()
    if cache_path.exists():
        try:
            cache = pd.read_csv(cache_path)
            cache["tarih"] = pd.to_datetime(cache["tarih"])
        except Exception:
            cache = pd.DataFrame()

    pieces = []
    keys = df[["il", "ilce"]].drop_duplicates().itertuples(index=False)

    for il, ilce in keys:
        key = (il, ilce)
        cached = cache[
            (cache["il"] == il) &
            (cache["ilce"] == ilce) &
            (cache["tarih"] >= start_date) &
            (cache["tarih"] <= end_date)
        ] if not cache.empty else pd.DataFrame()

        expected = (end_date - start_date).days + 1
        if len(cached) >= expected:
            pieces.append(cached)
            continue

        try:
            print(f"[weather] Open-Meteo: {ilce} / {il}")
            weather = _open_meteo_daily(ilce, il, start_date, end_date)
            if not weather.empty:
                pieces.append(weather)
        except Exception as exc:
            print(f"[weather] {ilce} / {il} alinamadi: {exc}")

    if pieces:
        new_weather = pd.concat(pieces, ignore_index=True)
        cache = pd.concat([cache, new_weather], ignore_index=True) if not cache.empty else new_weather
        cache = cache.drop_duplicates(["il", "ilce", "tarih"], keep="last")
        try:
            cache.to_csv(cache_path, index=False)
        except Exception as exc:
            print(f"[weather] Cache parquet yazilamadi: {exc}")

    weather = cache[
        (cache["tarih"] >= start_date) & (cache["tarih"] <= end_date)
    ].copy() if not cache.empty else pd.DataFrame()

    if weather.empty:
        # Hava verisi tamamen yoksa model inputunda NaN birakma.
        for col in WEATHER_FEATURE_COLS:
            df[col] = 0.0
        return df

    df = df.merge(weather, on=["il", "ilce", DATE_COL], how="left")

    # Eksik gunleri ilce medyani -> global medyan -> 0 sirasi ile doldur.
    for col in WEATHER_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df.groupby(["il", "ilce"])[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)

    return df

def prepare_features(df):
    df = add_calendar_features(df)
    df = add_numeric_features(df)
    df = ilce_features(df)
    df = add_holiday_features(df)
    df = add_solar_features(df)
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

