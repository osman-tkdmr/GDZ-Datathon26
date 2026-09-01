from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
ILCE_NUFUS = DATA_DIR / "izmir_manisa_ilce_nufuslari.xlsx"
WEATHER_CACHE_PATH = DATA_DIR / "open_meteo_cache.csv"
AREA_PATH = DATA_DIR / "izmir_manisa_yuzolcumleri.xlsx"
COORD_CACHE_PATH = DATA_DIR / "ilce_koordinatlari.json"

TARGET = "tuketim"
ID_COL = "id"
ENTITY_COL = "tanim"
DATE_COL = "tarih"

RANDOM_STATE = 42
VALID_DAYS = 122
USE_GPU = True
LGB_USE_GPU = False


CATEGORICAL_FEATURES = [
    ENTITY_COL,
    "lokasyon",
    "il",
    "bolge",
    "ilce",
]

NUMERIC_FEATURES = [
    'guc',
    'year',
    'month',
    'day',
    'dayofweek',
    'dayofyear',
    'weekofyear',
    'is_weekend',
    'is_month_start',
    'is_month_end',
    'guc_log',
    'guc_percentile',
    'guc_percentile_bin',
    'guc_ilce_share',
    'nufus23',
    'nufus24',
    'nufus25',
    'ilce_trafo_sayisi',
    'trafo_per_nufus23',
    'trafo_per_nufus24',
    'trafo_per_nufus25',
    'yuzolcumu_km2',
    'nufus_density23',
    'nufus_density24',
    'nufus_density25',
    'is_yilbasi',
    'is_egemenlik_cocuk',
    'is_emek_gunu',
    'is_genclik_spor',
    'is_zafer_gunu',
    'is_cumhuriyet_gunu',
    'is_demokrasi_gunu',
    'is_ramadan',
    'is_kurban',
    'is_any_holiday',
    'is_midterm_break',
    'is_semester_break',
    'is_summer_break',
    'is_fixed_holiday',
    'days_to_holiday',
    'is_any_holiday_prev1',
    'is_any_holiday_next1',
    'is_ramadan_prev1',
    'is_ramadan_next1',
    'is_kurban_prev1',
    'is_kurban_next1',
    'is_summer_break_prev1',
    'is_summer_break_next1',
    'is_semester_break_prev1',
    'is_semester_break_next1',
    'is_midterm_break_prev1',
    'is_midterm_break_next1',
    'is_holiday_prev1',
    'is_holiday_next1',
    'gunes_toplam_ghi_whm2',
    'gun_uzunlugu_saat',
    'ogle_zenit_derece',
    'has_solar'
]
