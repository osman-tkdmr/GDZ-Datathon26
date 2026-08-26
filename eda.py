#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TRAFO BAZLI GÜNLÜK TÜKETİM TAHMİNİ — KAPSAMLI EDA SCRIPTİ
================================================================================
Yarışma: Trafo-gün bazında aktif enerji tüketimi (kWh) tahmini
Metrik : RMSLE = sqrt( mean( (log(1+tahmin) - log(1+gerçek))^2 ) )

Girdi dosyaları (varsayılan olarak --data-dir altında aranır):
    train.csv             -> id yok, tanim/guc/tarih/tuketim/lokasyon (Oca 2025 - Mar 2026)
    test.csv               -> id/tanim/guc/tarih/lokasyon           (Nis 2026 - Tem 2026)
    sample_submission.csv  -> id, tuketim (format örneği)

Kullanım:
    python eda.py --data-dir /path/to/csvler --output-dir ./eda_ciktilari

Çıktılar:
    <output-dir>/figures/*.png     -> tüm grafikler
    <output-dir>/tables/*.csv      -> ara özet tabloları
    <output-dir>/eda_raporu.md     -> bulguların ve mimari önerilerinin toplandığı rapor

Notlar:
    - statsmodels kuruluysa ACF/PACF ve STL ayrıştırması da üretilir; kurulu
      değilse bu adımlar otomatik atlanır (script yine de baştan sona çalışır).
    - Script her ana bölümü try/except ile sarar: bir bölüm veri kaynaklı bir
      sorunla (beklenmeyen kolon, format vb.) karşılaşırsa diğer bölümler
      etkilenmeden çalışmaya devam eder ve hata rapora not düşülür.
================================================================================
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ------------------------------------------------------------------------
# Türkiye resmi tatil takvimi (Ocak 2025 - Temmuz 2026; train+test aralığı)
# Kaynak: Diyanet/İçişleri Bakanlığı duyurularına dayanan 2025-2026 resmi
# tatil listeleri. Arefe günleri "yarım gün" olarak ayrıca işaretlenmiştir.
# ------------------------------------------------------------------------
HOLIDAYS_TR = {
    "2025-01-01": "Yılbaşı",
    "2025-03-29": "Ramazan Bayramı Arefesi (yarım gün)",
    "2025-03-30": "Ramazan Bayramı 1. Gün",
    "2025-03-31": "Ramazan Bayramı 2. Gün",
    "2025-04-01": "Ramazan Bayramı 3. Gün",
    "2025-04-23": "Ulusal Egemenlik ve Çocuk Bayramı",
    "2025-05-01": "Emek ve Dayanışma Günü",
    "2025-05-19": "Atatürk'ü Anma Gençlik ve Spor Bayramı",
    "2025-06-05": "Kurban Bayramı Arefesi (yarım gün)",
    "2025-06-06": "Kurban Bayramı 1. Gün",
    "2025-06-07": "Kurban Bayramı 2. Gün",
    "2025-06-08": "Kurban Bayramı 3. Gün",
    "2025-06-09": "Kurban Bayramı 4. Gün",
    "2025-07-15": "Demokrasi ve Millî Birlik Günü",
    "2025-08-30": "Zafer Bayramı",
    "2025-10-28": "Cumhuriyet Bayramı Arefesi (yarım gün)",
    "2025-10-29": "Cumhuriyet Bayramı",
    "2026-01-01": "Yılbaşı",
    "2026-03-19": "Ramazan Bayramı Arefesi (yarım gün)",
    "2026-03-20": "Ramazan Bayramı 1. Gün",
    "2026-03-21": "Ramazan Bayramı 2. Gün",
    "2026-03-22": "Ramazan Bayramı 3. Gün",
    "2026-04-23": "Ulusal Egemenlik ve Çocuk Bayramı",
    "2026-05-01": "Emek ve Dayanışma Günü",
    "2026-05-19": "Atatürk'ü Anma Gençlik ve Spor Bayramı",
    "2026-05-26": "Kurban Bayramı Arefesi (yarım gün)",
    "2026-05-27": "Kurban Bayramı 1. Gün",
    "2026-05-28": "Kurban Bayramı 2. Gün",
    "2026-05-29": "Kurban Bayramı 3. Gün",
    "2026-05-30": "Kurban Bayramı 4. Gün",
    "2026-07-15": "Demokrasi ve Millî Birlik Günü",
}

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})

REPORT = []  # markdown rapor satırları biriktirilir


def log(msg=""):
    print(msg)


def rep(md_text=""):
    REPORT.append(md_text)


def section(title, level=2):
    bar = "=" * 90
    log(f"\n{bar}\n{title}\n{bar}")
    rep(f"\n{'#' * level} {title}\n")


def savefig(fig, output_dir, name):
    path = Path(output_dir) / "figures" / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  [grafik kaydedildi] {path}")
    return path


def savetable(df, output_dir, name):
    path = Path(output_dir) / "tables" / f"{name}.csv"
    df.to_csv(path, index=True)
    log(f"  [tablo kaydedildi] {path}")
    return path


def safe_section(func):
    """Bir EDA bölümünü try/except ile sarar; hata olursa rapora not düşer."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log(f"  [!] '{func.__name__}' bölümünde hata oluştu, atlanıyor: {e}")
            rep(f"> ⚠️ **Not:** `{func.__name__}` bölümü çalıştırılırken hata alındı ve atlandı: `{e}`\n")
            return None
    return wrapper


# ============================================================================
# 0) VERİ YÜKLEME
# ============================================================================

def load_data(data_dir):
    section("0) VERİ YÜKLEME", level=1)
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sub_path = data_dir / "sample_submission.csv"

    for p in [train_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"'{p}' bulunamadı. --data-dir ile train.csv/test.csv dosyalarının "
                f"bulunduğu klasörü belirtin."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path) if sub_path.exists() else None

    for df, name in [(train, "train"), (test, "test")]:
        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
        else:
            log(f"  [!] UYARI: '{name}' içinde 'tarih' kolonu bulunamadı!")

    train = train.sort_values(["tanim", "tarih"]).reset_index(drop=True)
    test = test.sort_values(["tanim", "tarih"]).reset_index(drop=True)

    log(f"train.csv : {train.shape[0]:,} satır, {train.shape[1]} kolon")
    log(f"test.csv  : {test.shape[0]:,} satır, {test.shape[1]} kolon")
    if sample_sub is not None:
        log(f"sample_submission.csv : {sample_sub.shape[0]:,} satır, {sample_sub.shape[1]} kolon")

    rep(f"- **train.csv**: {train.shape[0]:,} satır × {train.shape[1]} kolon "
        f"({train['tarih'].min().date()} → {train['tarih'].max().date()})")
    rep(f"- **test.csv**: {test.shape[0]:,} satır × {test.shape[1]} kolon "
        f"({test['tarih'].min().date()} → {test['tarih'].max().date()})")

    return train, test, sample_sub


# ============================================================================
# 1) GENEL BAKIŞ (dtypes, eksik değer, duplike kontrolü)
# ============================================================================

@safe_section
def genel_bakis(df, name, output_dir):
    section(f"1) GENEL BAKIŞ — {name}")

    log(f"Boyut: {df.shape}")
    log("\nDtypes:")
    log(df.dtypes.to_string())

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    log(f"\nBellek kullanımı: {mem_mb:.2f} MB")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"eksik_adet": missing, "eksik_yuzde": missing_pct})
    missing_table = missing_table[missing_table["eksik_adet"] > 0].sort_values(
        "eksik_adet", ascending=False
    )
    if len(missing_table):
        log("\nEksik değerler:")
        log(missing_table.to_string())
    else:
        log("\nEksik değer yok.")
    savetable(missing_table if len(missing_table) else pd.DataFrame({"eksik_adet": missing}),
              output_dir, f"{name}_eksik_degerler")

    dup_rows = df.duplicated().sum()
    log(f"\nTam duplike satır sayısı: {dup_rows}")

    if {"tanim", "tarih"}.issubset(df.columns):
        dup_key = df.duplicated(subset=["tanim", "tarih"]).sum()
        log(f"(tanim, tarih) ikilisinde duplike satır sayısı: {dup_key}"
            f"{'  -> BEKLENMEDİK! Anahtar tekil olmalı.' if dup_key else '  -> OK, anahtar tekil.'}")
        rep(f"- **{name}**: (tanim, tarih) anahtarında **{dup_key}** duplike satır "
            f"{'bulundu ⚠️' if dup_key else 'yok ✅'}.")

    rep(f"- **{name}** eksik değer: " +
        (", ".join(f"`{c}`={v}" for c, v in missing[missing > 0].items()) if missing.sum() else "yok ✅"))

    return {"missing": missing_table, "dup_rows": dup_rows, "mem_mb": mem_mb}


# ============================================================================
# 2) HEDEF DEĞİŞKEN ANALİZİ (tuketim) — sadece train
# ============================================================================

@safe_section
def hedef_analizi(train, output_dir):
    section("2) HEDEF DEĞİŞKEN ANALİZİ — tuketim (kWh)")

    y = train["tuketim"]
    desc = y.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    log(desc.to_string())

    n_zero = (y == 0).sum()
    n_neg = (y < 0).sum()
    n_missing = y.isna().sum()
    log(f"\nSıfır tüketim satırı     : {n_zero:,} ({n_zero/len(y)*100:.3f}%)")
    log(f"Negatif tüketim satırı   : {n_neg:,} ({n_neg/len(y)*100:.3f}%)")
    log(f"Eksik (NaN) tüketim      : {n_missing:,}")

    y_pos = y.dropna()
    skew_raw = stats.skew(y_pos)
    kurt_raw = stats.kurtosis(y_pos)
    y_log = np.log1p(y_pos.clip(lower=0))
    skew_log = stats.skew(y_log)
    kurt_log = stats.kurtosis(y_log)
    log(f"\nÇarpıklık (skew)  ham={skew_raw:.3f}  |  log1p sonrası={skew_log:.3f}")
    log(f"Basıklık (kurtosis) ham={kurt_raw:.3f}  |  log1p sonrası={kurt_log:.3f}")

    rep(f"- Ortalama tüketim **{desc['mean']:.1f} kWh**, medyan **{desc['50%']:.1f} kWh**, "
        f"p99 **{desc['99%']:.1f} kWh**, maksimum **{desc['max']:.1f} kWh** — dağılım ciddi "
        f"biçimde **sağa çarpık** (ham çarpıklık={skew_raw:.2f}, log1p sonrası={skew_log:.2f}).")
    rep(f"- Sıfır tüketim: **{n_zero:,}** satır ({n_zero/len(y)*100:.3f}%), "
        f"negatif tüketim: **{n_neg:,}** satır, eksik: **{n_missing:,}** satır.")
    if skew_log < skew_raw / 2:
        rep("- `log1p` dönüşümü çarpıklığı belirgin biçimde azaltıyor → RMSLE metriğiyle uyumlu "
            "olarak **hedefi log1p ölçeğinde modellemek** (ve tahmin sonrası expm1 ile geri çevirmek) mantıklı.")

    # --- Grafik: ham vs log dağılım + boxplot + zaman içindeki toplam ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].hist(y_pos.clip(upper=y_pos.quantile(0.99)), bins=80, color="#3b6ea5")
    axes[0, 0].set_title("Tüketim Dağılımı (ham, p99'da kırpılmış)")
    axes[0, 0].set_xlabel("tuketim (kWh)")
    axes[0, 0].set_ylabel("frekans")

    axes[0, 1].hist(y_log, bins=80, color="#3ba55c")
    axes[0, 1].set_title("log1p(Tüketim) Dağılımı")
    axes[0, 1].set_xlabel("log1p(tuketim)")
    axes[0, 1].set_ylabel("frekans")

    axes[1, 0].boxplot(y_pos.clip(upper=y_pos.quantile(0.99)), vert=False)
    axes[1, 0].set_title("Tüketim Boxplot (p99'da kırpılmış)")
    axes[1, 0].set_xlabel("tuketim (kWh)")

    qq_data = stats.probplot(y_log, dist="norm")
    axes[1, 1].scatter(qq_data[0][0], qq_data[0][1], s=6, alpha=0.4, color="#c0392b")
    slope, intercept = qq_data[1][0], qq_data[1][1]
    x_line = np.array([qq_data[0][0].min(), qq_data[0][0].max()])
    axes[1, 1].plot(x_line, slope * x_line + intercept, color="black", lw=1)
    axes[1, 1].set_title("Q-Q Plot — log1p(Tüketim) vs Normal")
    axes[1, 1].set_xlabel("teorik kantiller")
    axes[1, 1].set_ylabel("örneklem kantilleri")

    fig.suptitle("Hedef Değişken (tuketim) Analizi", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, output_dir, "01_hedef_dagilim")

    extremes = pd.concat([
        train.nlargest(10, "tuketim")[["tanim", "tarih", "guc", "tuketim", "lokasyon"]].assign(uc="en_buyuk10"),
        train.nsmallest(10, "tuketim")[["tanim", "tarih", "guc", "tuketim", "lokasyon"]].assign(uc="en_kucuk10"),
    ])
    savetable(extremes, output_dir, "hedef_ekstremler")

    return {"desc": desc, "skew_raw": skew_raw, "skew_log": skew_log,
            "n_zero": n_zero, "n_neg": n_neg}


# ============================================================================
# 3) GÜÇ (guc) ANALİZİ
# ============================================================================

@safe_section
def guc_analizi(train, test, output_dir):
    section("3) KURULU GÜÇ (guc) ANALİZİ")

    desc_tr = train["guc"].describe()
    log("train guc istatistikleri:")
    log(desc_tr.to_string())

    # guc trafo bazında sabit mi? (fiziksel olarak sabit olması beklenir)
    guc_nunique = train.groupby("tanim")["guc"].nunique()
    n_degisken = (guc_nunique > 1).sum()
    log(f"\n'guc' değeri zaman içinde değişen trafo sayısı: {n_degisken} / {guc_nunique.shape[0]}")
    rep(f"- `guc` (kurulu güç), trafoların **{n_degisken}** tanesinde zaman içinde "
        f"değişkenlik gösteriyor (toplam {guc_nunique.shape[0]:,} trafo). "
        + ("Beklenmedik bir durum, veri kalitesi kontrolü önerilir." if n_degisken else
           "Beklendiği gibi her trafo için sabit ✅ — güvenle statik (zamandan bağımsız) bir özellik olarak kullanılabilir."))

    # log-log korelasyon guc vs tuketim
    merged = train[["tanim", "guc", "tuketim"]].dropna()
    log_guc = np.log1p(merged["guc"])
    log_tuk = np.log1p(merged["tuketim"].clip(lower=0))
    pear = np.corrcoef(log_guc, log_tuk)[0, 1]
    spear = stats.spearmanr(merged["guc"], merged["tuketim"]).correlation
    log(f"\nlog1p(guc) - log1p(tuketim) Pearson korelasyonu : {pear:.3f}")
    log(f"guc - tuketim Spearman korelasyonu               : {spear:.3f}")
    rep(f"- `guc` ile `tuketim` arasında log ölçekte Pearson r=**{pear:.3f}**, "
        f"Spearman ρ=**{spear:.3f}** — güç, tüketimin güçlü bir belirleyicisi "
        f"{'(çok güçlü ilişki, önemli bir özellik olacaktır)' if abs(pear) > 0.5 else '(orta düzey ilişki)'}.")

    # yük faktörü proxy = ortalama günlük tüketim / (guc * 24)
    trafo_ozet = train.groupby("tanim").agg(
        guc=("guc", "first"), ort_tuketim=("tuketim", "mean"), gun_sayisi=("tuketim", "count")
    )
    trafo_ozet["yuk_faktoru_proxy"] = trafo_ozet["ort_tuketim"] / (trafo_ozet["guc"] * 24)
    log("\nYük faktörü proxy (ort_gunluk_tuketim / (guc*24)) özet istatistikleri:")
    log(trafo_ozet["yuk_faktoru_proxy"].describe().to_string())
    savetable(trafo_ozet, output_dir, "trafo_bazinda_guc_ozet")

    # train vs test guc dağılımı (kapsanan trafo evreni farklı mı?)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].scatter(merged["guc"], merged["tuketim"], s=3, alpha=0.15, color="#2c3e50")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("guc (kVA, log)"); axes[0].set_ylabel("tuketim (kWh, log)")
    axes[0].set_title("guc vs tuketim (log-log)")

    sns.histplot(train["guc"], bins=60, color="#3b6ea5", ax=axes[1], stat="density", label="train", alpha=0.6)
    if "guc" in test.columns:
        sns.histplot(test["guc"], bins=60, color="#e67e22", ax=axes[1], stat="density", label="test", alpha=0.5)
    axes[1].set_title("guc Dağılımı: train vs test")
    axes[1].legend()

    axes[2].hist(trafo_ozet["yuk_faktoru_proxy"].clip(upper=trafo_ozet["yuk_faktoru_proxy"].quantile(0.99)),
                 bins=60, color="#8e44ad")
    axes[2].set_title("Yük Faktörü Proxy Dağılımı (trafo bazında)")
    axes[2].set_xlabel("ort_tuketim / (guc*24)")

    fig.tight_layout()
    savefig(fig, output_dir, "02_guc_analizi")

    ks_stat, ks_p = (None, None)
    if "guc" in test.columns:
        train_guc_per_trafo = train.groupby("tanim")["guc"].first()
        test_guc_per_trafo = test.groupby("tanim")["guc"].first()
        ks_stat, ks_p = ks_2samp(train_guc_per_trafo, test_guc_per_trafo)
        log(f"\nKS testi (trafo bazında guc, train vs test): statistic={ks_stat:.4f}, p={ks_p:.4g}")
        rep(f"- Trafo bazında `guc` dağılımı için train/test KS testi: statistic=**{ks_stat:.4f}**, "
            f"p=**{ks_p:.4g}** — " +
            ("dağılımlar istatistiksel olarak anlamlı derecede farklı ⚠️ (kovaryans kayması riski)."
             if ks_p is not None and ks_p < 0.05 else "dağılımlar arasında anlamlı bir fark tespit edilmedi ✅."))

    return {"corr_pearson_log": pear, "corr_spearman": spear, "n_guc_degisken": n_degisken,
            "ks_stat": ks_stat, "ks_p": ks_p}


# ============================================================================
# 4) LOKASYON ANALİZİ
# ============================================================================

def _parse_lokasyon(df):
    parts = df["lokasyon"].fillna("").str.split(">")
    out = df.copy()
    out["il"] = parts.str[0].str.strip().replace("", np.nan)
    out["bolge"] = parts.str[1].str.strip() if parts.str.len().max() and parts.str.len().max() >= 2 else np.nan
    out["ilce"] = parts.str[2].str.strip() if parts.str.len().max() and parts.str.len().max() >= 3 else np.nan
    out["lokasyon_jenerik"] = out["lokasyon"].str.strip().eq("GEDİZ EDAŞ") | (parts.str.len() <= 1)
    return out


@safe_section
def lokasyon_analizi(train, test, output_dir):
    section("4) LOKASYON HİYERARŞİSİ ANALİZİ")

    train_l = _parse_lokasyon(train)
    test_l = _parse_lokasyon(test) if "lokasyon" in test.columns else None

    n_jenerik = train_l["lokasyon_jenerik"].sum()
    log(f"Jenerik ('GEDİZ EDAŞ' / seviyesiz) lokasyon satırı: {n_jenerik:,} "
        f"({n_jenerik/len(train_l)*100:.2f}%)")

    n_il = train_l["il"].nunique(dropna=True)
    n_bolge = train_l["bolge"].nunique(dropna=True) if "bolge" in train_l else 0
    n_ilce = train_l["ilce"].nunique(dropna=True) if "ilce" in train_l else 0
    log(f"Benzersiz İL sayısı   : {n_il}")
    log(f"Benzersiz BÖLGE sayısı: {n_bolge}")
    log(f"Benzersiz İLÇE sayısı : {n_ilce}")

    rep(f"- Lokasyon hiyerarşisinde **{n_il}** il, **{n_bolge}** bölge, **{n_ilce}** ilçe seviyesi "
        f"bulunuyor; satırların **%{n_jenerik/len(train_l)*100:.1f}**'i jenerik \"GEDİZ EDAŞ\" "
        f"etiketiyle (hiyerarşi bilgisi eksik) geliyor.")

    # lokasyon bazında ortalama tüketim (en yüksek/düşük 15)
    lok_ozet = train_l.groupby("lokasyon").agg(
        ort_tuketim=("tuketim", "mean"), trafo_sayisi=("tanim", "nunique"), satir_sayisi=("tuketim", "size")
    ).sort_values("ort_tuketim", ascending=False)
    savetable(lok_ozet, output_dir, "lokasyon_bazinda_ozet")

    # train / test lokasyon evreni farkı
    n_yeni_lokasyon = 0
    if test_l is not None:
        train_lok_set = set(train_l["lokasyon"].dropna().unique())
        test_lok_set = set(test_l["lokasyon"].dropna().unique())
        n_yeni_lokasyon = len(test_lok_set - train_lok_set)
        log(f"\ntest.csv'de train.csv'de hiç görülmemiş lokasyon sayısı: {n_yeni_lokasyon} / {len(test_lok_set)}")
        rep(f"- test.csv'deki **{len(test_lok_set)}** benzersiz lokasyondan **{n_yeni_lokasyon}** tanesi "
            f"train.csv'de hiç görülmemiş " +
            ("— bu lokasyonlar için hiyerarşik (il/bölge düzeyinde) fallback özellik/encoding önerilir."
             if n_yeni_lokasyon else "✅ — tüm test lokasyonları train'de temsil ediliyor."))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    top_il = train_l.groupby("il")["tuketim"].mean().sort_values(ascending=False).head(15)
    top_il.plot(kind="barh", ax=axes[0], color="#2980b9")
    axes[0].invert_yaxis()
    axes[0].set_title("İl Bazında Ortalama Tüketim (Top 15)")
    axes[0].set_xlabel("ortalama tuketim (kWh)")

    il_counts = train_l["il"].value_counts().head(15)
    il_counts.plot(kind="barh", ax=axes[1], color="#c0392b")
    axes[1].invert_yaxis()
    axes[1].set_title("İl Bazında Gözlem (Satır) Sayısı (Top 15)")
    axes[1].set_xlabel("satır sayısı")

    fig.tight_layout()
    savefig(fig, output_dir, "03_lokasyon_analizi")

    return {"n_il": n_il, "n_bolge": n_bolge, "n_ilce": n_ilce,
            "n_jenerik": n_jenerik, "n_yeni_lokasyon": n_yeni_lokasyon}


# ============================================================================
# 5) TRAFO (tanim) YAPISI ANALİZİ — panel veri boyutları
# ============================================================================

@safe_section
def trafo_yapisi_analizi(train, test, output_dir):
    section("5) TRAFO (tanim) YAPISI ANALİZİ — PANEL VERİ BOYUTLARI")

    train_trafolar = set(train["tanim"].unique())
    test_trafolar = set(test["tanim"].unique())
    ortak = train_trafolar & test_trafolar
    sadece_test = test_trafolar - train_trafolar
    sadece_train = train_trafolar - test_trafolar

    log(f"train.csv'deki benzersiz trafo sayısı : {len(train_trafolar):,}")
    log(f"test.csv'deki benzersiz trafo sayısı  : {len(test_trafolar):,}")
    log(f"İki kümede de olan (ortak) trafo       : {len(ortak):,} "
        f"({len(ortak)/len(test_trafolar)*100:.2f}% of test)")
    log(f"Sadece test.csv'de olan (cold-start)    : {len(sadece_test):,}")
    log(f"Sadece train.csv'de olan                : {len(sadece_train):,}")

    rep(f"- train'de **{len(train_trafolar):,}**, test'te **{len(test_trafolar):,}** benzersiz trafo var. "
        f"Test trafolarının **%{len(ortak)/len(test_trafolar)*100:.1f}**'i train'de de geçmiş "
        f"(ortak: {len(ortak):,} trafo).")
    if sadece_test:
        rep(f"- ⚠️ **{len(sadece_test):,} trafo yalnızca test.csv'de** bulunuyor (train'de geçmiş verisi yok) "
            f"→ **cold-start problemi**: bu trafolar için lag/rolling gibi geçmişe dayalı özellikler "
            f"üretilemez; guc + lokasyon bazlı fallback/benzer-trafo stratejisi gerekir.")
    else:
        rep("- ✅ test.csv'deki tüm trafolar train.csv'de de mevcut — cold-start problemi yok, "
            "her trafo için geçmiş zaman serisi kullanılabilir.")

    # trafo başına gözlem (gün) sayısı dağılımı
    gun_sayisi = train.groupby("tanim").size()
    log("\nTrafo başına satır (gün) sayısı dağılımı (train):")
    log(gun_sayisi.describe().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(gun_sayisi, bins=60, color="#16a085")
    axes[0].set_title("Trafo Başına Gözlem (Gün) Sayısı Dağılımı — train")
    axes[0].set_xlabel("gün sayısı"); axes[0].set_ylabel("trafo sayısı")

    trafo_ort = train.groupby("tanim")["tuketim"].mean()
    axes[1].hist(np.log1p(trafo_ort), bins=60, color="#d35400")
    axes[1].set_title("Trafo Başına Ortalama Tüketim Dağılımı (log1p)")
    axes[1].set_xlabel("log1p(ortalama tuketim)")

    fig.tight_layout()
    savefig(fig, output_dir, "04_trafo_yapisi")

    return {"n_train_trafo": len(train_trafolar), "n_test_trafo": len(test_trafolar),
            "n_ortak": len(ortak), "n_sadece_test": len(sadece_test)}


# ============================================================================
# 6) TARİH KAPSAMI VE VERİ TAMLIĞI (missing days per trafo)
# ============================================================================

@safe_section
def tarih_kapsam_analizi(train, test, output_dir):
    section("6) TARİH KAPSAMI VE GÜNLÜK VERİ TAMLIĞI")

    tr_min, tr_max = train["tarih"].min(), train["tarih"].max()
    te_min, te_max = test["tarih"].min(), test["tarih"].max()
    gap_days = (te_min - tr_max).days
    log(f"train tarih aralığı: {tr_min.date()} → {tr_max.date()}  ({(tr_max-tr_min).days+1} gün)")
    log(f"test  tarih aralığı: {te_min.date()} → {te_max.date()}  ({(te_max-te_min).days+1} gün)")
    log(f"train sonu ile test başlangıcı arasındaki boşluk: {gap_days} gün")

    rep(f"- train: **{tr_min.date()} → {tr_max.date()}**, test: **{te_min.date()} → {te_max.date()}**. "
        f"train sonu ile test başlangıcı arasında **{gap_days} gün** boşluk var "
        + ("(model, test dönemine kadar {} gün ileriye tahmin yapmak zorunda — bu ufuk uzunluğu "
           "feature/validation tasarımında dikkate alınmalı).".format(gap_days) if gap_days > 1
           else "(test, train'in hemen ardından başlıyor)."))

    # trafo bazında günlük veri tamlığı: beklenen gün sayısı vs gerçek satır sayısı
    trafo_range = train.groupby("tanim")["tarih"].agg(["min", "max", "count"])
    trafo_range["beklenen_gun"] = (trafo_range["max"] - trafo_range["min"]).dt.days + 1
    trafo_range["tamlik_orani"] = (trafo_range["count"] / trafo_range["beklenen_gun"]).clip(upper=1.0)
    ortalama_tamlik = trafo_range["tamlik_orani"].mean()
    n_eksiksiz = (trafo_range["tamlik_orani"] >= 0.999).sum()
    log(f"\nOrtalama trafo veri tamlık oranı: {ortalama_tamlik*100:.2f}%")
    log(f"Tamamen eksiksiz (>=%99.9) trafo sayısı: {n_eksiksiz:,} / {len(trafo_range):,}")
    savetable(trafo_range.sort_values("tamlik_orani"), output_dir, "trafo_veri_tamligi")

    rep(f"- Trafo bazında ortalama **veri tamlık oranı %{ortalama_tamlik*100:.1f}**; "
        f"**{n_eksiksiz:,}/{len(trafo_range):,}** trafo kendi min-max tarih aralığında hiç gün atlamıyor. "
        + ("Eksik günler için lag/rolling özellik hesaplarken tarih bazlı reindex + uygun imputation "
           "(ör. forward-fill değil, aynı haftanın günü ortalaması) önerilir." if ortalama_tamlik < 0.999 else
           "Seri neredeyse tam günlük ve düzenli — sekans modelleri için uygun bir yapı."))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(trafo_range["tamlik_orani"], bins=50, color="#2980b9")
    ax.set_title("Trafo Bazında Veri Tamlık Oranı Dağılımı")
    ax.set_xlabel("tamlik_orani (gerçek gün / beklenen gün)")
    ax.set_ylabel("trafo sayısı")
    fig.tight_layout()
    savefig(fig, output_dir, "05_veri_tamligi")

    return {"gap_days": gap_days, "ortalama_tamlik": ortalama_tamlik, "n_eksiksiz": n_eksiksiz}


# ============================================================================
# 7) ZAMAN SERİSİ DESENLERİ (trend, haftalık/aylık mevsimsellik, tatil etkisi)
# ============================================================================

@safe_section
def zaman_serisi_desenleri(train, output_dir):
    section("7) ZAMAN SERİSİ DESENLERİ — TREND, MEVSİMSELLİK, TATİL ETKİSİ")

    gunluk = train.groupby("tarih")["tuketim"].agg(["sum", "mean", "count"]).reset_index()
    gunluk["haftanin_gunu"] = gunluk["tarih"].dt.day_name()
    gunluk["hafta_sonu"] = gunluk["tarih"].dt.dayofweek >= 5
    gunluk["ay"] = gunluk["tarih"].dt.to_period("M").astype(str)
    gunluk["tatil_mi"] = gunluk["tarih"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS_TR.keys())

    # --- Toplam/aritmetik ortalama günlük tüketim zaman serisi ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(gunluk["tarih"], gunluk["sum"], color="#2c3e50", lw=0.9)
    axes[0].set_title("Tüm Trafolar — Günlük Toplam Tüketim")
    axes[0].set_ylabel("toplam tuketim (kWh)")
    for hdate in HOLIDAYS_TR:
        hd = pd.Timestamp(hdate)
        if gunluk["tarih"].min() <= hd <= gunluk["tarih"].max():
            axes[0].axvline(hd, color="red", alpha=0.15, lw=1)

    axes[1].plot(gunluk["tarih"], gunluk["mean"], color="#8e44ad", lw=0.9)
    axes[1].set_title("Trafo Başına Ortalama Günlük Tüketim (kırmızı çizgiler = resmi tatiller)")
    axes[1].set_ylabel("ortalama tuketim (kWh)")
    fig.tight_layout()
    savefig(fig, output_dir, "06_zaman_serisi_toplam")

    # --- Haftanın günü etkisi ---
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_tr = {"Monday": "Pzt", "Tuesday": "Sal", "Wednesday": "Çar", "Thursday": "Per",
                  "Friday": "Cum", "Saturday": "Cmt", "Sunday": "Paz"}
    train_wd = train.copy()
    train_wd["haftanin_gunu"] = pd.Categorical(train_wd["tarih"].dt.day_name(),
                                                categories=weekday_order, ordered=True)
    wd_means = train_wd.groupby("haftanin_gunu", observed=True)["tuketim"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar([weekday_tr[d] for d in wd_means.index], wd_means.values, color="#27ae60")
    axes[0].set_title("Haftanın Gününe Göre Ortalama Tüketim")
    axes[0].set_ylabel("ortalama tuketim (kWh)")

    ay_means = gunluk.groupby("ay")["mean"].mean()
    axes[1].plot(range(len(ay_means)), ay_means.values, marker="o", color="#c0392b")
    axes[1].set_xticks(range(len(ay_means)))
    axes[1].set_xticklabels(ay_means.index, rotation=60, ha="right")
    axes[1].set_title("Aylık Ortalama Tüketim (trafo başına)")
    fig.tight_layout()
    savefig(fig, output_dir, "07_haftalik_aylik_mevsimsellik")

    hafta_ici_ort = wd_means[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean()
    hafta_sonu_ort = wd_means[["Saturday", "Sunday"]].mean()
    fark_pct = (hafta_sonu_ort - hafta_ici_ort) / hafta_ici_ort * 100
    log(f"\nHafta içi ortalama tüketim : {hafta_ici_ort:.1f} kWh")
    log(f"Hafta sonu ortalama tüketim: {hafta_sonu_ort:.1f} kWh  (fark: {fark_pct:+.1f}%)")

    # --- Tatil etkisi ---
    tatil_ort = gunluk.loc[gunluk["tatil_mi"], "mean"].mean()
    normal_ort = gunluk.loc[~gunluk["tatil_mi"], "mean"].mean()
    tatil_fark_pct = (tatil_ort - normal_ort) / normal_ort * 100
    log(f"Resmi tatil günleri ortalama tüketim   : {tatil_ort:.1f} kWh")
    log(f"Normal günler ortalama tüketim         : {normal_ort:.1f} kWh  (fark: {tatil_fark_pct:+.1f}%)")

    rep(f"- **Haftalık mevsimsellik** belirgin: hafta sonu ortalama tüketim, hafta içine göre "
        f"**%{fark_pct:.1f}** farklı → `haftanin_gunu`/`hafta_sonu` özellikleri önemli.")
    rep(f"- **Resmi tatil etkisi**: tatil günlerinde ortalama tüketim normal günlere göre "
        f"**%{tatil_fark_pct:.1f}** farklı → bir `tatil_mi` (ve varsa `arefe_mi`) flag özelliği eklenmesi önerilir "
        f"(script içindeki `HOLIDAYS_TR` takvimi Oca 2025–Tem 2026 aralığını kapsar).")

    acf_info = {}
    if STATSMODELS_AVAILABLE:
        seri = gunluk.set_index("tarih")["mean"].asfreq("D").interpolate()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(seri, lags=60, ax=axes[0])
        axes[0].set_title("ACF — Trafo Başına Ortalama Günlük Tüketim")
        plot_pacf(seri, lags=60, ax=axes[1], method="ywm")
        axes[1].set_title("PACF — Trafo Başına Ortalama Günlük Tüketim")
        fig.tight_layout()
        savefig(fig, output_dir, "08_acf_pacf")

        try:
            period = 7
            stl = STL(seri, period=period, robust=True).fit()
            fig = stl.plot()
            fig.set_size_inches(13, 8)
            fig.suptitle("STL Ayrıştırması (haftalık periyot) — Toplam Seri", y=1.01)
            savefig(fig, output_dir, "09_stl_ayristirma")
        except Exception as e:
            log(f"  [!] STL ayrıştırması atlandı: {e}")

        rep("- ACF/PACF ve STL ayrıştırma grafikleri, güçlü **7 günlük (haftalık) periyodiklik** ve "
            "uzun vadeli bir **trend/mevsimsellik** bileşeni olup olmadığını görselleştirir; "
            "`figures/08_acf_pacf.png` ve `09_stl_ayristirma.png` dosyalarına bakınız.")
    else:
        rep("- `statsmodels` kurulu olmadığı için ACF/PACF ve STL ayrıştırması atlandı "
            "(`pip install statsmodels` ile etkinleştirilebilir).")

    # --- Örnek trafo serileri (küçük/orta/büyük guc) ---
    guc_bins = pd.qcut(train.groupby("tanim")["guc"].first(), q=3, labels=["kucuk", "orta", "buyuk"])
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for i, grup in enumerate(["kucuk", "orta", "buyuk"]):
        adaylar = guc_bins[guc_bins == grup].index
        secim = np.random.default_rng(42).choice(adaylar, size=min(3, len(adaylar)), replace=False)
        for tanim_id in secim:
            seri = train[train["tanim"] == tanim_id].set_index("tarih")["tuketim"]
            axes[i].plot(seri.index, seri.values, lw=0.8, alpha=0.85, label=f"tanim={tanim_id}")
        axes[i].set_title(f"Örnek Trafo Serileri — guc grubu: {grup}")
        axes[i].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    savefig(fig, output_dir, "10_ornek_trafo_serileri")

    return {"hafta_sonu_fark_pct": fark_pct, "tatil_fark_pct": tatil_fark_pct}


# ============================================================================
# 8) ID FORMAT DOĞRULAMA (test.csv: id = tanim_tarih)
# ============================================================================

@safe_section
def id_format_dogrulama(test, sample_sub, output_dir):
    section("8) TEST.CSV ID FORMAT DOĞRULAMASI")

    if "id" not in test.columns:
        log("test.csv içinde 'id' kolonu bulunamadı, adım atlanıyor.")
        return None

    parsed = test["id"].astype(str).str.rsplit("_", n=1, expand=True)
    parsed.columns = ["tanim_from_id", "tarih_from_id"]
    parsed["tarih_from_id"] = pd.to_datetime(parsed["tarih_from_id"], errors="coerce")

    tanim_match = (parsed["tanim_from_id"].astype(str) == test["tanim"].astype(str)).mean()
    tarih_match = (parsed["tarih_from_id"] == test["tarih"]).mean()
    log(f"id'den ayrıştırılan tanim, 'tanim' kolonuyla eşleşme oranı: {tanim_match*100:.2f}%")
    log(f"id'den ayrıştırılan tarih, 'tarih' kolonuyla eşleşme oranı: {tarih_match*100:.2f}%")

    rep(f"- test.csv `id` alanı `tanim_tarih` formatına **%{min(tanim_match, tarih_match)*100:.1f}** oranında uyuyor "
        + ("✅." if min(tanim_match, tarih_match) > 0.999 else
           "⚠️ — bazı satırlarda format tutarsızlığı olabilir, submission oluşturmadan önce doğrulanmalı."))

    if sample_sub is not None:
        id_match = set(test["id"]) == set(sample_sub["id"])
        log(f"test.csv id kümesi == sample_submission.csv id kümesi: {id_match}")
        rep(f"- test.csv id kümesi, sample_submission.csv id kümesiyle "
            f"{'birebir eşleşiyor ✅.' if id_match else 'BİREBİR EŞLEŞMİYOR ⚠️ — submission formatını kontrol edin.'}")

    return {"tanim_match": tanim_match, "tarih_match": tarih_match}


# ============================================================================
# 9) RMSLE-ODAKLI DİAGNOSTİKLER
# ============================================================================

@safe_section
def rmsle_diagnostikleri(train, output_dir):
    section("9) RMSLE METRİĞİNE ÖZEL DİAGNOSTİKLER")

    trafo_stats = train.groupby("tanim")["tuketim"].agg(["mean", "std"]).dropna()
    trafo_stats = trafo_stats[trafo_stats["mean"] > 0]
    log_mean = np.log1p(trafo_stats["mean"])
    log_std = np.log1p(trafo_stats["std"].clip(lower=0))
    corr = np.corrcoef(log_mean, log_std)[0, 1]
    log(f"log1p(ortalama) - log1p(std) korelasyonu (trafo bazında): {corr:.3f}")
    rep(f"- Trafo bazında ortalama tüketim ile standart sapma arasında log ölçekte "
        f"korelasyon=**{corr:.3f}** — " +
        ("değişkenlik seviyeyle birlikte artıyor (heteroskedastisite); ham ölçekte MSE/RMSE "
         "optimize etmek büyük trafoları domine ettirir. RMSLE zaten göreli hataya odaklandığından "
         "**log1p(tuketim) hedefiyle RMSE optimize etmek** RMSLE'yi optimize etmeye pratikte çok yakın "
         "sonuç verir." if corr > 0.3 else
         "seviyeyle çok güçlü bir ilişki gözlenmedi, yine de log dönüşümü RMSLE hizalaması için önerilir."))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(trafo_stats["mean"], trafo_stats["std"], s=8, alpha=0.35, color="#16a085")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trafo ortalama tüketim (log)")
    ax.set_ylabel("trafo tüketim std sapması (log)")
    ax.set_title("Heteroskedastisite: Ortalama vs Std (trafo bazında)")
    fig.tight_layout()
    savefig(fig, output_dir, "11_heteroskedastisite")

    # basit örnekle RMSLE'nin ölçek-bağımsız ceza mantığını göster
    ornekler = pd.DataFrame({
        "gercek": [100, 100, 100000, 100000],
        "tahmin": [200, 50, 200000, 50000],
        "aciklama": ["küçük trafo x2 fazla", "küçük trafo x2 az",
                     "büyük trafo x2 fazla", "büyük trafo x2 az"],
    })
    ornekler["rmsle_katkisi"] = (np.log1p(ornekler["tahmin"]) - np.log1p(ornekler["gercek"])) ** 2
    log("\nRMSLE'nin ölçekten bağımsız ceza mantığı — örnek satırlar:")
    log(ornekler.to_string(index=False))
    savetable(ornekler, output_dir, "rmsle_ornek_ceza_tablosu")
    rep("- Örnek hesap tablosu (`tables/rmsle_ornek_ceza_tablosu.csv`) 2 katlık göreli hatanın, "
        "100 kWh'lik küçük bir trafoda da 100.000 kWh'lik dev bir trafoda da **aynı RMSLE katkısını** "
        "verdiğini gösteriyor — bu da modelin **küçük trafolarda göreli hatayı büyük trafolar kadar "
        "ciddiye alması** gerektiği anlamına gelir (ör. MAPE'ye yakın davranan bir kayıp fonksiyonu).")

    return {"corr_mean_std": corr}


# ============================================================================
# 10) KORELASYON MATRİSİ (sayısal + türetilmiş takvim özellikleri)
# ============================================================================

@safe_section
def korelasyon_analizi(train, output_dir):
    section("10) KORELASYON MATRİSİ (sayısal + takvim özellikleri)")

    df = train.copy()
    df["log_tuketim"] = np.log1p(df["tuketim"].clip(lower=0))
    df["log_guc"] = np.log1p(df["guc"])
    df["gun_hafta"] = df["tarih"].dt.dayofweek
    df["ay"] = df["tarih"].dt.month
    df["yil_gunu"] = df["tarih"].dt.dayofyear
    df["hafta_sonu"] = (df["gun_hafta"] >= 5).astype(int)
    df["tatil_mi"] = df["tarih"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS_TR.keys()).astype(int)

    cols = ["log_tuketim", "log_guc", "gun_hafta", "ay", "yil_gunu", "hafta_sonu", "tatil_mi"]
    corr = df[cols].corr()
    log(corr.round(3).to_string())
    savetable(corr, output_dir, "korelasyon_matrisi")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Korelasyon Matrisi — log_tuketim ve Takvim Özellikleri")
    fig.tight_layout()
    savefig(fig, output_dir, "12_korelasyon_matrisi")

    return {"corr": corr}


# ============================================================================
# 11) ÖZET RAPOR + MİMARİ ÖNERİLERİ
# ============================================================================

def mimari_onerileri_ekle(train, test):
    n_trafo = train["tanim"].nunique()
    n_gun_train = train["tarih"].nunique()
    section("11) BULGULARIN MİMARİ TASARIMA YANSIMASI — ÖNERİLER", level=1)

    rep(f"""
Veri, **~{n_trafo:,} trafo × ~{n_gun_train:,} gün**lük bir **panel (uzun format) zaman serisi**
yapısında: her trafonun kendi seviyesi (guc ile ölçeklenen), kendi mevsimselliği ve
lokasyon/güç bazında paylaşılan ortak desenleri var. Bu yapı mimari seçimini doğrudan
şekillendiriyor:

### A) Güçlü ve hızlı bir başlangıç noktası: Gradient Boosting (LightGBM/XGBoost/CatBoost)
- **Hedef dönüşümü**: `log1p(tuketim)` üzerinde eğitip tahminleri `expm1` ile geri çevirin;
  RMSLE metriğiyle doğrudan hizalı ve büyük/küçük trafo dengesizliğini azaltır.
  Negatif tahminleri 0'a clip'leyin (metrik zaten bunu varsayıyor).
- **Özellik mühendisliği**:
  - Trafo bazında **lag** (1, 7, 14, 28 gün) ve **rolling mean/std** (7, 14, 30 gün) özellikleri
    — panelin veri tamlığı yüksek olduğu için (bkz. Bölüm 6) bu özellikler güvenilir olacaktır.
  - **Takvim özellikleri**: haftanın günü, hafta sonu flag'i, ay, yılın günü, `tatil_mi`/`arefe_mi`
    (Bölüm 7'de haftalık ve tatil etkisi doğrulandı).
  - **Statik özellikler**: `guc` (log ölçekte), `lokasyon` hiyerarşisinin her seviyesi
    (il/bölge/ilçe) için frekans veya hedef (target) encoding, `tanim` için de target encoding
    (K-fold içi sızıntısız hesaplanmalı).
  - Yük faktörü proxy'si (`tuketim/(guc*24)`) gibi türetilmiş oran özellikleri.
- **Validasyon**: Rastgele K-fold **kullanmayın** — zaman sızıntısı yaratır. Bunun yerine
  **zaman bazlı (walk-forward) CV**: train'in son bloğunu (örn. son 60-90 gün) validasyon
  olarak ayırıp, test'e olan **{'{gap} günlük'.format(gap=(test['tarih'].min()-train['tarih'].max()).days) if len(test) else ''}
  boşluğu** validasyon kurgusunda da simüle edin.

### B) Global (panel) derin öğrenme modelleri
Trafo sayısı yeterince yüksekse (bkz. Bölüm 5), tüm serileri **tek bir global model** olarak
eğitmek klasik tek-seri modellerden (ARIMA/ETS) genelde daha güçlü sonuç verir:
- **N-BEATS / N-HiTS / TFT (Temporal Fusion Transformer) / DeepAR**: `tanim` ve `lokasyon`
  hiyerarşisini **statik kategorik kovaryans** (embedding), `guc`'u statik sayısal kovaryans,
  takvim özelliklerini **bilinen gelecek kovaryans** olarak veren mimariler bu problem için
  doğal bir uyum sağlar.
- **LSTM/GRU + Entity Embedding**: `tanim` ve `lokasyon` için embedding katmanları, girdi olarak
  geçmiş pencere (ör. son 28-60 gün) + takvim özellikleri.
- Cold-start trafolar varsa (Bölüm 5), bu mimarilerde embedding'i **guc + lokasyon benzerliğine**
  dayalı bir "ortalama profil" ile initialize etmek veya ayrı bir "yeni trafo" fallback modeli
  kurmak gerekir.

### C) Hibrit / ensemble yaklaşımı
Pratikte en iyi sonuç genelde **GBM (tablo özellikleriyle) + global DL sekans modelinin ensemble'ı**
ile alınır: GBM güçlü statik/etkileşim özelliklerini yakalar, DL modeli seri içi kalıpları
(otokorelasyon, uzun vadeli trend) daha iyi öğrenir. Bölüm 7'deki ACF/PACF ve STL çıktıları,
haftalık periyodikliğin güçlü olduğu doğrulanırsa, bu iki ailenin tamamlayıcı olacağını gösterir.

### D) Veri kalitesi & sızıntı kontrol listesi (script çıktısına göre doldurulmalı)
- [ ] (tanim, tarih) anahtarı her iki dosyada da tekil mi? (Bölüm 1)
- [ ] `guc` her trafo için zaman içinde gerçekten sabit mi? (Bölüm 3)
- [ ] test'te train'de hiç görülmemiş trafo/lokasyon var mı, varsa fallback stratejisi tanımlı mı?
      (Bölüm 4-5)
- [ ] Trafo bazında eksik gün oranı feature penceresini (ör. 28 günlük rolling) etkileyecek kadar
      yüksek mi? (Bölüm 6)
- [ ] Zaman bazlı CV, train sonu-test başı arasındaki boşluğu (Bölüm 6) doğru simüle ediyor mu?
""")


@safe_section
def raporu_yaz(output_dir, meta):
    out_path = Path(output_dir) / "eda_raporu.md"
    header = f"""# Trafo Bazlı Günlük Tüketim Tahmini — EDA Raporu

*Otomatik oluşturulma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

Bu rapor `eda.py` scripti tarafından, sağlanan `train.csv` / `test.csv` dosyaları
üzerinden otomatik olarak üretilmiştir. Tüm sayısal bulgular çalıştırma anındaki
gerçek veriye dayanır; grafikler `figures/`, ara tablolar `tables/` klasöründedir.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(REPORT))
    log(f"\n[rapor kaydedildi] {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trafo tüketim tahmini yarışması için kapsamlı EDA scripti")
    parser.add_argument("--data-dir", default=".", help="train.csv/test.csv/sample_submission.csv klasörü")
    parser.add_argument("--output-dir", default="./eda_ciktilari", help="Grafik/tablo/rapor çıktı klasörü")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    log("#" * 90)
    log("TRAFO BAZLI GÜNLÜK TÜKETİM TAHMİNİ — KAPSAMLI EDA")
    log("#" * 90)
    log(f"statsmodels mevcut: {STATSMODELS_AVAILABLE}")

    train, test, sample_sub = load_data(args.data_dir)

    genel_bakis(train, "train", output_dir)
    genel_bakis(test, "test", output_dir)

    hedef_analizi(train, output_dir)
    guc_analizi(train, test, output_dir)
    lokasyon_analizi(train, test, output_dir)
    trafo_yapisi_analizi(train, test, output_dir)
    tarih_kapsam_analizi(train, test, output_dir)
    zaman_serisi_desenleri(train, output_dir)
    id_format_dogrulama(test, sample_sub, output_dir)
    rmsle_diagnostikleri(train, output_dir)
    korelasyon_analizi(train, output_dir)
    mimari_onerileri_ekle(train, test)

    raporu_yaz(output_dir, meta={})

    log("\n" + "=" * 90)
    log(f"TAMAMLANDI. Tüm çıktılar: {output_dir.resolve()}")
    log("=" * 90)


if __name__ == "__main__":
    main()