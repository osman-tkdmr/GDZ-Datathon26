# ⚡ GDZ Datathon 26: Trafo Tüketim Tahmini — Ensemble Mimarisi

> **Üretim-hazır güç tüketim tahmini sistemi**: Gradient Boosting ensemble'i, Optuna hiperparametre optimizasyonu ve zamansal validasyonla 5,344 trafo üzerinde RMSLE minimize edilir.

## 🎯 Sistem Özeti

Bu proje, **1.2M eğitim örneği** ve **5,344 transformatör** içeren ölçeklenebilir bir panel veri tahmini sistemidir:

| Metrik | Değer |
|--------|-------|
| **Eğitim Süresi** | 2025-01-01 → 2026-03-31 (16 ay) |
| **Test Süresi** | 2026-04-01 → 2026-07-31 (4 ay) |
| **Trafo Sayısı** | 5,344 (eğitim) / 7,036 (test) |
| **Veri Tamlığı** | %95.3 (hiç duplike yok) |
| **Coğrafi Kapsam** | 2 il, 3 bölge, 47 ilçe |
| **Mimari** | **3-Model Ensemble** (LightGBM + CatBoost + XGBoost) |

### 🔑 Temel Özellikler

- ✅ **Multi-Model Ensemble**: Her modeli Optuna ile ayrı ayrı tunelendi, ağırlıkları RMSLE'yi minimize edecek şekilde optimize edildi
- ✅ **Temporal (Zamansal) Validasyon**: Zaman sızıntısını önlemek için walk-forward CV (son 122 gün = validation)
- ✅ **Otomatik GPU Fallback**: LightGBM ve CatBoost CPU'ya otomatik düşer (GPU kurulum sorunu varsa)
- ✅ **Kapsamlı Özellik Mühendisliği**: Hava durumu, güneş ışınları, nüfus, alan yoğunluğu, tatil takvimi
- ✅ **Veri Kalitesi Doğrulaması**: Cold-start trafolar tespiti, eksik değer imputation, kategori alignment

## 📋 Gereksinimler

- Python `>= 3.12`
- [`uv`](https://docs.astral.sh/uv/) paket yöneticisi (veya pip)

**Bağımlılıklar** `pyproject.toml` dosyasında tanımlanıdır ve Optuna, CatBoost, XGBoost, PyTorch, statsmodels, holidays, pvlib, geopy dahil tüm heavy dependencies'i içerir.

## 🏗️ Mimari: 3-Model Ensemble

Sistem, **gradient boosting** modellerinin güçlü yönlerini birleştirir:

```
┌─────────────────────────────────────────────────────────────────┐
│  VERİ HAZIRLIĞI: load_data() → add_location_hierarchy()         │
│                  prepare_features() → align_categories()        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
      ┌───────────────────────┬────────────────────────┐
      ↓                       ↓                        ↓
 ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
 │  LightGBM   │      │   CatBoost   │      │    XGBoost      │
 │             │      │              │      │                 │
 │ • Hızlı     │      │ • Native     │      │ • Kategorik     │
 │ • GBDT      │      │   Kategorik  │      │   native support│
 │ • GPU opt.  │      │ • GPU opt.   │      │ • Paralel       │
 └─────────────┘      └──────────────┘      └─────────────────┘
      ↓                       ↓                        ↓
 ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
 │ Optuna Tune │      │ Optuna Tune  │      │  Optuna Tune    │
 │(n_trials=50)|      │ (n_trials=50)│      │ (n_trials=50)   │
 └─────────────┘      └──────────────┘      └─────────────────┘
      ↓                       ↓                        ↓
 ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
 │  Validation │      │ Validation   │      │  Validation     │
 │  Pred (val) │      │  Pred (val)  │      │  Pred (val)     │
 └─────────────┘      └──────────────┘      └─────────────────┘
      ↓                       ↓                        ↓
      └───────────────────────┬────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │ ENSEMBLE AĞIRLIKLARI OPTİMİZE │
              │ minimize RMSLE(weights·preds) │
              └───────────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │ FINAL BLEND & SUBMISSION      │
              │ submission.csv (test tahmin)  │
              └───────────────────────────────┘
```

**İş Akışı:**
1. **Veri Yükleme**: train.csv / test.csv → pandas
2. **Lokasyon Hiyerarşisi**: `lokasyon` → `il`, `bölge`, `ilçe`
3. **Özellik Mühendisliği**: Takvim, sayısal, kategorik özellikler
4. **Optuna Tuning** (Her Model): Walk-forward CV ile hyperparametre optimize
5. **Validation Tahminleri**: En iyi hyperparameterlerle son 122 gündü validation seti üzerinde tahmin
6. **Ağırlık Optimizasyonu**: 3 modelin tahminlerinin RMSLE'yi minimize eden ağırlıklarını hesapla
7. **Sonuç**: Test tahminlerini blend et, `submission.csv` yaz

## 💾 Kurulum

Projeyi klonladıktan sonra proje klasöründe çalıştırın:

```bash
uv sync
```

Bu komut `.venv` sanal ortamını oluşturur veya günceller ve `pyproject.toml` içindeki bağımlılıkları `uv.lock` dosyasındaki sabit sürümlerle kurar. Ortamı ayrıca etkinleştirmek gerekmez; komutları `uv run` ile çalıştırabilirsiniz.

## 📊 Veri

Aşağıdaki dosyalar `data/` klasöründe bulunmalıdır:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

### Veri Kalitesi Özeti

| Özellik | Değer |
|---------|-------|
| **Duplike Satırlar** | 0 ✅ |
| **Eksik Değerler** | 0 ✅ |
| **Veri Tamlık Oranı** | %95.3 |
| **Sıfır Tüketim Oranı** | %4.7 |
| **Negatif Tüketim** | 0 ✅ |

### Temel Kolonlar

| Kolon | Açıklama |
| --- | --- |
| `id` | Tahmin kaydının kimliği (`tanim_tarih` formatı) |
| `tarih` | Gözlem tarihi (YYYY-MM-DD) |
| `tanim` | Trafo kimliği/tanımı |
| `lokasyon` | `il>bolge>ilce` biçimindeki konum bilgisi |
| `guc` | Trafo gücü (kVA) |
| `tuketim` | Eğitim verisindeki hedef kolon (kWh) |

**Notlar:**
- `tuketim` yalnızca eğitim verisinde bulunur
- `lokasyon` alanı `>` karakteriyle ayrıştırılarak `il`, `bolge` ve `ilce` özellikleri oluşturulur
- Test setinde 2,024 trafo (**%28.7**) train'de görülmüş değildir (cold-start problemi)

## 🔧 Model ve Özellik Mühendisliği

### Özelliklerin Yapısı

Sistem **90+** özellikten oluşan kapsamlı bir özellik seti kullanır:

#### 📅 **Takvim Özellikleri** (~10 özellik)
- **Temel**: yıl, ay, gün, haftanın günü
- **Periyodik**: yılın günü, haftanın numarası
- **Göstergeler**: hafta sonu flag, ay başı/sonu flag
- **Tatil**: Türkiye resmi tatilleri (holidays kütüphanesi)

#### 💪 **Güç & Yükleme Özellikleri** (~10 özellik)
- `guc`: Trafo kurulu gücü (kVA)
- `guc_log`: Log ölçekte dönüştürülen güç
- `guc_percentile`: İlçe içinde güç dağılımındaki yüzdelik
- `guc_percentile_bin`: Güç kategorileri (beş seviye)
- `guc_ilce_share`: İlçedeki trafolara göre güç oranı

#### 🌍 **Coğrafi & Nüfus Özellikleri** (~15 özellik)
- **Hiyerarşi**: İl, bölge, ilçe (kategorik)
- **Nüfus**: 2023, 2024, 2025 yıl nüfusu
- **Yoğunluk**: İlçe nüfus yoğunluğu (nüfus/km²)
- **Alan**: İlçe yüzölçümü (km²)
- **Trafo Yoğunluğu**: İlçe başına trafo sayısı

#### ☀️ **Hava Durumu & Güneş** (~12 özellik)
Open Meteo API'sinden alınan gerçek veriler:
- **Sıcaklık**: Ortalama, maksimum, derece-günler (HDD/CDD)
- **Nem**: Ortalama, maksimum
- **Güneş**: Toplam global ışınım (GHI), gün uzunluğu, öğle zenit açısı
- **Bulutluluk**: Bulutluluk yüzdesi

#### 🔗 **Kategorik Özellikler** (Entity-Level)
- `tanim` (Trafo ID): Target encoding ile sayısallaştırılmış
- `lokasyon`: Hiyerarşiye göre katman-katman kategoriler

### Model Parametreleri

#### LightGBM
```python
{
    'learning_rate': [0.001, 0.01, 0.1],
    'num_leaves': [32, 64, 128, 256],
    'max_depth': [5, 8, 12],
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.6, 0.8, 1.0],
    'lambda_l1': [0, 0.1, 1.0],
    'lambda_l2': [0, 0.1, 1.0],
}
```
**GPU Desteği**: Nvidia CUDA destekli GPU varsa otomatik `device_type='gpu'` kullanılır.

#### CatBoost
```python
{
    'learning_rate': [0.001, 0.01, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 10],
    'subsample': [0.5, 0.8, 1.0],
    'random_strength': [0, 1],
}
```
**Kategorik Desteği**: Native kategorik desteğiyle kategorik özellikleri doğrudan işler.

#### XGBoost
```python
{
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_lambda': [0.1, 1, 10],
}
```
**Kategorik Desteği**: `enable_categorical=True` ile kategorik özellikleri native olarak işler.

### Transformasyon ve Normalleştirme

- **Hedef Dönüşümü**: `tuketim → log1p(tuketim)` 
  - RMSLE metriğiyle tam hizalanmış
  - Tahmin sonrası: `expm1(pred_log) → clip(0, ∞)`
- **Güç Dönüşümü**: `guc → log1p(guc)` (sağa çarpık dağılım)
- **Kategorik Özellikler**: Label encoding (sınıf sayısı yüksek değil)

## ✅ Validasyon & Hiperparametre Optimizasyonu

### Zamansal Validasyon (Temporal Split)

Rastgele K-Fold yerine zamansal ayırma kullanılır. Eğitim verisinin son 122 günü validation seti, önceki tarihler ise eğitim seti olarak ayrılır:

```python
Train: 2025-01-01 → 2026-01-08    (1,090 gün)
Valid: 2026-01-09 → 2026-04-30    (122 gün) ← Zaman sızıntısı yok ✅
```

### RMSLE Metriği

Validation hedefi ve model hedefi `log1p` dönüşümüyle hizalanır. Tahminler `expm1` ile gerçek ölçeğe çevrilir ve negatif değerler sıfıra kırpılır; RMSLE bu gerçek ölçekli tahminlerle hesaplanır:

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(1+\hat{y}_i) - \log(1+y_i))^2}$$

### Optuna Hiperparametre Optimizasyonu

Her model için ayrı bir Optuna çalışması yürütülür:

- **LightGBM**: 50 trial, learning_rate / num_leaves / depth optimize
- **CatBoost**: 50 trial, depth / l2_leaf_reg / subsample optimize  
- **XGBoost**: 50 trial, learning_rate / max_depth / lambda optimize

En iyi hyperparametreler seçildikten sonra validation seti üzerinde tahmin üretilir ve sonra ağırlık optimizasyonunda kullanılır.

## 🚀 Çalıştırma

### Eğitim ve Submission

Validation ve submission üretmek için:

```bash
uv run train_ensemble.py
```

**İş akışı:**
1. Eğitim verisi yüklenir ve özellikler hazırlanır
2. LightGBM, CatBoost ve XGBoost için ayrı Optuna çalışmaları başlatılır (50 trial her biri)
3. Her model için en iyi hyperparametreler seçilir
4. En iyi parametrelerle validation tahmini üretilir
5. Validation seti üzerinde 3 modelin tahminleri ağırlıklandırılır (RMSLE minimize)
6. Full eğitim seti üzerinde tüm modeller yeniden eğitilir
7. Test tahminleri üretilir ve ağırlıkla blend edilir
8. `outputs/submission.csv` dosyası yazılır

Başarılı çalıştırmanın sonunda validation skoru terminale yazdırılır ve şu dosyalar oluşturulur:

```text
outputs/validation_predictions.csv    # Validation set tahminleri
outputs/submission.csv                # Test set tahminleri
```

### Keşifsel Veri Analizi (EDA)

EDA betiği train/test verilerini inceler; veri kalitesi, hedef dağılımı, güç, lokasyon, trafo panel yapısı, tarih kapsamı, mevsimsellik, RMSLE davranışı ve korelasyonlar için grafik ve özet tablolar üretir.

```bash
uv run eda.py --data-dir data --output-dir eda_ciktilari
```

`--data-dir` içinde `train.csv`, `test.csv` ve varsa `sample_submission.csv` aranır. Çıktılar:

```text
eda_ciktilari/eda_raporu.md  # Otomatik oluşturulan bulgu ve öneri raporu
eda_ciktilari/figures/       # PNG grafikler (12 farklı analiz)
eda_ciktilari/tables/        # CSV özet tabloları
```

**Grafiklerin Özeti:**
- `01_hedef_dagilim.png` - Tüketim dağılımı (ham vs. log ölçek)
- `02_guc_analizi.png` - Trafo gücü analizi ve tüketimle korelasyon
- `03_lokasyon_analizi.png` - Il/ilçe bazında tüketim profilleri
- `04_trafo_yapisi.png` - Trafo sayısı ve panel yapısı
- `05_veri_tamligi.png` - Trafo bazında veri tamlığı haritası
- `06_zaman_serisi_toplam.png` - Toplam tüketim trendi
- `07_haftalik_aylik_mevsimsellik.png` - Haftalık/aylık desenleri
- `08_acf_pacf.png` - Otokorelasyon ve kısmi otokorelasyon
- `09_stl_ayristirma.png` - STL ayrıştırması (trend + mevsimsellik)
- `10_ornek_trafo_serileri.png` - Seçili trafolar için zaman serisi örnekleri
- `11_heteroskedastisite.png` - Varyans heterojenliği analizi
- `12_korelasyon_matrisi.png` - Sayısal özellikler korelasyon matrisi

ACF/PACF ve STL grafikleri için `statsmodels` kuruluysa ilgili analizler de üretilir; kurulu değilse EDA’nın diğer bölümleri çalışmaya devam eder.

## 📁 Proje Yapısı

```text
├── config.py                  # Global konfigürasyon (yollar, özellikler, parametreler)
├── data.py                    # Veri yükleme ve lokasyon hiyerarşisi işleme
├── features.py                # Özellik mühendisliği (takvim, sayısal, kategorik, hava, güneş)
├── validation.py              # RMSLE metriği ve temporal split
├── models_zoo.py              # LightGBM, CatBoost, XGBoost sarmalayıcıları
├── optuna_tuning.py           # Optuna hiperparametre optimizasyonu
├── cv.py                      # Walk-forward cross-validation
├── ensemble.py                # Ensemble ağırlık optimizasyonu ve blending
├── gpu_utils.py               # GPU algılama ve fallback mekanizması
├── train_ensemble.py          # Ana eğitim scripti (Optuna → Validation → Blend → Submit)
├── eda.py                     # Keşifsel veri analizi
│
├── data/                      # Girdi verisi klasörü
│   ├── train.csv              # Eğitim seti (1.2M satır)
│   ├── test.csv               # Test seti (~714K satır)
│   ├── sample_submission.csv  # Submission şablonu
│   ├── open_meteo_cache.csv   # Hava durumu cache'i
│   ├── ilce_koordinatlari.json# İlçe koordinatları
│   ├── izmir_manisa_ilce_nufuslari.xlsx  # Nüfus verileri
│   └── izmir_manisa_yuzolcumleri.xlsx   # Alan verileeri
│
├── outputs/                   # Çıktı dosyaları
│   ├── validation_predictions.csv   # Validation set tahminleri
│   └── submission.csv               # Test set tahminleri (final submission)
│
├── eda_ciktilari/             # EDA raporları ve görselleri
│   ├── eda_raporu.md          # Otomatik oluşturulan bulgu raporu
│   ├── figures/               # 12 analiz grafiği (PNG)
│   └── tables/                # Özet tablolar (CSV)
│
├── catboost_info/             # CatBoost eğitim logları (otomatik)
├── pyproject.toml             # Proje bağımlılıkları (uv)
├── uv.lock                    # Sabit sürüm tanımı
├── requirements.txt           # Eski pip tabanlı kurulum
└── README.md                  # Bu dosya
```

## 🔬 Model Mimarisi İçeriği

### Özellik Kütüphaneleri
- **Veri İşleme**: pandas, numpy, scipy
- **Modeller**: LightGBM, CatBoost, XGBoost
- **Hiperparametre Tuning**: Optuna (150 trial toplam, 3 model × 50)
- **Hava Durumu**: Open Meteo (gerçek meteoroloji verisi, cache'li)
- **Güneş Hesapları**: pvlib (Global Horizontal Irradiance, gün uzunluğu)
- **Takvim**: holidays (Türkiye resmi tatilleri)
- **Coğrafi**: geopy (Nominatim geocoding, koordinatlar)
- **Görselleştirme**: matplotlib, seaborn
- **Zaman Serisi Analizi**: statsmodels (STL ayrıştırması, ACF/PACF)
- **GPU Desteği**: PyTorch (CUDA başarı durumu algılaması)

### Ensemble Stratejisi

1. **Tabanlar**: 3 gradient boosting modeli (herbiri farklı varyansla)
2. **Ağırlıklandırma**: SciPy SLSQP optimizer (RMSLE minimize)
3. **Blending**: Doğrusal ağırlık kombinasyonu
4. **Seed Bag (isteğe bağlı)**: Aynı modeli farkli seed'lerle eğitip varyans azaltma

## 📈 Beklenen Performans

**Validation RMSLE (Target)**: < 0.40

**Başlıca Başarı Faktörleri:**
- ✅ Gradient Boosting ensemble'in güç kombinasyonu
- ✅ Optuna ile kapsamlı hyperparametre araştırması (150 trial)
- ✅ Log1p(tuketim) hedefi RMSLE metriğiyle mükemmel hizalama
- ✅ Temporal split zamansal sızıntıyı önleme
- ✅ Coğrafi + meteoroloji + takvim özelliklerinin synerjisi
- ✅ Panel veri yapısının exploit edilmesi (5,344 trafo)

## 🔮 İleri Geliştirmeler

- **Lag/Rolling Özellikleri**: Trafo bazında 7/14/28 günlük lag ve rolling mean/std
- **Hedef Encoding**: Trafo ve ilçe seviyesi hedef encoding (K-fold içi sızıntı kontrolü)
- **Cold-Start Stratejisi**: Benzer trafo fallback (guc + lokasyon cluster'ı)
- **Multi-Output**: Diğer hedefler varsa multi-target learning
- **Derin Öğrenme**: N-BEATS, TFT, DeepAR (global panel modeli)
- **Stacking**: Ridge meta-model ikinci seviye
- **Uncertainty Quantile**: Tahmin belirsizliği modelleme

## 📝 Notlar

- GPU desteği otomatiktir: CUDA tespit edildiyse LightGBM ve CatBoost GPU modunda çalışır, aksi halde CPU'ya geçer
- Test seti 2,024 trafo içerir (~%29) train'de görülmemiş (cold-start) — lokasyon + güç fallback tavsiye edilir
- Veri kalitesi mükemmel: %0 duplike, %0 eksik, %95.3 tamlık
- EDA otomatik olarak veri profili ve öneriler raporu üretir
