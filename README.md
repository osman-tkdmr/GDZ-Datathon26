# GDZ Datathon 26: Trafo Tüketim Tahmini

Bu proje, trafo bazında elektrik tüketimini tahmin eden bir baseline modelidir. Model, RMSLE metriğiyle uyumlu olması için `tuketim` hedefinin `log1p` dönüşümü üzerinde LightGBM regresyonu eğitir.

## Gereksinimler

- Python `>= 3.12`
- [`uv`](https://docs.astral.sh/uv/)

Bağımlılıkların ve sürümlerin kaynağı `pyproject.toml` ile `uv.lock` dosyalarıdır. `requirements.txt` eski pip tabanlı kurulumlar için tutulur; önerilen yöntem `uv` kullanmaktır.

## Kurulum

Projeyi klonladıktan sonra proje klasöründe çalıştırın:

```bash
uv sync
```

Bu komut `.venv` sanal ortamını oluşturur veya günceller ve `pyproject.toml` içindeki bağımlılıkları `uv.lock` dosyasındaki sabit sürümlerle kurar. Ortamı ayrıca etkinleştirmek gerekmez; komutları `uv run` ile çalıştırabilirsiniz.

## Veri

Aşağıdaki dosyalar `data/` klasöründe bulunmalıdır:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

Eğitim ve test verilerinde kullanılan temel kolonlar:

| Kolon | Açıklama |
| --- | --- |
| `id` | Tahmin kaydının kimliği |
| `tarih` | Gözlem tarihi |
| `tanim` | Trafo kimliği/tanımı |
| `lokasyon` | `il>bolge>ilce` biçimindeki konum bilgisi |
| `guc` | Trafo gücü |
| `tuketim` | Eğitim verisindeki hedef kolon |

`tuketim` yalnızca eğitim verisinde bulunur. `lokasyon` alanı `>` karakteriyle ayrıştırılarak `il`, `bolge` ve `ilce` özellikleri oluşturulur.

## Model ve özellikler

Modelin kullandığı kategorik özellikler:

- `tanim`, `lokasyon`, `il`, `bolge`, `ilce`

Sayısal özellikler:

- `guc` ve `guc_log`
- yıl, ay, gün, haftanın günü, yılın günü ve hafta numarası
- hafta sonu, ay başlangıcı ve ay sonu göstergeleri

Lag ve rolling özellikleri bu baseline’a dahil değildir. Testte geçmişi bulunmayan cold-start trafolar ve çok adımlı test tahmini bu özelliklerin dikkatli bir tasarım gerektirmesinin temel nedenleridir.

## Validasyon

Rastgele K-Fold yerine zamansal ayırma kullanılır. Eğitim verisinin son 60 günü validation seti, önceki tarihler ise eğitim seti olarak ayrılır.

Validation hedefi ve model hedefi `log1p` dönüşümüyle hizalanır. Tahminler `expm1` ile gerçek ölçeğe çevrilir ve negatif değerler sıfıra kırpılır; RMSLE bu gerçek ölçekli tahminlerle hesaplanır.

## Çalıştırma

Validation ve submission üretmek için:

```bash
uv run train.py
```

Başarılı çalıştırmanın sonunda validation skoru terminale yazdırılır ve şu dosyalar oluşturulur:

```text
outputs/validation_predictions.csv
outputs/submission.csv
```

`submission.csv` dosyası `id` ve tahmin edilen `tuketim` kolonlarını içerir. `outputs/` klasörü yoksa eğitim betiği tarafından otomatik oluşturulur.

## Keşifsel veri analizi (EDA)

EDA betiği train/test verilerini inceler; veri kalitesi, hedef dağılımı, güç, lokasyon, trafo panel yapısı, tarih kapsamı, mevsimsellik, RMSLE davranışı ve korelasyonlar için grafik ve özet tablolar üretir.

```bash
uv run eda.py --data-dir data --output-dir eda_ciktilari
```

`--data-dir` içinde `train.csv`, `test.csv` ve varsa `sample_submission.csv` aranır. Çıktılar:

```text
eda_ciktilari/eda_raporu.md  # Otomatik oluşturulan bulgu ve öneri raporu
eda_ciktilari/figures/       # PNG grafikler
eda_ciktilari/tables/        # CSV özet tabloları
```

ACF/PACF ve STL grafikleri için `statsmodels` kuruluysa ilgili analizler de üretilir; kurulu değilse EDA’nın diğer bölümleri çalışmaya devam eder.

## Proje yapısı

```text
config.py       # Yollar, kolon adları ve LightGBM parametreleri
data.py         # CSV okuma ve lokasyon hiyerarşisi
features.py     # Takvim, sayısal ve kategorik özellikler
model.py        # LightGBM eğitim ve tahmin yardımcıları
validation.py   # Zamansal ayırma ve RMSLE
train.py        # Uçtan uca eğitim ve çıktı üretimi
eda.py          # Kapsamlı keşifsel veri analizi
data/           # Girdi CSV dosyaları
outputs/        # Validation ve submission çıktıları
eda_ciktilari/  # EDA raporu, grafikler ve tablolar
```

## Gelecek geliştirmeler

Bu sürüm sağlam bir başlangıç baseline’ıdır. Model performansını artırmak için trafo lag/rolling istatistikleri, tatil özellikleri, trafo ve lokasyon hedef encoding’i, cold-start fallback’i ve modeller arası ensemble değerlendirilebilir.
