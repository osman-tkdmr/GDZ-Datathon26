# Trafo Bazlı Günlük Tüketim Tahmini — EDA Raporu

*Otomatik oluşturulma zamanı: 2026-08-28 22:00*

Bu rapor `eda.py` scripti tarafından, sağlanan `train.csv` / `test.csv` dosyaları
üzerinden otomatik olarak üretilmiştir. Tüm sayısal bulgular çalıştırma anındaki
gerçek veriye dayanır; grafikler `figures/`, ara tablolar `tables/` klasöründedir.

# 0) VERİ YÜKLEME

- **train.csv**: 1,226,237 satır × 5 kolon (2025-01-01 → 2026-03-31)
- **test.csv**: 714,688 satır × 5 kolon (2026-04-01 → 2026-07-31)

## 1) GENEL BAKIŞ — train

- **train**: (tanim, tarih) anahtarında **0** duplike satır yok ✅.
- **train** eksik değer: yok ✅

## 1) GENEL BAKIŞ — test

- **test**: (tanim, tarih) anahtarında **0** duplike satır yok ✅.
- **test** eksik değer: yok ✅

## 2) HEDEF DEĞİŞKEN ANALİZİ — tuketim (kWh)

- Ortalama tüketim **3251.9 kWh**, medyan **1075.2 kWh**, p99 **14655.0 kWh**, maksimum **50403051.0 kWh** — dağılım ciddi biçimde **sağa çarpık** (ham çarpıklık=243.40, log1p sonrası=-1.35).
- Sıfır tüketim: **57,536** satır (4.692%), negatif tüketim: **0** satır, eksik: **0** satır.
- `log1p` dönüşümü çarpıklığı belirgin biçimde azaltıyor → RMSLE metriğiyle uyumlu olarak **hedefi log1p ölçeğinde modellemek** (ve tahmin sonrası expm1 ile geri çevirmek) mantıklı.

## 3) KURULU GÜÇ (guc) ANALİZİ

- `guc` (kurulu güç), trafoların **0** tanesinde zaman içinde değişkenlik gösteriyor (toplam 5,344 trafo). Beklendiği gibi her trafo için sabit ✅ — güvenle statik (zamandan bağımsız) bir özellik olarak kullanılabilir.
- `guc` ile `tuketim` arasında log ölçekte Pearson r=**0.478**, Spearman ρ=**0.666** — güç, tüketimin güçlü bir belirleyicisi (orta düzey ilişki).
- Trafo bazında `guc` dağılımı için train/test KS testi: statistic=**0.0336**, p=**0.002032** — dağılımlar istatistiksel olarak anlamlı derecede farklı ⚠️ (kovaryans kayması riski).

## 4) LOKASYON HİYERARŞİSİ ANALİZİ

- Lokasyon hiyerarşisinde **2** il, **3** bölge, **47** ilçe seviyesi bulunuyor; satırların **%0.0**'i jenerik "GEDİZ EDAŞ" etiketiyle (hiyerarşi bilgisi eksik) geliyor.
- test.csv'deki **47** benzersiz lokasyondan **0** tanesi train.csv'de hiç görülmemiş ✅ — tüm test lokasyonları train'de temsil ediliyor.

## 5) TRAFO (tanim) YAPISI ANALİZİ — PANEL VERİ BOYUTLARI

- train'de **5,344**, test'te **7,036** benzersiz trafo var. Test trafolarının **%71.2**'i train'de de geçmiş (ortak: 5,012 trafo).
- ⚠️ **2,024 trafo yalnızca test.csv'de** bulunuyor (train'de geçmiş verisi yok) → **cold-start problemi**: bu trafolar için lag/rolling gibi geçmişe dayalı özellikler üretilemez; guc + lokasyon bazlı fallback/benzer-trafo stratejisi gerekir.

## 6) TARİH KAPSAMI VE GÜNLÜK VERİ TAMLIĞI

- train: **2025-01-01 → 2026-03-31**, test: **2026-04-01 → 2026-07-31**. train sonu ile test başlangıcı arasında **1 gün** boşluk var (test, train'in hemen ardından başlıyor).
- Trafo bazında ortalama **veri tamlık oranı %95.3**; **4,104/5,344** trafo kendi min-max tarih aralığında hiç gün atlamıyor. Eksik günler için lag/rolling özellik hesaplarken tarih bazlı reindex + uygun imputation (ör. forward-fill değil, aynı haftanın günü ortalaması) önerilir.

## 7) ZAMAN SERİSİ DESENLERİ — TREND, MEVSİMSELLİK, TATİL ETKİSİ

- **Haftalık mevsimsellik** belirgin: hafta sonu ortalama tüketim, hafta içine göre **%6.1** farklı → `haftanin_gunu`/`hafta_sonu` özellikleri önemli.
- **Resmi tatil etkisi**: tatil günlerinde ortalama tüketim normal günlere göre **%-18.3** farklı → bir `tatil_mi` (ve varsa `arefe_mi`) flag özelliği eklenmesi önerilir (script içindeki `HOLIDAYS_TR` takvimi Oca 2025–Tem 2026 aralığını kapsar).
- ACF/PACF ve STL ayrıştırma grafikleri, güçlü **7 günlük (haftalık) periyodiklik** ve uzun vadeli bir **trend/mevsimsellik** bileşeni olup olmadığını görselleştirir; `figures/08_acf_pacf.png` ve `09_stl_ayristirma.png` dosyalarına bakınız.

## 8) TEST.CSV ID FORMAT DOĞRULAMASI

- test.csv `id` alanı `tanim_tarih` formatına **%100.0** oranında uyuyor ✅.
- test.csv id kümesi, sample_submission.csv id kümesiyle birebir eşleşiyor ✅.

## 9) RMSLE METRİĞİNE ÖZEL DİAGNOSTİKLER

- Trafo bazında ortalama tüketim ile standart sapma arasında log ölçekte korelasyon=**0.881** — değişkenlik seviyeyle birlikte artıyor (heteroskedastisite); ham ölçekte MSE/RMSE optimize etmek büyük trafoları domine ettirir. RMSLE zaten göreli hataya odaklandığından **log1p(tuketim) hedefiyle RMSE optimize etmek** RMSLE'yi optimize etmeye pratikte çok yakın sonuç verir.
- Örnek hesap tablosu (`tables/rmsle_ornek_ceza_tablosu.csv`) 2 katlık göreli hatanın, 100 kWh'lik küçük bir trafoda da 100.000 kWh'lik dev bir trafoda da **aynı RMSLE katkısını** verdiğini gösteriyor — bu da modelin **küçük trafolarda göreli hatayı büyük trafolar kadar ciddiye alması** gerektiği anlamına gelir (ör. MAPE'ye yakın davranan bir kayıp fonksiyonu).

## 10) KORELASYON MATRİSİ (sayısal + takvim özellikleri)


# 11) BULGULARIN MİMARİ TASARIMA YANSIMASI — ÖNERİLER


Veri, **~5,344 trafo × ~455 gün**lük bir **panel (uzun format) zaman serisi**
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
  olarak ayırıp, test'e olan **1 günlük
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
