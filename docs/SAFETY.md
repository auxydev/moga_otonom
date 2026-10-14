# Moga – Seviye 4 Güvenlik Konsepti

Bu belge, Moga otonomi yazılımının SAE J3016 **Seviye 4** gereksinimini nasıl
karşıladığını anlatır: tanımlı bir Operasyonel Tasarım Alanı (ODD) içinde dinamik
sürüş görevinin **ve** geri dönüş (fallback) görevinin tamamı sistem tarafından
yürütülür; hiçbir durumda insan sürücünün devralması beklenmez. Sistem bir hatayı
ya da ODD dışına çıkışı kendisi tespit eder ve aracı **Minimum Risk Durumuna (MRC)**
kendisi getirir.

> Not: Bu bir simülasyon prototipidir. Gerçek araçta kullanım için ISO 26262
> (fonksiyonel güvenlik), ISO 21448 / SOTIF (amaçlanan işlevin güvenliği) ve
> UL 4600 (otonom ürünlerin güvenlik değerlendirmesi) süreçleri, donanım
> yedekliliği ve saha doğrulaması gerekir (bkz. “Sınırlamalar”).

## 1. Operasyonel Tasarım Alanı (ODD)

| Boyut | ODD içi | Degrade | ODD dışı → MRM |
|---|---|---|---|
| Yol | HD haritası olan otoyol, kırsal yol, şehir ızgarası | – | haritasız alan (> 2 s şerit eşleşmesi yok) |
| Hız limiti | ≤ 100 km/h | – | > 100 km/h |
| Yağmur (araç yağmur sensörü) | < 0.5 | 0.5–0.9 (80 km/h) | ≥ 0.9 |
| Görüş (LiDAR uzak dönüş mesafesi) | ≥ 90 m | 70–90 m | < 70 m (yoğun sis) |
| Aydınlatma | gündüz / gece | – | – |
| Konum | şerit seviyesi (yanal σ < 0.5 m) | 0.5–1.0 m | > 1.0 m |

ODD izleme `moga/autonomy/safety/odd.py` içindedir; koşullar yalnızca aracın
kendi sensörlerinden kestirilir (yağmur sensörü, LiDAR dönüş dağılımı, lokalizasyon).

## 2. Güvenlik mimarisi

```
 sensörler ─┬─► algılama / lokalizasyon / planlama / kontrol (nominal zincir)
            │                                     │ komut
            ├─► Sağlık izleyici (health.py) ──┐    ▼
            ├─► ODD izleyici (odd.py) ────────┼─► Güvenlik denetçisi (supervisor.py)
            │                                 │      NOMINAL / DEGRADED / MRM / MRC
            │                                 │      hız sınırı, takip mesafesi, MRM türü
            ├─► AEB (aeb.py, bağımsız son savunma; izler + ham LiDAR)
            └─► EDR (edr.py, olay veri kaydı)
```

* **Sağlık izleyici**: her mesaj akışının canlılığı (beklenen periyodun 4 katı),
  verinin makullüğü (LiDAR nokta sayısı, kamera görüntüsü istatistikleri,
  lokalizasyon kovaryansı, GNSS tutarlılığı) ve aktüatör tepkisi (ölçülen
  tekerlek açısı ↔ komutun gecikme/hız sınırı modeli; ölçülen yavaşlama ↔
  komut). Hatalar iyileşmeden sonra 3 s mandallanır (aralıklı hatalar durumu
  sürekli değiştirmez).
* **Güvenlik denetçisi**: bileşen durumlarını hata tepki matrisine göre sistem
  durumuna çevirir. MRM mandallıdır; araç durduktan 1 s sonra MRC'ye geçilir ve
  yalnızca uzaktan operatör / servis tarafından çözülebilir.
* **AEB**: planlayıcıdan bağımsız; yalnızca TTC < 2.5 s ve gereken yavaşlama
  > 5 m/s² iken 8.5 m/s² ile frenler. Yalnızca taze (< 0.35 s) LiDAR noktaları
  kullanılır.
* **Planlayıcı tarafı**: her döngüde uygulanabilir bir “dur” adayı vardır;
  hiçbir aday güvenli değilse önceki yol üzerinde azami frenle durma
  (`EMERGENCY_STOP`) üretilir.

## 3. Sistem durumları

| Durum | Anlamı | Kısıtlar |
|---|---|---|
| `NOMINAL` | tam yetenek | – |
| `DEGRADED` | yedeklilik kaybı / sınırda koşul | ≤ 80 km/h, takip mesafesi ×1.5, keyfi şerit değiştirme ve bypass yok; neden ortadan kalkınca 5 s sonra `NOMINAL` |
| `MRM` | minimum risk manevrası (mandallı) | ≤ 50 km/h, dörtlü flaşör |
| `MRC` | minimum risk durumu | duruş, fren tutulur, dörtlü flaşör |

**MRM türleri** (`planning/planner.py`):
* **Kenara çekilme (`pullover`)**: boşluk kabulüyle şerit şerit sağa, emniyet
  şeridi varsa ona geçilir; yoksa şeridin sağ kenarına yanaşılır. Duruş noktası
  bir kez seçilir (ortalama 2 m/s²) ve kavşak alanı ile yaya geçitlerinin dışında
  tutulur.
* **Şerit içinde durma (`stop`)**: yanal manevranın artık güvenli olmadığı
  durumlar (LiDAR / IMU kaybı, lokalizasyon kaybı): şerit içinde ortalama
  2.2 m/s² ile durma; LiDAR yoksa takip radar izleriyle sürer.

## 4. Hata tepki matrisi

| Hata / koşul | Tespit | Tepki | Senaryo |
|---|---|---|---|
| LiDAR arızası | veri yok (0.45 s) / nokta sayısı < %15 | MRM şerit içinde durma (radar yedek) | `m6_lidar_failure` |
| LiDAR kirlenmesi | nokta sayısı < %60 | DEGRADED | `m6_lidar_degraded` |
| Kamera arızası / körlenme | veri yok / karanlık, düşük kontrastlı görüntü | MRM kenara (emniyet şeridine) çekilme | `m6_camera_failure_pullover` |
| Radar arızası | veri yok | DEGRADED | `m6_radar_failure_degraded` |
| GNSS kesintisi (tünel, alıcı) | fix yok / zaman aşımı | DEGRADED (ölü hesap + şerit lokalizasyonu); 120 s üzeri → MRM | `m1_highway_tunnel`, `m6_gnss_outage_recovery` |
| GNSS tutarsızlığı | EKF ardışık reddi | DEGRADED | birim testi |
| IMU arızası | veri yok | MRM şerit içinde durma | birim testi |
| Teker odometrisi arızası | veri yok | DEGRADED | birim testi |
| Lokalizasyon bozulması | yanal σ > 0.5 / 1.0 m | DEGRADED / MRM şerit içinde durma | birim testi; SPS GNSS + kamera füzyonu kapalı (`tests/test_m2_scenarios.py`) |
| Trafik ışığı algılayıcı arızası | veri yok | önde sinyalize kavşak varsa MRM kenara çekilme, yoksa DEGRADED | `m6_urban_light_detector_failure` |
| Direksiyon aktüatörü bozulması | ölçülen açı ↔ komut modeli, 0.8 s | MRM kenara çekilme | `m6_steering_degraded` |
| Fren bozulması | ölçülen ↔ komut yavaşlama, 0.8 s | MRM kenara çekilme | birim testi |
| Yoğun yağmur / sis (ODD dışı) | yağmur sensörü / LiDAR görüşü | DEGRADED → MRM kenara çekilme | `m6_heavy_rain_odd_exit` |
| Haritasız alan | şerit eşleşmesi yok > 2 s | MRM | birim testi |
| Planlama başarısızlığı | uygun aday yok | azami frenle durma (`EMERGENCY_STOP`) | M4 senaryoları |
| Ani çarpışma riski | AEB (TTC + gereken yavaşlama) | 8.5 m/s² fren, dörtlü flaşör | `m4_aeb_occluded_pedestrian` |

## 5. Olay veri kaydedici (EDR)

`autonomy/safety/edr.py`: 10 Hz halka tampon (son 20 s) – konum, hız, komutlar,
sürüş modu, güvenlik durumu, arızalı bileşenler, en yakın nesne. Tetikleyiciler:
AEB, acil durma, MRM / MRC geçişi, bileşen arızası. Tetiklemede tampon dondurulur,
10 s daha kayıt alınır ve olay kapatılır (`StackConfig.edr_dir` verilirse JSON).

## 6. Doğrulama

* 38 kapalı çevrim senaryo (M0–M6), her biri çarpışma, kural ihlali (kırmızı ışık,
  yaya geçidi), konfor (yavaşlama, yanal ivme), davranış (modlar, güvenlik
  durumları) ve görev kriterleriyle; tamamı her değişiklikte yeniden koşulur.
* Hata enjeksiyonu: `moga/sim/faults.py` (sensör kaybı / bozulması, aktüatör
  bozulması, hava değişimi).
* Determinizm: aynı senaryo farklı süreçlerde ve farklı `PYTHONHASHSEED` ile
  birebir aynı sonucu verir (`tests/test_determinism.py`).
* CI: `.github/workflows/ci.yml` – lint, birim testleri, milestone başına
  senaryo matrisi, senaryo raporu.
* Rapor: [`docs/SCENARIO_REPORT.md`](SCENARIO_REPORT.md).

## 7. Sınırlamalar ve gerçek araca geçiş

* Trafik ışığı algılama nesne seviyesinde modellenmiştir (görüntü işleme yok);
  gerçek araçta kamera tabanlı ışık sınıflandırıcı + V2X/SPaT ile çeşitlendirme
  önerilir.
* Yalnız-radar izleri hareketli hedefler içindir; LiDAR kaybında duran araçlar
  radar tarafından görülmeyebilir (bu yüzden tepki hemen şerit içinde durmadır).
  Gerçek araçta ikinci bir LiDAR veya kamera tabanlı nesne algılama yedeği gerekir.
* Simülasyondaki NPC davranışları (IDM, ışık uyumu, dolu kavşağa girmeme)
  insan sürücülerden daha düzenlidir; saldırgan / kural dışı davranışlar için
  ek senaryolar gereklidir.
* Donanım yedekliliği (çift güç beslemesi, yedek fren / direksiyon
  aktüatörleri, iki bağımsız hesaplama birimi) yazılımın varsaydığı
  “güvenli duruş yapılabilir” koşulunun ön şartıdır.
* Uzaktan destek (teleoperasyon / görev iptali, MRC'den çıkış) ve yolcu
  arayüzü bu prototipin kapsamı dışındadır.
* Gerçek sensör verisiyle ölçülen algılama sınırları (M7): işaretli yolda araç
  algılama koridorda %94, ancak yoğun şehir dokusunda geometrik LiDAR dedektörünün
  kesinliği ~%30 – haritasız kipte yol kenarı nesneleri yol kullanıcısı
  sanılabilir. Gerçek araç için öğrenmeli dedektör + HD harita ilgi bölgesi şarttır
  ([`REAL_DATA.md`](REAL_DATA.md)).
