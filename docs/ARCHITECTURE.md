# Mimari

Proje iki ana parçadan oluşur:

1. **MogaSim** (`moga/sim`, `moga/map`) – deterministik, headless çalışan,
   Python tabanlı 2.5B sürüş simülatörü. Yer gerçeğini (ground truth) üretir,
   sensör ölçümlerini modeller, aracın fiziksel tepkisini hesaplar ve
   metrikleri ölçer.
2. **Moga otonomi yığını** (`moga/autonomy`) – yalnızca sensör mesajlarını,
   önceden hazırlanmış HD haritayı ve GPS görevini görerek sürüş komutu üretir.
   Simülatörün iç durumuna **erişemez**; bu sayede simülasyondaki başarı
   gerçek dünyaya taşınabilir bir başarıdır.

```
                 ┌────────────────────────── MogaSim ───────────────────────────┐
                 │  HD harita (yer gerçeği) · trafik (IDM) · yayalar · ışıklar   │
                 │  araç dinamiği (dinamik bisiklet + EV güç aktarma + batarya)  │
                 │  hava durumu · hata enjeksiyonu · metrikler                   │
                 └──────┬──────────────────────────────────────────▲───────────┘
            SensorBundle│ GNSS · IMU · teker · LiDAR · kamera ·      │ControlCommand
                        │ radar · trafik ışığı · araç durumu         │(direksiyon, ivme,
                        ▼                                            │ sinyal, dörtlü)
 ┌──────────────────────────────── Moga otonomi yığını ──────────────────────────────┐
 │                                                                                   │
 │  Lokalizasyon (EKF: GNSS+IMU+odometri+şerit)  ──►  Rota planlama (A*, şerit grafı)│
 │          │                                              │                         │
 │  Algılama: kamera şerit · LiDAR nesne · radar · ışık    │                         │
 │          │                                              ▼                         │
 │  Takip (KF + Macar) ──► Tahmin ──►  Davranış (FSM) ──► Hareket planlama (Frenet)  │
 │                                          ▲                     │                  │
 │  Güvenlik: sağlık · ODD · MRM · EDR ─────┘                     ▼                  │
 │                         AEB (bağımsız) ─────────────────►  Kontrol (Stanley/PID)  │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

## Planlama ve kontrol hattı

```
GPS görevi ─► Rota planlama (A*, şerit grafı) ─► Rota takibi (zorunlu şerit değişimi, kalan yol)
                                                        │
Tahminler ─► Nesne akıl yürütme (Frenet: şerit başına boşluk/hız, durağan engel / kuyruk)
Işık algısı + tahminler ─► Trafik kuralları (ışık, sarı ikilemi, yaya geçidi, kavşak yol hakkı)
                                                        │  durma noktaları, hız sınırları
                       Davranış FSM: LANE_KEEP · PREPARE_LC · LANE_CHANGE · BLOCKED · BYPASS
                                                        │  hedef ofsetler, hedef hızlar, durma noktaları
                       Frenet kafes planlayıcı (aday üretimi → fizibilite → çarpışma/ST → maliyet)
                                                        │  6 s yörünge (0.05 s çözünürlük)
                       Yörünge kontrolcüsü (Stanley + hız/ivme ileri besleme)  ◄── AEB (bağımsız)
```

Planlayıcı her döngüde önceki planın Frenet durumundan dikiş yaparak başlar; uygun aday
kalmazsa önceki yol üzerinde maksimum frenle durma yörüngesi üretilir (`EMERGENCY_STOP`).
AEB izlerden ve ham LiDAR noktalarından bağımsız olarak çarpışma riskini değerlendirir ve
planlayıcının çıktısını geçersiz kılabilir.

## Güvenlik katmanı

Nominal zincire paralel çalışan, ondan bağımsız bileşenler
(ayrıntı: [`SAFETY.md`](SAFETY.md)):

| Bileşen | Dosya | Görev |
|---|---|---|
| Sağlık izleyici | `autonomy/safety/health.py` | sensör canlılığı / makullüğü, aktüatör tepkisi |
| ODD izleyici | `autonomy/safety/odd.py` | yağmur, görüş, haritalı alan, hız limiti |
| Güvenlik denetçisi | `autonomy/safety/supervisor.py` | NOMINAL / DEGRADED / MRM / MRC, kısıtlar |
| MRM planlama | `autonomy/planning/planner.py` | kenara / emniyet şeridine çekilme, şerit içinde durma |
| AEB | `autonomy/safety/aeb.py` | son savunma acil fren |
| EDR | `autonomy/safety/edr.py` | olay veri kaydı |
| Hata enjeksiyonu (sim) | `sim/faults.py` | sensör / aktüatör / çevre hataları |

## Koordinat sistemleri

| Çerçeve | Tanım |
|---------|-------|
| WGS‑84 | GNSS alıcısının verdiği enlem/boylam/yükseklik |
| Harita (ENU) | HD haritanın coğrafi orijinine bağlı yerel Doğu‑Kuzey‑Yukarı, metre |
| base_link | Arka aks merkezi; x ileri, y sol, z yukarı (z = 0 zeminde) |
| Sensör çerçeveleri | `T_vehicle_<sensör>` 4×4 dönüşümleri, araç başına bir `SensorRig` JSON dosyası (`common/calibration.py`) |
| Kamera | Doğrultulmuş (rectified) kamera: x sağ, y aşağı, z ileri; `P` (3×4) + `T_cam_vehicle` |
| Frenet | Referans çizgi boyunca `s` (yay uzunluğu) ve `d` (sola pozitif yanal sapma) |

## Zamanlama

| Bileşen | Frekans |
|---------|---------|
| Simülasyon adımı / araç dinamiği | 50 Hz (dinamik 200 Hz alt adım) |
| IMU, teker odometrisi, kontrol | 50 Hz |
| GNSS | 10 Hz |
| LiDAR, kamera, radar, planlama | 10 Hz |

Tüm rastgelelik senaryo tohumundan (seed) türetilir; aynı senaryo aynı
sonucu verir. Bu, regresyon testlerini güvenilir kılar.

## Dizin yapısı

```
moga/
  common/      geometri, geodezi, mesaj tipleri, araç parametreleri
  map/         HD harita modeli, harita oluşturucu, harita kütüphanesi
  sim/         dünya, araç modeli, aktörler, sensörler, hatalar, simülasyon koşucusu, oracle
  autonomy/    otonomi yığını (lokalizasyon, algılama, takip, tahmin, planlama, kontrol, güvenlik)
  eval/        metrikler, toplu koşucu, raporlama, gerçek kayıtlarda açık çevrim değerlendirme (replay)
  io/          gerçek sensör kayıtları (KITTI raw okuyucu)
  bridges/     üçüncü taraf simülatör köprüleri (MetaDrive)
  viz/         üstten görünüm render, GIF/PNG
  scenarios/   milestone bazlı test senaryoları
scripts/       komut satırı araçları
tests/         birim ve senaryo testleri
docs/          dokümantasyon ve görseller
```

## Mesajlar

`moga/common/messages.py` ROS 2 mesajlarına benzer veri sınıflarını içerir
(`GnssFix`, `ImuMeasurement`, `PointCloud`, `CameraImage`, `RadarScan`,
`TrackedObject`, `Trajectory`, `ControlCommand` …).  Otonomi yığını yalnızca bu
mesajlarla konuşur; ileride ROS 2 köprüsü / gerçek donanım sürücüleri aynı
arayüze bağlanabilir.

Aynı arayüz üç kaynaktan beslenir ve yığın kodu değişmeden çalışır
(M7, ayrıntılar: [`REAL_DATA.md`](REAL_DATA.md)):

| Kaynak | Sağlayıcı | Kullanım |
|---|---|---|
| MogaSim | `sim/sensors/*` | kapalı çevrim senaryolar, hata enjeksiyonu, CI |
| Gerçek kayıt (KITTI raw) | `io/kitti.py` | açık çevrim algılama değerlendirmesi (`eval/replay.py`) |
| MetaDrive | `bridges/metadrive.py` | bağımsız simülatörde kapalı çevrim sürüş |
