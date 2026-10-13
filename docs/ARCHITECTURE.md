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
Tahminler ─► Nesne akıl yürütme (Frenet: şerit başına boşluk/hız, durağan engel)
                                                        │
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

## Koordinat sistemleri

| Çerçeve | Tanım |
|---------|-------|
| WGS‑84 | GNSS alıcısının verdiği enlem/boylam/yükseklik |
| Harita (ENU) | HD haritanın coğrafi orijinine bağlı yerel Doğu‑Kuzey‑Yukarı, metre |
| base_link | Arka aks merkezi; x ileri, y sol, z yukarı |
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
  eval/        metrikler, toplu koşucu, raporlama
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
