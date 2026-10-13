# Moga Otonomi – Milestone Yol Haritası

Her milestone sonunda **çalışan bir prototip** çıkar: kod + otomatik testler +
MogaSim'de kapalı çevrim (closed-loop) koşan senaryolar + görsel çıktı.
Bir milestone, kendi kabul kriterleri ve **önceki tüm milestone testleri**
yeşilken tamamlanmış sayılır (regresyon koruması).

| # | Milestone | Kapsam | Prototip / Demo | Durum |
|---|-----------|--------|-----------------|-------|
| M0 | Simülatör çekirdeği (MogaSim) | HD harita modeli, harita kütüphanesi, dinamik bisiklet araç modeli + EV güç aktarma/batarya, IDM trafik, yayalar, trafik ışıkları, metrikler, görselleştirme, senaryo koşucusu | Referans (oracle) sürücü otoyol / pist / şehir senaryolarını kazasız sürer | ✅ |
| M1 | Lokalizasyon + GPS tabanlı otonom sürüş | GNSS/IMU/teker odometrisi sensör modelleri, WGS‑84↔ENU, EKF füzyonu, GPS görev (lat/lon) → şerit seviyesinde rota (A*), pure‑pursuit + PID | Araç yalnızca sensör verisiyle GPS hedeflerine gider | ✅ |
| M2 | Kamera ile şerit algılama + şerit takibi | Kamera sensör modeli, IPM, sırt (ridge) filtresi, kayan pencere, polinom uydurma, zamansal filtre, Stanley kontrolcü, şerit‑göreli lokalizasyon düzeltmesi | Virajlı yolda kamera tabanlı şerit takibi, GNSS kesintisinde şeritle konum düzeltme | ⏳ |
| M3 | LiDAR/Radar algılama + çoklu nesne takibi | Işın izleme LiDAR, radar, zemin ayrıştırma (RANSAC), harita ROI, kümeleme, L‑shape kutu uydurma, sınıflandırma, KF + Macar algoritması ile takip, radar füzyonu, şerit tabanlı tahmin | Trafik içinde nesneler tespit/takip edilir, ACC ile takip | ⏳ |
| M4 | Davranış + hareket planlama | Davranış durum makinesi, Frenet kafes (lattice) planlayıcı, çarpışma kontrolü, ACC, şerit değiştirme, sollama, engelden kaçınma, araya girme (cut‑in), AEB | Otoyolda sollama / şerit değiştirme / acil fren senaryoları | ⏳ |
| M5 | Şehir içi sürüş | Kavşak bağlayıcıları, trafik ışığı algılama, stop çizgisi, yaya geçidi, yayalara yol verme, kavşak çatışma kontrolü, kentsel GPS görevi | Sinyalize grid üzerinde A noktasından B noktasına otonom sürüş | ⏳ |
| M6 | Seviye 4 güvenlik | Sağlık izleme, ODD izleme, hata enjeksiyonu, degrade modlar, Minimum Risk Manevrası (emniyet şeridine çekilip durma), olay veri kaydedici (EDR), tam senaryo seti + CI | Sensör/aktüatör arızasında araç kendini güvenli duruma getirir | ⏳ |

## Seviye 4 (SAE J3016) gereksinimlerinin karşılanması

SAE Seviye 4: tanımlı bir **Operasyonel Tasarım Alanı (ODD)** içinde sürüş
görevinin tamamını ve **geri dönüş (fallback)** görevini sistem üstlenir;
insan sürücüye devretme beklenmez.  Bunun yazılımdaki karşılıkları:

| L4 gereksinimi | Modül | Milestone |
|----------------|-------|-----------|
| Kendi konumunu bilmek (şerit seviyesi) | `autonomy/localization` (EKF, şerit düzeltmesi) | M1, M2 |
| Çevresini algılamak (şerit, araç, yaya, engel, ışık) | `autonomy/perception`, `autonomy/tracking` | M2, M3, M5 |
| Diğer katılımcıların davranışını öngörmek | `autonomy/prediction` | M3 |
| Görev / rota planlama (GPS hedefi) | `autonomy/planning/route_planner` | M1, M5 |
| Trafik kurallarına uyum (hız, ışık, yol hakkı) | `autonomy/planning/behavior` | M4, M5 |
| Güvenli ve konforlu yörünge | `autonomy/planning/frenet` | M4 |
| Araç kontrolü | `autonomy/control` | M1, M2 |
| Bağımsız acil fren katmanı | `autonomy/safety/aeb` | M4 |
| Arıza tespiti ve degrade çalışma | `autonomy/safety/health` | M6 |
| ODD dışına çıkışı algılama | `autonomy/safety/odd` | M6 |
| Sürücüsüz geri dönüş: Minimum Risk Durumu | `autonomy/safety/mrm` | M6 |
| Olay kaydı / denetlenebilirlik | `autonomy/safety/edr` | M6 |

## M1 – Lokalizasyon ve GPS tabanlı sürüş (tamamlandı)

**Sensör modelleri** (`moga/sim/sensors`)
* GNSS: RTK‑fixed / RTK‑float / DGPS / SPS modları; Gauss‑Markov korelasyonlu hata +
  beyaz gürültü, çok yollu yansıma (multipath) aykırı değerleri, kapsama dışı bölgeler
  (tünel), Doppler hız, WGS‑84 çıktı.
* IMU: ivme / açısal hız gürültüsü, sabit + rastgele yürüyüş bias.
* Teker odometrisi: lastik yarıçap (ölçek) hatası, gürültü, direksiyon açısı.

**Otonomi** (`moga/autonomy`)
* `localization/ekf.py` – 7 durumlu EKF: konum, yönelim, hız, gyro bias, ivme bias,
  odometri ölçek faktörü. GNSS anten kaldıraç kolu, Doppler hız, teker hızı,
  duruşta sıfır‑hız güncellemesi (ZUPT), Mahalanobis (χ²) aykırı değer reddi,
  Joseph formu kovaryans güncellemesi.
* `planning/route_planner.py` – şerit grafında A*: ardıl bağlantılar, kesikli çizgide
  şerit değiştirme kenarları, kavşak dönüş maliyetleri, kapalı pistte tur.
  GPS (lat/lon) görev noktaları → şerit + yay uzunluğu.
* `planning/route_path.py` – rotadan referans yol (şerit değişimleri 5. derece
  geçişlerle), hız limiti / eğrilik / hedefte durma hız profili.
* `control/controllers.py` – Stanley (ön aks, eksik yönlenme telafisi, yaw sönümleme),
  pure pursuit, PID + ileri besleme + sarsıntı (jerk) sınırlı boylamsal kontrol.

**Senaryolar ve sonuçlar** (`python scripts/run_scenario.py --milestone M1`)

| Senaryo | İçerik | Sonuç (seed 0) |
|---|---|---|
| `m1_gps_loop_rtk` | Oval pistte 3 GPS hedefli görev + şerit değiştirme | lok. RMSE 0.02 m, şerit RMS 0.04 m |
| `m1_rural_rtk` | Virajlı kırsal yolda GPS hedefi | lok. RMSE 0.02 m |
| `m1_urban_route` | Şehir ızgarasında dönüşlü rota, 2 ara hedef | lok. RMSE 0.02 m |
| `m1_highway_tunnel` | 90 km/h'te 350 m tünel (GNSS yok) | maks. hata < 0.6 m (6 tohum) |
| `m1_rural_multipath` | RTK‑float + %5 çok yollu yansıma (8 m) | aykırı değerler reddedilir |

**Öğrenilen:** DGPS/SPS'in korelasyonlu hatası (0.5–2 m) EKF ile giderilemez; şerit
içinde kalmak için kamera ile şerit‑göreli konum düzeltmesi gerekir (M2).

![](images/m1_urban_route.png)
![](images/m1_tunnel_localization.png)
