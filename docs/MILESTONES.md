# Moga Otonomi – Milestone Yol Haritası

Her milestone sonunda **çalışan bir prototip** çıkar: kod + otomatik testler +
MogaSim'de kapalı çevrim (closed-loop) koşan senaryolar + görsel çıktı.
Bir milestone, kendi kabul kriterleri ve **önceki tüm milestone testleri**
yeşilken tamamlanmış sayılır (regresyon koruması).

| # | Milestone | Kapsam | Prototip / Demo | Durum |
|---|-----------|--------|-----------------|-------|
| M0 | Simülatör çekirdeği (MogaSim) | HD harita modeli, harita kütüphanesi, dinamik bisiklet araç modeli + EV güç aktarma/batarya, IDM trafik, yayalar, trafik ışıkları, metrikler, görselleştirme, senaryo koşucusu | Referans (oracle) sürücü otoyol / pist / şehir senaryolarını kazasız sürer | ✅ |
| M1 | Lokalizasyon + GPS tabanlı otonom sürüş | GNSS/IMU/teker odometrisi sensör modelleri, WGS‑84↔ENU, EKF füzyonu, GPS görev (lat/lon) → şerit seviyesinde rota (A*), pure‑pursuit + PID | Araç yalnızca sensör verisiyle GPS hedeflerine gider | ✅ |
| M2 | Kamera ile şerit algılama + şerit takibi | Kamera sensör modeli, IPM, sırt (ridge) filtresi, kayan pencere, polinom uydurma, zamansal filtre, Stanley kontrolcü, şerit‑göreli lokalizasyon düzeltmesi | Virajlı yolda kamera tabanlı şerit takibi, GNSS kesintisinde şeritle konum düzeltme | ✅ |
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

## M2 – Kamera ile şerit algılama ve şerit‑göreli lokalizasyon (tamamlandı)

**Kamera sensör modeli** (`moga/sim/sensors/camera.py`): 400×225 gri tonlu pinhole kamera
(60° FOV, 3° aşağı eğim). Piksel → zemin eşlemesi bir kez hesaplanır; her karede yüzey
rasterı (asfalt/emniyet şeridi/çim/boya: yaya geçidi, stop çizgisi), KD‑ağacındaki şerit
çizgisi örnekleri (kesikli 3 m/6 m, düz, çift sarı) mesafeye bağlı kenar yumuşatmasıyla
çizilir; diğer araçlar ışın‑kutu kesişimiyle şeritleri kapatır (oklüzyon); yağmur/sis
kontrastı düşürür, gürültü ekler; gece modu.

**Şerit algılama** (`moga/autonomy/perception/lane_detection.py`)
1. IPM ile kuş bakışı ızgara (x 4.5–42 m, y ±8 m),
2. mesafeye uyarlanan sırt (koyu‑açık‑koyu) filtresi → çift çizgileri de ayırır,
3. histogram ile başlatma veya takip edilen sınırlardan öngörü,
4. kayan pencereler (çizgi bitince durur – kavşak),
5. sağlam ağırlıklı en küçük kareler ile ikinci derece polinom + katsayı kovaryansı,
6. ego hareketiyle telafi edilen sınır başına Kalman filtresi,
7. çizgi tipi (kesikli / düz / çift) + zamansal oylama, şerit genişliği tutarlılığı.

Doğruluk (10 m'de sınır konumu, yer gerçeğine göre): otoyol **1 cm**, kırsal yol (çift sarı)
**2–3 cm**, yoğun yağmurda **2 cm**.

**Şerit‑göreli lokalizasyon** (`localization/localizer.py`): algılanan şerit merkezine yanal
sapma + bağıl yönelim, EKF'ye haritadaki şerit merkez çizgisine göre ölçüm olarak girer.
Hangi şeridin görüldüğü (veri ilişkilendirme) aday şeritler arasında Mahalanobis mesafesi
ve **çizgi tipi uyumu** (ör. en sol şerit: solda düz, sağda kesikli) ile seçilir; belirsizse
güncelleme yapılmaz. Kavşak içi / kavşak öncesi (çizgisiz) bölgeler dışlanır.

**Fiziksel modelleme düzeltmeleri** (virajlarda 0.3 m hatayı giderdi)
* EKF hareket modeline IMU yanal ivmesinden arka aks **yanal kayma açısı** eklendi,
* Stanley ileri beslemesi yalnızca **ön lastik kayma açısını** ekleyecek şekilde düzeltildi
  (ön aks yön hatası arka kayma açısını zaten içerir) + küçük integral terimi.

**Lokalizasyon hazır kapısı:** şerit seviyesinde konum (yanal σ < 0.35 m) yoksa rota
planlanmaz; araç `LOCALIZING` modunda kamerayla doğrudan şerit takibi yapar.

| Senaryo | İçerik | Sonuç (seed 0) |
|---|---|---|
| `m2_curvy_sps_lane_keep` | Virajlı otoyol, 90 km/h, standart GPS (~1.5 m) | şerit RMS 0.05 m |
| `m2_rural_dgps_fusion` | Kırsal yol, DGPS + çok yollu yansıma | şerit RMS 0.07 m (M1'de 0.3 m) |
| `m2_highway_sps_lane_change` | Standart GPS ile iki şerit değiştirip hedefe | şerit RMS 0.01 m |
| `m2_curvy_long_tunnel` | ~800 m virajlı GNSS kesintisi | şerit maks. 0.12 m |
| `m2_rain_rural` | Yoğun yağmur + DGPS | şerit RMS 0.04 m |

![](images/m2_camera_rural.png)
![](images/m2_fusion_comparison.png)
