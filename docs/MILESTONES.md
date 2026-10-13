# Moga Otonomi – Milestone Yol Haritası

Her milestone sonunda **çalışan bir prototip** çıkar: kod + otomatik testler +
MogaSim'de kapalı çevrim (closed-loop) koşan senaryolar + görsel çıktı.
Bir milestone, kendi kabul kriterleri ve **önceki tüm milestone testleri**
yeşilken tamamlanmış sayılır (regresyon koruması).

| # | Milestone | Kapsam | Prototip / Demo | Durum |
|---|-----------|--------|-----------------|-------|
| M0 | Simülatör çekirdeği (MogaSim) | HD harita modeli, harita kütüphanesi, dinamik bisiklet araç modeli + EV güç aktarma/batarya, IDM trafik, yayalar, trafik ışıkları, metrikler, görselleştirme, senaryo koşucusu | Referans (oracle) sürücü otoyol / pist / şehir senaryolarını kazasız sürer | ✅ |
| M1 | Lokalizasyon + GPS tabanlı otonom sürüş | GNSS/IMU/teker odometrisi sensör modelleri, WGS‑84↔ENU, EKF füzyonu, GPS görev (lat/lon) → şerit seviyesinde rota (A*), pure‑pursuit + PID | Araç yalnızca sensör verisiyle GPS hedeflerine gider | ⏳ |
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
