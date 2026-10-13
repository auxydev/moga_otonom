# Moga Otonom – Seviye 4 Otonom Sürüş Yazılımı

Moga elektrikli otonom aracı için sıfırdan yazılmış **SAE Seviye 4** otonom
sürüş yazılımı ve onu sürekli test etmek için geliştirilen **MogaSim**
simülasyon ortamı.

Geliştirme milestone mantığıyla ilerler; her milestone sonunda simülasyonda
kapalı çevrim çalışan bir prototip ve otomatik testler bulunur.
Yol haritası ve durum: [`docs/MILESTONES.md`](docs/MILESTONES.md) ·
Mimari: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Neden kendi simülatörümüz?

CARLA / LGSVL gibi simülatörler GPU ve ağır kurulum gerektirir; CI içinde,
headless ve deterministik koşamaz. MogaSim:

* saf Python + NumPy/SciPy, GPU gerektirmez, gerçek zamandan hızlı çalışır,
* her koşu tohumla (seed) tekrarlanabilir → regresyon testi yapılabilir,
* sensörleri ham veri seviyesinde modeller (GNSS lat/lon, IMU, LiDAR nokta
  bulutu, kamera görüntüsü, radar) – otonomi yığını yer gerçeğini göremez,
* hata enjeksiyonu (sensör kaybı, GNSS kesintisi, aktüatör arızası, hava) ile
  Seviye 4 geri dönüş davranışlarını test etmeye olanak verir.

Otonomi yığını simülatörden bağımsızdır; mesaj arayüzü ROS 2 tiplerine
benzer şekilde tasarlandı, böylece ileride ROS 2 / gerçek araç donanımına
köprülenebilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
# senaryoları listele
python scripts/run_scenario.py --list

# tek senaryo + GIF / genel bakış görseli
python scripts/run_scenario.py m1_urban_route --gif out/urban.gif --overview out/urban.png

# bir milestone'un tüm senaryoları + rapor
python scripts/run_scenario.py --milestone M0 --driver oracle --report out/report.md

# testler
python -m pytest -q
```

## Durum

| Milestone | Durum |
|-----------|-------|
| M0 – Simülatör çekirdeği | ✅ |
| M1 – Lokalizasyon + GPS tabanlı sürüş | ✅ |
| M2 – Kamera şerit algılama + şerit takibi | ⏳ |
| M3 – LiDAR/radar algılama + nesne takibi | ⏳ |
| M4 – Davranış + hareket planlama (şerit değiştirme, sollama, AEB) | ⏳ |
| M5 – Şehir içi sürüş (kavşak, ışık, yaya) | ⏳ |
| M6 – Seviye 4 güvenlik (arıza, ODD, MRM) | ⏳ |

## Görseller

| M0 – otoyol trafiği (oracle) | M0 – sinyalize şehir ızgarası |
|---|---|
| ![](docs/images/m0_highway.png) | ![](docs/images/m0_urban_overview.png) |
| **M1 – GPS görevi, şerit seviyesinde rota** | **M1 – tünelde ölü hesap (GNSS yok)** |
| ![](docs/images/m1_urban_route.png) | ![](docs/images/m1_tunnel_localization.png) |

## Araç varsayımları

Gerçek Moga parametreleri `moga/common/vehicle_params.py` içinde tek yerde
toplanmıştır (boyutlar, kütle, lastik, direksiyon/fren aktüatörleri, 150 kW
motor, 60 kWh batarya, sensör montaj konumları). Gerçek ölçümler geldiğinde
bu dosya güncellenmelidir.
