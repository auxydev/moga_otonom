# Moga Otonom – kendi bilgisayarınızda çalıştırma

GPU gerekmez; Windows 10/11, macOS ve Linux'ta çalışır.  Disk: ana ortam
≈ 350 MB, MetaDrive ortamı ≈ 900 MB.

| Bileşen | Python | Ne yapar |
|---|---|---|
| MogaSim + Moga otonomi yığını | 3.10 – 3.13 | 38 kapalı çevrim senaryo (otoyol, şehir, kavşak, yaya, arıza / MRM), canlı izleme, GIF |
| MetaDrive köprüsü (isteğe bağlı) | **3.11** (ayrı ortam) | Moga yığını bağımsız MetaDrive simülatöründe, rastgele haritalarda, isteğe bağlı 3B pencere |
| KITTI değerlendirmesi (isteğe bağlı) | ana ortam | gerçek LiDAR + kamera kaydında algılama ölçümü (veri ayrıca indirilir) |

Python'u https://www.python.org/downloads/ adresinden kurun (Windows'ta
"Add python.exe to PATH" ve "py launcher" seçili olsun).  MetaDrive için
ayrıca **Python 3.11** kurun (aynı bilgisayarda birden fazla sürüm sorun
değildir).

## 1. Tek komutla kurulum + demo

Zip'i açın, klasörün içinde:

| Sistem | Komut |
|---|---|
| Windows | `setup_and_demo.bat` dosyasına çift tıklayın (MetaDrive ile: komut satırında `setup_and_demo.bat -MetaDrive`) |
| macOS / Linux | `bash scripts/setup_and_demo.sh` (MetaDrive ile: `bash scripts/setup_and_demo.sh --metadrive`) |

Betik sırasıyla: `.venv` sanal ortamını kurar, paketleri yükler, hızlı
testleri çalıştırır (~30 s) ve bir demo senaryosu koşar (~1 dk):
**`out/demo.gif`** (animasyon) ve **`out/demo.png`** (rota + hız).
`--metadrive` ile ayrıca `.venv-metadrive` ortamını kurar ve MetaDrive'da iki
sürüş yapar (`out/metadrive/`).

## 2. Elle kurulum

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Denenecekler

Aşağıdaki komutlar etkin `.venv` ortamında, proje klasöründe çalıştırılır.

**Simülasyonu canlı izlemek** (pencere açılır; kapatınca durur):

```bash
python scripts/run_scenario.py m5_urban_mission --live              # şehir içi GPS görevi
python scripts/run_scenario.py m4_rural_bypass --live --speed 2     # 2× hızlı
python scripts/run_scenario.py m6_camera_failure_pullover --live    # kamera arızası → emniyet şeridine çekilme
```

Pencere açılmazsa matplotlib için bir GUI kütüphanesi eksiktir: Linux'ta
`sudo apt install python3-tk` (ya da `pip install PyQt5`), Homebrew Python'da
`brew install python-tk`.

**GIF / görsel üretmek**:

```bash
python scripts/run_scenario.py m4_overtake_slow_truck --gif out/sollama.gif --overview out/sollama.png
python scripts/run_scenario.py --list                     # tüm senaryolar
python scripts/run_scenario.py m4_cut_in --seed 3 --gif out/cutin.gif   # farklı rastgelelik tohumu
```

Önerilen senaryolar:

| Senaryo | Ne görülür |
|---|---|
| `m1_urban_route` | yalnızca GPS hedefiyle şehir içinde rota |
| `m2_curvy_sps_lane_keep` | kamera şerit takibi, standart GPS ile |
| `m3_acc_stop_and_go` | LiDAR/radar ile takip, dur-kalk trafik |
| `m4_overtake_slow_truck` | boşluk bekleyip sollama |
| `m4_rural_bypass` | karşı trafiği bekleyip arızalı aracı geçme |
| `m4_aeb_occluded_pedestrian` | park etmiş aracın arkasından çıkan yayaya acil fren |
| `m5_urban_mission` | trafik ışıkları, yaya geçidi, kavşaklar |
| `m6_lidar_failure` | LiDAR arızası → şerit içinde güvenli duruş |
| `m6_heavy_rain_odd_exit` | yoğun yağmur (ODD dışı) → emniyet şeridine çekilme |

**Testler ve rapor**:

```bash
python -m pytest -q -n 4                                   # tüm testler + 38 senaryo (~8 dk)
python scripts/run_scenario.py --all --report out/rapor.md # tüm senaryoların raporu
```

## 4. MetaDrive (bağımsız simülatör)

Kurulum (betik kullanmıyorsanız):

```bash
python3.11 -m venv .venv-metadrive          # Windows: py -3.11 -m venv .venv-metadrive
.venv-metadrive/bin/pip install -r requirements-metadrive.txt
.venv-metadrive/bin/pip install --no-deps metadrive-simulator==0.2.6.0
# Windows'ta .venv-metadrive\Scripts\pip ...
```

(MetaDrive 0.2.6'nın kendi beyan ettiği eski bağımlılık sürümleri Python
3.11'de kurulamadığı için iki adımda, test edilmiş sürümlerle kurulur.)

```bash
# 4 rastgele harita, trafikle; her sürüş için harita + hız grafiği
.venv-metadrive/bin/python scripts/run_metadrive.py --seeds 0 1 2 3 --map 3 --plot --out out/metadrive
# 3B pencerede izleme (OpenGL gerekir)
.venv-metadrive/bin/python scripts/run_metadrive.py --seeds 16 --map 4 --density 0.15 --render
# yol kazaları (arızalı araç, koni) ve daha yoğun trafik
.venv-metadrive/bin/python scripts/run_metadrive.py --seeds 20 21 --map 3 --accident 1.0 --plot
# belirli blok dizisi: S düz, C viraj, r giriş rampası, R çıkış rampası, X kavşak, T T-kavşak, O döner kavşak
.venv-metadrive/bin/python scripts/run_metadrive.py --seeds 5 --map SCrRX --plot
```

Sonuç her sürüş için MetaDrive'ın kendi ölçütüdür: `arrive_dest` (hedefe
varış), `crash_vehicle`, `crash_object`, `out_of_road`, `timeout`.  Bir sürüş
1–3 dakika sürer (`--workers` ile paralel).

## 5. Gerçek kayıt: KITTI (isteğe bağlı)

1. http://www.cvlibs.net/datasets/kitti/raw_data.php adresinden (ücretsiz kayıt)
   `2011_09_26_calib.zip`, `2011_09_26_drive_0015_sync.zip` ve
   `2011_09_26_drive_0015_tracklets.zip` dosyalarını indirin (0005 için aynısı).
   Zip'leri açmanız gerekmez.  Lisans: CC BY-NC-SA 3.0 (ticari olmayan kullanım).
2. Çalıştırın:

```bash
python scripts/replay_kitti.py --root /indirilen/klasör --drives 15 --figures 150 250 --out out/kitti
```

Ekrana algılama / takip / şerit metriklerini yazar; `out/kitti/` içine kamera
görüntüsü üzerinde 3B kutular + kuşbakışı LiDAR görselleri üretir.

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| `python` bulunamadı (Windows) | Python'u "Add to PATH" ile yeniden kurun ya da `py` kullanın |
| PowerShell betik çalıştırmıyor | `setup_and_demo.bat` kullanın (ExecutionPolicy'yi yalnızca o çalıştırma için aşar) |
| `--live` pencere açmıyor | GUI kütüphanesi: `python3-tk` / `PyQt5` (yukarıda) |
| MetaDrive `--render` hata veriyor | grafik sürücüsü / OpenGL; pencere olmadan (`--render` olmadan) çalışır |
| MetaDrive kurulumu numpy derlemeye çalışıyor | Python sürümü 3.11 değil; `.venv-metadrive` ortamını 3.11 ile kurun |

Ayrıntılı belgeler: `README.md`, `docs/MILESTONES.md`, `docs/ARCHITECTURE.md`,
`docs/SAFETY.md`, `docs/REAL_DATA.md`.
