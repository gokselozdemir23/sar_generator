# Synthetic SAR Log Data Generator for Telco Cloud

Üretim Telco Cloud ortamlarını (OpenStack / CEPH) taklit eden gerçekçi SAR (System Activity Report) log verisi üreten, yapılandırılabilir sentetik veri jeneratörü. Performans testi, izleme sistemi geliştirme ve ML model eğitimi için üretim verisine erişim gerektirmeden kullanılabilir.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Kurulum](#kurulum)
- [Web Arayüzü (Streamlit GUI)](#web-arayüzü-streamlit-gui)
- [Yapılandırma](#yapılandırma)
  - [YAML Yapılandırma Örneği](#yaml-yapılandırma-örneği)
  - [Node Tipleri](#node-tipleri)
  - [Anomali Senaryoları](#anomali-senaryoları)
  - [Veri Kalitesi Seviyeleri](#veri-kalitesi-seviyeleri)
  - [Çıktı Yapılandırması](#çıktı-yapılandırması)
- [CLI Referansı](#cli-referansı)
- [Çıktı Formatları](#çıktı-formatları)
  - [CSV (SAR Uyumlu)](#csv-sar-uyumlu)
  - [JSON (Düz)](#json-düz)
  - [JSON (Nested / Dokümantasyon Formatı)](#json-nested--dokümantasyon-formatı)
  - [NDJSON (Streaming Pipelines)](#ndjson-streaming-pipelines)
- [Veritabanı Yazıcıları](#veritabanı-yazıcıları)
- [Gerçek Zamanlı Streaming](#gerçek-zamanlı-streaming)
- [Dosya Rotasyonu ve Sıkıştırma](#dosya-rotasyonu-ve-sıkıştırma)
- [Doğrulama ve Görselleştirme](#doğrulama-ve-görselleştirme)
- [Referans SAR Karşılaştırması](#referans-sar-karşılaştırması)
- [Performans Ölçümü](#performans-ölçümü)
- [Test](#test)
- [Docker ile Kullanım](#docker-ile-kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Mimari Genel Bakış](#mimari-genel-bakış)
- [SAR Metrikleri Referansı](#sar-metrikleri-referansı)
- [Teknoloji Yığını](#teknoloji-yığını)

---

## Özellikler

**Veri Üretim Motoru**
- 132 SAR kolon desteği ile tam uyumlu çıktı
- 4 node tipi: Compute, CEPH Storage, Control Plane, Network
- Her node tipi için ayrı istatistiksel profiller (ortalama, standart sapma, min/max sınırları)
- 14 çapraz metrik korelasyon kuralı (CPU↔Memory, Disk↔Network, Load cascades)
- AR(1) otokorelasyon ile gerçekçi zaman serisi üretimi
- Diurnal (gündüz/gece) ve weekly (hafta içi/sonu) pattern jeneratörleri
- Yapılandırılabilir gürültü seviyesi ve veri kalitesi presetleri

**Anomali ve Senaryo Motoru**
- 7 anomali senaryosu: Storage contention, memory pressure, network saturation, CPU steal spike, cascading failure, gradual degradation, backup storm
- Ramp-up / ramp-down envelope ile yumuşak geçişler
- Yapılandırılabilir anomali sıklığı (none / low / medium / high) ve şiddet seviyesi (low / medium / high / critical)
- Node-bazlı rastgele mikro anomali enjeksiyonu

**Çıktı Adaptörleri**
- CSV (SAR uyumlu kolon sırası ve formatlama)
- JSON (düz ve nested/gruplu formatlar)
- NDJSON (streaming pipeline'lar için)
- Node bazlı veya birleşik çıktı
- gzip sıkıştırma desteği

**Veritabanı Yazıcıları**
- InfluxDB v2 (line protocol, batch yazma)
- PostgreSQL (otomatik tablo oluşturma, connection pool)
- Prometheus exporter (HTTP metrics endpoint)

**Gerçek Zamanlı Streaming**
- stdout (satır satır)
- TCP socket (multi-client broadcast)
- WebSocket (asyncio tabanlı)

**Dosya Rotasyonu**
- Boyut bazlı (max_mb aşılınca yeni dosya)
- Zaman bazlı (interval_minutes geçince yeni dosya)
- Hibrit strateji (ikisi birden)
- Eski dosya temizleme (keep_last_n)

**Doğrulama ve Analiz**
- İstatistiksel doğrulama (dağılım, korelasyon, anomali frekansı kontrolleri)
- Gerçek SAR referans profilleriyle karşılaştırma (built-in veya özel referans)
- Kolmogorov-Smirnov dağılım testi
- matplotlib görselleştirme (CPU, memory, network, disk I/O, anomaly boxplot)
- Null ve range kontrolleri

**Performans**
- >1M satır/dakika throughput (vektörize NumPy işlemleri)
- Chunk bazlı üretim ile bellek verimli büyük veri setleri
- Performans benchmark aracı (rows/s, peak/avg memory, chunk p50/p95)

**Test Altyapısı**
- pytest birim testleri (generator, output, anomaly, config)
- Hypothesis ile property-based testler (rastgele konfigürasyonlarla invariant doğrulama)
- Performans kriteri testleri (100 node × 7 gün < 5 dakika, >1M rows/min)
- Karşılaştırma modülü testleri
- Genişletilmiş modül testleri (database, streaming, rotation, validation, benchmark)

**Web Arayüzü (Streamlit GUI)**
- Tarayıcı üzerinden tam özellikli görsel arayüz (`app.py`)
- Sidebar'dan dinamik node grubu ve senaryo ekleme/silme
- 5 sekmeli ana içerik: Overview, Results, Visualize, Validate, Download
- Tüm grafikler interaktif host seçici ile anlık render (önbellek destekli)
- Tarayıcıdan CSV ve JSON indirme + diske kaydetme
- İstatistiksel doğrulama ve referans karşılaştırması sonuçlarını tablo olarak görüntüleme
- CLI (`main.py`) tamamen korunmuş; iki arayüz birbirinden bağımsız

**Dağıtım**
- Docker container (Dockerfile + docker-compose.yml)
- PyPI-ready paketleme (pyproject.toml)
- Yapılandırılabilir opsiyonel bağımlılıklar (viz, db, streaming, test)

---

## Hızlı Başlangıç

### CLI (Komut Satırı)

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Built-in varsayılan yapılandırma ile üret (1 hafta, 18 node)
python main.py

# Özel yapılandırma ile üret
python main.py -c config.yaml

# Örnek yapılandırma dosyası oluştur
python main.py --write-example-config

# Yapılandırmayı doğrula (üretim yapmadan)
python main.py --validate-only -c config.yaml

# Sadece CSV çıktı, belirli dizine
python main.py -f csv -o ./my_output

# Node bazlı ayrı dosyalar
python main.py --by-node

# Detaylı log
python main.py -v
```

### Web Arayüzü (Streamlit GUI)

```bash
# Streamlit kurulumu (requirements.txt içinde zaten mevcut)
pip install -r requirements.txt

# Arayüzü başlat
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresi otomatik olarak açılır.

---

## Kurulum

### Kaynaktan

```bash
git clone <repository-url>
cd sar_generator
pip install -r requirements.txt
```

### Opsiyonel Bağımlılıklarla

```bash
# Tüm opsiyonel bağımlılıklar
pip install -e ".[all]"

# Sadece görselleştirme
pip install -e ".[viz]"

# Sadece veritabanı adaptörleri
pip install -e ".[db]"

# Sadece test araçları
pip install -e ".[test]"
```

### Gerekli Bağımlılıklar

| Paket       | Minimum Versiyon | Amaç                              |
|-------------|------------------|------------------------------------|
| numpy       | ≥ 1.24.0         | Vektörize veri üretimi             |
| pandas      | ≥ 2.0.0          | DataFrame işlemleri ve CSV I/O     |
| scipy       | ≥ 1.10.0         | İstatistiksel dağılımlar ve testler|
| pydantic    | ≥ 2.0.0          | Yapılandırma şema doğrulama        |
| PyYAML      | ≥ 6.0            | YAML yapılandırma dosyası okuma    |
| click       | ≥ 8.0            | CLI arayüzü                        |

### Opsiyonel Bağımlılıklar

| Paket              | Amaç                          |
|--------------------|-------------------------------|
| matplotlib         | Görselleştirme (`--visualize`)|
| streamlit          | Web arayüzü (`app.py`)        |
| influxdb-client    | InfluxDB v2 yazıcı            |
| psycopg2-binary    | PostgreSQL yazıcı              |
| prometheus-client  | Prometheus exporter            |
| websockets         | WebSocket streaming            |
| hypothesis         | Property-based testler         |
| pytest             | Test framework                 |

---

## Web Arayüzü (Streamlit GUI)

`app.py`, CLI (`main.py`) ile tamamen aynı işlevselliği sunan görsel bir web arayüzüdür. İki arayüz birbirinden bağımsızdır; CLI kodu değiştirilmemiştir.

### Başlatma

```bash
streamlit run app.py
# → http://localhost:8501
```

### Arayüz Yapısı

```
┌──────────────────────────┬──────────────────────────────────────────────────┐
│  SIDEBAR (Yapılandırma)  │  ANA İÇERİK                                      │
│                          │                                                  │
│  📅 Simulation Period    │  Sekmeler:                                        │
│  🔧 Advanced Settings    │  ┌──────────────────────────────────────────────┐│
│  🖥 Node Groups          │  │ 📋 Overview                                  ││
│     [+ Add / Remove]     │  │   Node grubu + senaryo özet tablosu          ││
│  ⚡ Anomaly Scenarios    │  │   Tahminî satır sayısı, süre, metrik kartlar  ││
│     [+ Add / Remove]     │  ├──────────────────────────────────────────────┤│
│  📁 Output Settings      │  │ 📊 Results                                   ││
│                          │  │   DataFrame önizleme (host filtreli)         ││
│  [🚀 Generate SAR Data]  │  │   Temel istatistikler tablosu                ││
│                          │  ├──────────────────────────────────────────────┤│
│                          │  │ 📈 Visualize                                 ││
│                          │  │   Host seçici + grafik seçici                ││
│                          │  │   CPU / Memory / Network / Disk / Anomaly    ││
│                          │  ├──────────────────────────────────────────────┤│
│                          │  │ 🔍 Validate                                  ││
│                          │  │   İstatistiksel doğrulama (MetricCheck,      ││
│                          │  │   CorrelationCheck, AnomalyCheck)            ││
│                          │  │   Referans SAR karşılaştırması (KS testi)    ││
│                          │  ├──────────────────────────────────────────────┤│
│                          │  │ ⬇ Download                                  ││
│                          │  │   Tarayıcıdan CSV / JSON indirme             ││
│                          │  │   Diske kaydetme (output dizinine)           ││
│                          │  └──────────────────────────────────────────────┘│
└──────────────────────────┴──────────────────────────────────────────────────┘
```

### Özellikler

| Özellik | Açıklama |
|---|---|
| Dinamik node grupları | Sidebar'dan istediğiniz kadar node grubu ekleyin veya silin |
| Dinamik senaryolar | Anomali senaryolarını anında ekleyin, yapılandırın, kaldırın |
| Anlık önizleme | Yapılandırma değiştiğinde Overview sekmesi otomatik güncellenir |
| Önbellekli grafikler | Aynı veri seti için grafikler yeniden üretilmez (`@st.cache_data`) |
| İndirme | Tarayıcıdan tek tıkla CSV / JSON indirme |
| Diske yazma | `OutputManager` ile yapılandırılmış output dizinine kaydetme |
| Doğrulama | `StatisticalValidator` ve `SARPatternComparator` sonuçlarını tablo olarak gösterir |

### CLI ile Karşılaştırma

| İşlem | CLI | GUI |
|---|---|---|
| Veri üretme | `python main.py -c config.yaml` | Sidebar → Generate |
| Görselleştirme | `python main.py --visualize file.csv` | Visualize sekmesi |
| İstatistiksel doğrulama | `python main.py --validate-data file.csv` | Validate sekmesi |
| Referans karşılaştırması | `python main.py --compare-reference file.csv` | Validate sekmesi |
| CSV indirme | Dosya sistemi | Download sekmesi → tarayıcı |
| Yapılandırma | YAML dosyası | Sidebar widget'ları |

---

## Yapılandırma

### YAML Yapılandırma Örneği

```yaml
simulation:
  start_time: "2024-01-01 00:00:00"
  end_time: "2024-01-07 23:59:59"
  interval_seconds: 300          # 5 dakikada bir örnekleme
  random_seed: 42                # Tekrarlanabilir sonuçlar
  diurnal_pattern: true          # Gündüz/gece paterni
  weekly_pattern: true           # Hafta içi/sonu paterni
  noise_level: 0.05              # Gürültü seviyesi (0.0 - 1.0)
  data_quality: normal           # clean | normal | noisy | degraded
  description: "Telco Cloud 1-week simulation"

  nodes:
    - type: "ceph_storage"
      count: 3
      base_load: 0.6
      anomaly_frequency: "low"
      total_memory_kb: 262144000
      cpu_count: 32
      disk_count: 24

    - type: "compute"
      count: 10
      base_load: 0.4
      anomaly_frequency: "medium"
      total_memory_kb: 131072000
      cpu_count: 64
      disk_count: 4

    - type: "control_plane"
      count: 3
      base_load: 0.3
      anomaly_frequency: "low"
      total_memory_kb: 65536000
      cpu_count: 16
      disk_count: 2

    - type: "network"
      count: 2
      base_load: 0.35
      anomaly_frequency: "low"
      total_memory_kb: 32768000
      cpu_count: 8
      disk_count: 2

  scenarios:
    - type: "storage_contention"
      start_time: "2024-01-03 14:00:00"
      duration_hours: 2
      severity: "high"
      target_node_types: ["ceph_storage", "compute"]
      ramp_up_minutes: 5
      ramp_down_minutes: 5

    - type: "memory_pressure"
      start_time: "2024-01-05 02:00:00"
      duration_hours: 3
      severity: "medium"
      target_node_types: ["compute"]

    - type: "backup_storm"
      start_time: "2024-01-06 01:00:00"
      duration_hours: 4
      severity: "medium"

  output:
    format: "both"               # csv | json | both
    output_dir: "./output"
    filename_prefix: "sar_synthetic"
    compress: false
    chunk_size: 100000
    include_header: true

  # Opsiyonel: Veritabanı çıktısı
  database:
    influxdb:
      enabled: false
      host: localhost
      port: 8086
      token: "my-token"
      org: "my-org"
      bucket: "sar_metrics"
      batch_size: 5000
    postgresql:
      enabled: false
      host: localhost
      port: 5432
      database: sar_metrics
      user: postgres
      password: secret
      table: sar_data
    prometheus:
      enabled: false
      port: 9100
      prefix: sar

  # Opsiyonel: Gerçek zamanlı streaming
  streaming:
    enabled: false
    mode: stdout                 # stdout | socket | websocket
    host: 0.0.0.0
    port: 9000
    record_format: json          # json | csv
    flush_interval_rows: 100
```

### Node Tipleri

| Node Tipi        | Açıklama                                 | Kritik SAR Metrikleri                          |
|------------------|------------------------------------------|-----------------------------------------------|
| `compute`        | OpenStack Nova hypervisor'ları, VM/VNF çalıştırır | `%steal`, `%iowait`, `kbmemused`, `txkB/s`  |
| `ceph_storage`   | CEPH OSD node'ları, dağıtık depolama      | `await`, `svctm`, `%util`, `tps`, `bread/s`, `bwrtn/s` |
| `control_plane`  | OpenStack controller servisleri            | `CPU`, `ldavg-1`, `%iowait`                  |
| `network`        | Neutron ağ node'ları (OVS, DPDK)          | `rxpck/s`, `txpck/s`, `%soft`, `%ifutil`     |

Her node tipi için tanımlı çapraz metrik korelasyonlar:

| Kaynak Metrik  | Hedef Metrik  | Node Tipi        | Açıklama                                      |
|----------------|---------------|------------------|-----------------------------------------------|
| `kbmemused`    | `fault/s`     | Compute          | Yüksek bellek → sayfa hataları                 |
| `fault/s`      | `%sys`        | Compute          | Sayfa hataları → CPU sys                       |
| `%steal`       | `%iowait`     | Compute          | Steal → IO bekleme                             |
| `bwrtn/s`      | `txkB/s`      | Compute, CEPH    | Depolama yazmaları → ağ TX (replikasyon)       |
| `%util`        | `await`       | CEPH             | Disk kullanımı → bekleme süresi                |
| `tps`          | `%util`       | CEPH             | IOPS → disk kullanımı                          |
| `await`        | `%iowait`     | CEPH             | Disk gecikme → CPU iowait                      |
| `ldavg-1`      | `%iowait`     | Control Plane    | Yüksek yük → IO bekleme                       |
| `tps`          | `ldavg-1`     | Control Plane    | Disk IOPS → yük ortalaması                    |
| `rxpck/s`      | `%soft`       | Network          | RX paketleri → softirq CPU                    |
| `%soft`        | `%sys`        | Network          | Softirq → sys CPU                             |
| `%ifutil`      | `rxdrop/s`    | Network          | Arayüz doygunluğu → paket düşüşleri          |

### Anomali Senaryoları

| Senaryo                  | Açıklama                                                        | Etkilenen Metrikler                                    |
|--------------------------|-----------------------------------------------------------------|-------------------------------------------------------|
| `storage_contention`     | Depolama çekişmesi: yüksek await, %util, tps                   | `%util`, `await`, `svctm`, `tps`, `bwrtn/s`, `%iowait`, `txkB/s` |
| `memory_pressure`        | Bellek baskısı: sayfa taraması, swap kullanımı, hata artışı     | `kbmemused`, `pgscand/s`, `fault/s`, `majflt/s`, `%sys`, `kbswpused` |
| `network_saturation`     | Ağ doygunluğu: paket oranları tavan, düşüşler başlar            | `rxpck/s`, `txpck/s`, `%ifutil`, `rxdrop/s`, `%soft`  |
| `cpu_steal_spike`        | CPU steal artışı: hypervisor çekişmesi                          | `%steal`, `%iowait`, `ldavg-1`, `await`                |
| `cascading_failure`      | Kademeli çökme: CPU→Memory→Storage sırayla bozulur              | Faz 1: `%steal`; Faz 2: `fault/s`, `%sys`; Faz 3: `await`, `%util` |
| `gradual_degradation`    | Yavaş bozulma: metrikler zaman içinde kötüleşir                | `%iowait`, `%sys`, `await`, `ldavg-1`, `fault/s`      |
| `backup_storm`           | Yedekleme fırtınası: masif okuma I/O, yüksek ağ trafiği         | `rtps`, `bread/s`, `%iowait`, `%util`, `txkB/s`       |

Her senaryo için yapılandırılabilir parametreler:
- `severity`: low (×1.5), medium (×3.0), high (×6.0), critical (×12.0)
- `ramp_up_minutes` / `ramp_down_minutes`: Yumuşak geçiş süresi
- `target_node_types`: Hedef node tipleri (null ise tümü)

### Veri Kalitesi Seviyeleri

| Seviye     | Gürültü | AR(1) Katsayısı | Açıklama                                    |
|------------|---------|-----------------|---------------------------------------------|
| `clean`    | 0.01    | 0.9             | Minimal gürültü, düzgün sinyaller            |
| `normal`   | config  | 0.7             | Varsayılan gerçekçi davranış                  |
| `noisy`    | 0.15    | 0.4             | Yüksek gürültü, sık mikro anomaliler          |
| `degraded` | 0.10    | 0.5             | Sistematik sapma ve artefaktlar               |

`normal` modunda `noise_level` yapılandırma değeri kullanılır. Diğer modlarda preset değerleri `noise_level`'ı geçersiz kılar.

### Çıktı Yapılandırması

| Parametre          | Varsayılan       | Açıklama                                |
|--------------------|------------------|-----------------------------------------|
| `format`           | `csv`            | `csv`, `json` veya `both`              |
| `output_dir`       | `./output`       | Çıktı dizini                            |
| `filename_prefix`  | `sar_synthetic`  | Dosya adı ön eki                        |
| `compress`         | `false`          | gzip sıkıştırma                         |
| `chunk_size`       | `100000`         | Chunk başına satır (bellek optimizasyonu)|
| `include_header`   | `true`           | CSV başlık satırı                       |

---

## CLI Referansı

```
python main.py [SEÇENEKLER]

Seçenekler:
  -c, --config DOSYA              YAML/JSON yapılandırma dosyası
  -o, --output-dir DİZİN          Çıktı dizinini geçersiz kıl
  -f, --format [csv|json|both]    Çıktı formatını geçersiz kıl
  --write-example-config          Örnek YAML yapılandırma oluştur
  --validate-only                 Yapılandırmayı doğrula (üretim yok)
  --by-node                       Node başına ayrı dosya yaz
  --stream                        Gerçek zamanlı streaming etkinleştir
  --visualize CSV_YOLU            CSV'den grafikler üret
  --validate-data CSV_YOLU        İstatistiksel doğrulama çalıştır
  --compare-reference CSV_YOLU    Referans SAR profilleriyle karşılaştır
  --plots-dir DİZİN               Grafik çıktı dizini (varsayılan: ./plots)
  -v, --verbose                   DEBUG loglama etkinleştir
  --help                          Bu mesajı göster
```

---

## Çıktı Formatları

### CSV (SAR Uyumlu)

SAR aracının ürettiği orijinal kolon sırasına uygun, toplam 132 kolon:

```
DateTime,hostname,CPU,%usr,%nice,%sys,%iowait,%steal,%irq,%soft,%guest,%gnice,%idle,...
2024-01-01 00:00:00,ceph-01,all,15.20,0.00,8.10,5.30,0.10,0.05,2.10,0.00,0.00,69.15,...
2024-01-01 00:00:00,compute-01,all,28.50,0.10,4.80,2.10,0.60,0.10,0.80,5.20,0.00,57.80,...
```

### JSON (Düz)

Hiyerarşik yapı: metadata + node bazlı gruplanmış kayıtlar.

```json
{
  "metadata": {
    "generated_at": "2024-01-08T12:00:00",
    "total_records": 36288,
    "nodes": ["ceph-01", "ceph-02", "compute-01", "..."],
    "time_range": { "start": "2024-01-01 00:00:00", "end": "2024-01-07 23:55:00" }
  },
  "nodes": {
    "compute-01": {
      "node_type": "compute",
      "records": [
        { "DateTime": "2024-01-01 00:00:00", "%usr": 28.5, "%sys": 4.8, "..." }
      ]
    }
  }
}
```

### JSON (Nested / Dokümantasyon Formatı)

Metrikler cpu / memory / disk / network olarak gruplandırılmış format:

```json
{
  "records": [
    {
      "timestamp": "2024-01-01 00:00:00",
      "hostname": "compute-07",
      "node_type": "compute",
      "metrics": {
        "cpu": { "%usr": 25.1, "%sys": 12.4, "%iowait": 8.7, "%steal": 0.5, "%idle": 51.2 },
        "memory": { "kbmemused": 72089600, "%memused": 55.0, "kbcached": 19660800 },
        "disk": { "tps": 120.5, "await": 4.2, "%util": 28.3 },
        "network": { "rxpck/s": 8500.0, "txpck/s": 9200.0, "txkB/s": 1250.0 }
      }
    }
  ],
  "total": 36288
}
```

Nested çıktı almak için programatik API:
```python
from adapters.output import OutputManager
mgr = OutputManager(config.output)
mgr.write_nested_json(df)
```

### NDJSON (Streaming Pipelines)

Her satır bağımsız bir JSON nesnesi (newline-delimited):

```
{"DateTime":"2024-01-01 00:00:00","hostname":"ceph-01","%usr":15.2,...}
{"DateTime":"2024-01-01 00:00:00","hostname":"compute-01","%usr":28.5,...}
```

---

## Veritabanı Yazıcıları

### InfluxDB v2

```yaml
database:
  influxdb:
    enabled: true
    host: localhost
    port: 8086
    token: "my-token"
    org: "my-org"
    bucket: "sar_metrics"
    batch_size: 5000
    retries: 3
```

Her SAR satırı bir InfluxDB Point'e dönüştürülür. Measurement: hostname, Tag: hostname + CPU.

### PostgreSQL

```yaml
database:
  postgresql:
    enabled: true
    host: localhost
    port: 5432
    database: sar_metrics
    user: postgres
    password: secret
    table: sar_data
    pool_size: 5
```

Tablo otomatik oluşturulur (dynamic schema). Threaded connection pool ile batch insert.

### Prometheus

```yaml
database:
  prometheus:
    enabled: true
    port: 9100
    prefix: sar
```

`http://localhost:9100/metrics` endpoint'inde SAR metrikleri gauge olarak yayınlanır. Her hostname için ayrı label.

---

## Gerçek Zamanlı Streaming

```yaml
streaming:
  enabled: true
  mode: websocket        # stdout | socket | websocket
  host: 0.0.0.0
  port: 9000
  record_format: json    # json | csv
  flush_interval_rows: 100
  max_clients: 10
```

```bash
# Streaming ile üret
python main.py -c config.yaml --stream

# Test (stdout modu)
python main.py -c config.yaml --stream  # Terminale satır satır yazar

# Test (socket modu)
nc localhost 9000                        # Başka terminalde dinle

# Test (websocket modu)
wscat -c ws://localhost:9000             # WebSocket client ile bağlan
```

---

## Dosya Rotasyonu ve Sıkıştırma

```yaml
output:
  rotation:
    strategy: size       # size | time | hybrid
    max_mb: 100          # Boyut eşiği (MB)
    interval_minutes: 60 # Zaman eşiği (dakika)
    compress: true       # Rotasyon sonrası gzip sıkıştırma
    keep_last_n: 10      # Son N dosyayı tut, eskilerini sil
```

Programatik kullanım:

```python
from adapters.rotation import RotatingCSVAdapter, RotationConfig, RotationStrategy

cfg = RotationConfig(strategy=RotationStrategy.SIZE, max_mb=50, compress=True)
with RotatingCSVAdapter("./output", "sar_data", cfg) as writer:
    for chunk in generator.generate_chunks(50000):
        writer.write(chunk)
```

---

## Doğrulama ve Görselleştirme

### İstatistiksel Doğrulama

Üretilen verinin kalitesini 3 boyutta kontrol eder:

1. **Metrik dağılım kontrolü**: Mean/std değerleri beklenen aralıklarda mı?
2. **Çapraz metrik korelasyon**: Pearson r beklenen yönde mi?
3. **Anomali frekansı**: 3σ üzeri değerlerin oranı %0-5 aralığında mı?

```bash
python main.py --validate-data output/sar_synthetic_full.csv
```

```python
from validation.statistics import StatisticalValidator
import pandas as pd

df = pd.read_csv("output/sar_synthetic_full.csv")
validator = StatisticalValidator(df)
report = validator.run_all()
report.print_summary()

# Ek kontroller
nulls = validator.null_check()          # Null değer sayıları
ranges = validator.range_check()        # Min/max değerler
summary = validator.distribution_summary()  # describe() tablosu
```

### Görselleştirme

5 grafik tipi üretir (matplotlib gerekli):

1. **CPU Zaman Serisi**: %usr, %sys, %iowait, %steal, %idle
2. **Bellek Trendleri**: %memused, %swpused
3. **Ağ Throughput**: rxkB/s, txkB/s
4. **Disk I/O**: tps, await, %util
5. **Anomali Dağılımı**: Seçili metriklerin boxplot'u

```bash
python main.py --visualize output/sar_synthetic_full.csv
python main.py --visualize output/sar_synthetic_full.csv --plots-dir ./my_plots
```

```python
from validation.plots import generate_all_plots, plot_cpu_timeseries

# Tüm grafikleri üret
paths = generate_all_plots("output/sar_synthetic_full.csv", "./plots")

# Belirli bir host için tek grafik
plot_cpu_timeseries("output/sar_synthetic_full.csv",
                    hostname="compute-01",
                    output_dir="./plots")
```

---

## Referans SAR Karşılaştırması

Üretilen sentetik veriyi gerçek SAR referans profilleriyle karşılaştırır.

### Built-in Profiller

4 node tipi için gerçek Telco Cloud ortamlarından derlenmiş referans profiller dahildir (ortalama ve standart sapma değerleri).

```bash
python main.py --compare-reference output/sar_synthetic_full.csv
```

### Gerçek Referans Verisiyle

```python
from validation.comparison import SARPatternComparator
import pandas as pd

synthetic_df = pd.read_csv("output/sar_synthetic_full.csv")
reference_df = pd.read_csv("real_sar_data.csv")

comparator = SARPatternComparator(synthetic_df)

# Built-in profiller ile
report = comparator.compare_builtin()
report.print_summary()

# Gerçek referans ile
report = comparator.compare_with(reference_df)
report.print_summary()

# JSON çıktı
print(report.to_json())
```

Karşılaştırma yöntemleri:
- **Mean/std benzerlik testi**: Ortalama fark < %60
- **Kolmogorov-Smirnov testi**: Dağılım benzerliği

---

## Performans Ölçümü

### Benchmark CLI

```bash
python benchmark.py                              # Varsayılan: 18 node, 7 gün
python benchmark.py --nodes 100 --days 7         # 100 node, 7 gün
python benchmark.py --nodes 50 --days 1 --interval 60
python benchmark.py --json --out result.json     # JSON çıktı
```

### Çıktı Örneği

```
============================================================
  SAR Generator – Performance Benchmark
============================================================
  Config         : 100 node × 7 gün @ 300s interval
  Total rows     :      201,600
  Total time     :       12.34s
  Rows/second    :      16,340
  Peak memory    :       245.3 MB
  Avg memory     :       180.2 MB
  Chunks         :           5
  Chunk p50      :      234.5 ms
  Chunk p95      :      412.3 ms
  Performans     : ✅ Çok İyi (>500K rows/s)
============================================================
```

### Programatik API

```python
from benchmark.performance import PerformanceBenchmark

bm = PerformanceBenchmark(
    total_nodes=100,
    days=7,
    interval_seconds=300,
    warmup_runs=1,
    chunk_size=50_000,
)
result = bm.run()
result.print_report()
print(result.to_dict())
```

---

## Test

### Tüm Testleri Çalıştır

```bash
# Temel testler (hızlı, <30 saniye)
pytest tests/ -v

# Yavaş performans testleri dahil
pytest tests/ -v -m "not slow"      # Yavaşları hariç tut
pytest tests/ -v --timeout=600      # Tüm testler (yavaşlar dahil)
```

### Test Dosyaları

| Dosya                    | Kapsam                                                   |
|--------------------------|----------------------------------------------------------|
| `test_generator.py`      | Config, generator, anomaly, output adaptörleri           |
| `test_extensions.py`     | Database, streaming, rotation, validation, benchmark     |
| `test_comparison.py`     | Referans SAR karşılaştırma modülü                        |
| `test_properties.py`     | Hypothesis property-based testler (rastgele konfigürasyon)|
| `test_performance.py`    | Performans kriterleri (100 node < 5dk, >1M rows/min)     |

### Property-Based Testler

Hypothesis ile rastgele üretilen konfigürasyonlarda doğrulanan invariantlar:

- Tüm 132 SAR kolonu mevcut
- CPU yüzdeleri toplamı ~100
- Yüzde kolonları ≥ 0
- Satır sayısı = `num_intervals × toplam_node`
- `kbmemused + kbmemfree = total_memory_kb`
- Hostname sayısı = toplam node sayısı
- DateTime kolonları parse edilebilir
- Sayısal kolonlarda NaN yok

```bash
pytest tests/test_properties.py -v --hypothesis-show-statistics
```

---

## Docker ile Kullanım

### Temel Kullanım

```bash
# İmaj oluştur
docker build -t sar-generator .

# Varsayılan yapılandırma ile üret
docker run -v $(pwd)/output:/app/output sar-generator

# Özel yapılandırma ile üret
docker run -v $(pwd)/output:/app/output \
           -v $(pwd)/config.yaml:/app/config.yaml:ro \
           sar-generator -c config.yaml

# Yardım
docker run sar-generator --help
```

### Docker Compose (Veritabanı Stack'i ile)

```bash
# Sadece jeneratör
docker compose up sar-generator

# InfluxDB + PostgreSQL + Grafana ile
docker compose --profile db up

# Arka planda çalıştır
docker compose --profile db up -d
```

Servisler:
- **sar-generator**: Veri üretici
- **influxdb**: InfluxDB v2 (port 8086)
- **postgres**: PostgreSQL 16 (port 5432)
- **grafana**: Grafana (port 3000, http://localhost:3000)

---

## Proje Yapısı

```
sar_generator/
├── config.py                     # Yapılandırma yönetimi (Pydantic v2)
├── main.py                       # CLI giriş noktası (Click)
├── app.py                        # Streamlit web arayüzü (GUI)
├── benchmark.py                  # Benchmark CLI wrapper
├── requirements.txt              # Python bağımlılıkları
├── pyproject.toml                # PyPI paketleme yapılandırması
├── Dockerfile                    # Docker container tanımı
├── docker-compose.yml            # Docker Compose (DB stack dahil)
├── README.md                     # Bu dosya
│
├── engine/                       # Çekirdek üretim motoru
│   ├── generator.py              #   Zaman serisi üretimi, çapraz korelasyonlar
│   └── anomaly.py                #   Senaryo ve anomali enjeksiyonu
│
├── models/                       # Veri modelleri
│   └── node_profiles.py          #   Node tipi profilleri ve korelasyon kuralları
│
├── adapters/                     # Çıktı adaptörleri
│   ├── output.py                 #   CSV, JSON, nested JSON çıktı
│   ├── database.py               #   InfluxDB, PostgreSQL, Prometheus
│   └── rotation.py               #   Dosya rotasyonu ve sıkıştırma
│
├── streaming/                    # Gerçek zamanlı streaming
│   └── streamer.py               #   stdout, TCP socket, WebSocket
│
├── validation/                   # Doğrulama ve analiz
│   ├── statistics.py             #   İstatistiksel doğrulama
│   ├── plots.py                  #   Matplotlib görselleştirme
│   └── comparison.py             #   Referans SAR karşılaştırması
│
├── benchmark/                    # Performans ölçümü
│   └── performance.py            #   Benchmark motoru ve raporlama
│
├── compat/                       # Uyumluluk
│   └── pydantic_compat.py        #   Pydantic olmadan çalışma (stdlib)
│
├── tests/                        # Test suite
│   ├── conftest.py               #   Ortak pytest yapılandırması
│   ├── test_generator.py         #   Generator birim testleri
│   ├── test_extensions.py        #   Genişletilmiş modül testleri
│   ├── test_comparison.py        #   Karşılaştırma modülü testleri
│   ├── test_properties.py        #   Hypothesis property-based testler
│   └── test_performance.py       #   Performans kriteri testleri
│
├── docs/                         # Dokümantasyon (ek dosyalar)
├── output/                       # Varsayılan çıktı dizini
└── plots/                        # Varsayılan grafik dizini
```

---

## Mimari Genel Bakış

```
┌────────────────────────────────┬────────────────────────────────────┐
│   main.py  (CLI — Click)       │   app.py  (Streamlit Web GUI)      │
│   python main.py [OPTIONS]     │   streamlit run app.py             │
└───────────────┬────────────────┴──────────────┬─────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        config.py (Pydantic v2)                      │
│  SimulationConfig → NodeConfig → ScenarioConfig → OutputConfig      │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     engine/generator.py                              │
│  SARDataGenerator → TimeSeriesGenerator (per node)                  │
│    ├── Diurnal/weekly pattern multipliers                           │
│    ├── AR(1) autocorrelated base metrics (from node_profiles)       │
│    ├── Cross-metric correlation engine                              │
│    ├── CPU normalization (%idle = 100 - active sum)                 │
│    ├── Memory resolution (fraction → absolute kB)                   │
│    └── Data quality preset application                              │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     engine/anomaly.py                                │
│  AnomalyEngine.apply()                                              │
│    ├── Scenario handlers (7 types × severity × ramp envelope)       │
│    └── Random micro-anomaly injection (per anomaly_frequency)       │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────┬──────────────────┬───────────────────────────────┐
│  adapters/       │  streaming/      │  validation/                  │
│  output.py       │  streamer.py     │  statistics.py                │
│   ├── CSV        │   ├── stdout     │  plots.py                    │
│   ├── JSON       │   ├── TCP socket │  comparison.py               │
│   └── Nested JSON│   └── WebSocket  │                               │
│  database.py     │                  │                               │
│   ├── InfluxDB   │                  │                               │
│   ├── PostgreSQL │                  │                               │
│   └── Prometheus │                  │                               │
│  rotation.py     │                  │                               │
│   └── Size/Time  │                  │                               │
└──────────────────┴──────────────────┴───────────────────────────────┘
```

---

## SAR Metrikleri Referansı

Üretilen 132 SAR kolonu aşağıdaki kategorilerde gruplandırılmıştır:

| Kategori          | Kolonlar                                                                                   |
|-------------------|--------------------------------------------------------------------------------------------|
| **Meta**          | `DateTime`, `hostname`, `CPU`                                                              |
| **CPU**           | `%usr`, `%nice`, `%sys`, `%iowait`, `%steal`, `%irq`, `%soft`, `%guest`, `%gnice`, `%idle` |
| **Process**       | `proc/s`, `cswch/s`                                                                        |
| **Swap**          | `pswpin/s`, `pswpout/s`                                                                    |
| **Paging**        | `pgpgin/s`, `pgpgout/s`, `fault/s`, `majflt/s`, `pgfree/s`, `pgscank/s`, `pgscand/s`, `pgsteal/s`, `%vmeff` |
| **Block I/O**     | `tps`, `rtps`, `wtps`, `bread/s`, `bwrtn/s`, `frmpg/s`, `bufpg/s`, `campg/s`              |
| **Memory**        | `kbmemfree`, `kbmemused`, `%memused`, `kbbuffers`, `kbcached`, `kbcommit`, `%commit`, `kbactive`, `kbinact`, `kbdirty` |
| **Swap Memory**   | `kbswpfree`, `kbswpused`, `%swpused`, `kbswpcad`, `%swpcad`                                |
| **Huge Pages**    | `kbhugfree`, `kbhugused`, `%hugused`                                                       |
| **Kernel**        | `dentunusd`, `file-nr`, `inode-nr`, `pty-nr`                                               |
| **Load**          | `runq-sz`, `plist-sz`, `ldavg-1`, `ldavg-5`, `ldavg-15`, `blocked`                        |
| **TTY**           | `TTY`, `rcvin/s`, `xmtin/s`, `framerr/s`, `prtyerr/s`, `brk/s`, `ovrun/s`                 |
| **Disk Extended** | `rd_sec/s`, `wr_sec/s`, `avgrq-sz`, `avgqu-sz`, `await`, `svctm`, `%util`, `rkB/s`, `wkB/s`, `areq-sz`, `aqu-sz` |
| **Network**       | `rxpck/s`, `txpck/s`, `rxkB/s`, `txkB/s`, `rxcmp/s`, `txcmp/s`, `rxmcst/s`, `rxerr/s`, `txerr/s`, `coll/s`, `rxdrop/s`, `txdrop/s`, `txcarr/s`, `rxfram/s`, `rxfifo/s`, `txfifo/s`, `txmtin/s`, `%ifutil` |
| **NFS Client**    | `call/s`, `retrans/s`, `read/s`, `write/s`, `access/s`, `getatt/s`                        |
| **NFS Server**    | `scall/s`, `badcall/s`, `packet/s`, `udp/s`, `tcp/s`, `hit/s`, `miss/s`, `sread/s`, `swrite/s`, `saccess/s`, `sgetatt/s` |
| **Ext. Memory**   | `kbavail`, `kbanonpg`, `kbslab`, `kbkstack`, `kbpgtbl`, `kbvmused`                        |
| **Socket**        | `totsck`, `tcpsck`, `udpsck`, `rawsck`, `ip-frag`, `tcp-tw`                               |
| **Softnet**       | `total/s`, `dropd/s`, `squeezd/s`, `rx_rps/s`, `flw_lim/s`                                |

---

## Teknoloji Yığını

| Katman          | Teknoloji                                    |
|-----------------|----------------------------------------------|
| Dil             | Python 3.9+                                  |
| Veri İşleme     | NumPy, Pandas, SciPy                         |
| Yapılandırma    | Pydantic v2, PyYAML                          |
| CLI             | Click                                        |
| Web Arayüzü     | Streamlit ≥ 1.35                             |
| Görselleştirme  | matplotlib (opsiyonel)                        |
| Veritabanı      | influxdb-client, psycopg2, prometheus-client  |
| Streaming       | websockets (opsiyonel)                        |
| Test            | pytest, hypothesis                            |
| Dağıtım         | Docker, PyPI (pyproject.toml)                 |
