# Dataset Information

Bu çalışmada kullanılan veri yapısı YOLO nesne tespiti formatındadır.

Beklenen klasör yapısı:

```text
air_defense_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Her görsel dosyası için aynı isimde bir `.txt` label dosyası bulunur. Label formatı:

```text
class_id x_center y_center width height
```

Sınıflar:

```text
0: missile
1: uav
2: helicopter
3: fighter_aircraft
```
