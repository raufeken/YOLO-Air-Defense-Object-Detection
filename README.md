# YOLO11 Air Defense Object Detection Project

## Projenin Amacı

Bu proje, hava savunma alanında kullanılan farklı hava hedeflerini görüntü ve video üzerinde tespit etmek için hazırlanmış bir YOLO11 nesne tespit çalışmasıdır. Model; tek resim, resim klasörü, video ve canlı kamera görüntüsü üzerinde çalışacak şekilde düzenlenmiştir.

## Tespit Edilen Sınıflar

Model dört sınıfı tespit eder:

```text
0: missile
1: uav
2: helicopter
3: fighter_aircraft
```

## Veri Seti Yapısı

Veri seti YOLO formatına uygun şekilde hazırlanmıştır. Görseller ve etiket dosyaları eğitim, doğrulama ve test bölümlerine ayrılmıştır. Etiket dosyalarında her satır şu formatı kullanır:

```text
class_id x_center y_center width height
```

Koordinat değerleri 0 ile 1 arasında normalize edilmiştir. Sınıf isimleri ve örnek `data.yaml` bilgisi `dataset_info` klasöründe yer alır.

## Eğitim Süreci

Eğitim YOLO11n modeli ile yapılmıştır. Görseller 640 piksel giriş boyutunda kullanılmıştır. Eğitimde otomatik batch seçimi, 20 epoch ve 0.001 başlangıç öğrenme oranı kullanılmıştır. CUDA destekli ekran kartı bulunduğunda eğitim ve çıkarım işlemleri GPU üzerinde çalıştırılabilir.

## Kullanılan Model

Çalıştırılacak model dosyası:

```text
model/best.pt
```

Demo kodları varsayılan olarak bu modeli kullanır. Confidence değeri varsayılan olarak `0.40` ayarlanmıştır.

## Sonuçların Yorumlanması

Modelin test sonuçlarında genel mAP@0.5 değeri yaklaşık `0.721`, mAP@0.5:0.95 değeri yaklaşık `0.521` olarak ölçülmüştür. Missile sınıfında test mAP@0.5 değeri yaklaşık `0.739` olarak elde edilmiştir. Confusion matrix, PR curve, F1 curve ve eğitim sonuç grafikleri `results` ve `report_assets` klasörlerinde bulunur.

## Demo / Inference Kullanımı

Demo klasörüne girip ana menü çalıştırılır:

```powershell
cd demo
python main.py
```

Menüden tek resim, resim klasörü, tek video, video klasörü veya webcam seçilebilir. Resim dosyaları `demo/test_images`, video dosyaları `demo/test_videos` klasörüne eklenir. Çıktılar `demo/image_detected` ve `demo/video_detected` klasörlerine kaydedilir.

## Klasör Yapısı

```text
project_root/
├── README.md
├── model/
├── results/
├── demo/
├── dataset_info/
├── report_assets/
└── scripts/
```

## Notlar

- Model dosyası `model/best.pt` yolunda bulunmalıdır.
- CUDA varsa demo kodları GPU kullanır, yoksa CPU ile çalışır.
- Büyük video dosyaları çalışma süresini artırabilir.
- `results` klasörü sayısal ve görsel eğitim sonuçlarını içerir.
