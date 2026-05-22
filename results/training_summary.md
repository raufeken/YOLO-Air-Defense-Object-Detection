# YOLO11n Balanced Training Summary

## Training Run

- Run: `results`
- Model: `yolo11n`
- Dataset: balanced air defense dataset
- Epochs: `20`
- Workers: `4`
- Batch: `auto`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

## Training Validation Metrics

| class | precision | recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| all | 0.822 | 0.692 | 0.750 | 0.553 |
| missile | 0.774 | 0.619 | 0.680 | 0.401 |
| uav | 0.864 | 0.490 | 0.579 | 0.355 |
| helicopter | 0.915 | 0.939 | 0.971 | 0.806 |
| fighter_aircraft | 0.736 | 0.719 | 0.770 | 0.650 |

## Test Metrics

| class | precision | recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| all | 0.789 | 0.669 | 0.721 | 0.521 |
| missile | 0.750 | 0.695 | 0.739 | 0.453 |
| uav | 0.823 | 0.458 | 0.529 | 0.326 |
| helicopter | 0.896 | 0.868 | 0.920 | 0.735 |
| fighter_aircraft | 0.687 | 0.652 | 0.697 | 0.572 |

## Missile Performance

- Validation missile mAP@0.5: `0.680`
- Validation missile mAP@0.5:0.95: `0.401`
- Test missile mAP@0.5: `0.739`
- Test missile mAP@0.5:0.95: `0.453`

## Main Artifacts

- Best checkpoint: `weights\best.pt`
- Training results CSV: `results\results.csv`
- Training results plot: `results\results.png`
- Training confusion matrix: `results\confusion_matrix.png`
- Training PR curve: `results\BoxPR_curve.png`
- Test metrics JSON: `results\test_eval_metrics.json`

