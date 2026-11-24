# Kelpod benchmark on synthetic plans

2026-11-23 20:30 UTC, 10 plans, degrade 0.5, OCR `rapidocr`, Linux x86_64, Python 3.12.16. Synthetic plans only: these numbers say nothing about real scans.

| seed | rooms | ok | closed | node cl. | room cl. | room IoU | room P/R | wall F1 | doors | windows | area err | total s | clean | segment | ocr | geometry | model | export | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | yes | yes | 1.00 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 4.6 | 0.3 | 0.2 | 3.5 | 0.0 | 0.0 | 0.6 |  |
| 1 | 4 | yes | yes | 1.00 | 1.00 | 0.99 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.001 | 4.5 | 0.3 | 0.3 | 3.4 | 0.0 | 0.0 | 0.5 |  |
| 2 | 3 | yes | no | 0.78 | 1.00 | 0.66 | 1.00/0.67 | 0.99 | 1.00 | 1.00 | 0.000 | 3.7 | 0.3 | 0.3 | 2.7 | 0.0 | 0.0 | 0.4 |  |
| 3 | 7 | yes | yes | 1.00 | 1.00 | 0.99 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.002 | 5.3 | 0.4 | 0.4 | 3.7 | 0.0 | 0.0 | 0.8 |  |
| 4 | 8 | yes | yes | 1.00 | 1.00 | 0.98 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.001 | 6.5 | 0.4 | 0.4 | 4.9 | 0.0 | 0.0 | 0.8 |  |
| 5 | 6 | yes | yes | 1.00 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 4.9 | 0.4 | 0.4 | 3.6 | 0.0 | 0.0 | 0.6 |  |
| 6 | 6 | yes | yes | 1.00 | 1.00 | 0.99 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 4.9 | 0.4 | 0.4 | 3.6 | 0.0 | 0.0 | 0.6 |  |
| 7 | 5 | yes | yes | 1.00 | 1.00 | 0.98 | 1.00/1.00 | 1.00 | 1.00 | 0.75 | 0.009 | 4.2 | 0.3 | 0.3 | 3.0 | 0.0 | 0.0 | 0.6 |  |
| 8 | 5 | yes | yes | 1.00 | 1.00 | 0.99 | 1.00/1.00 | 1.00 | 1.00 | 1.00 | 0.001 | 3.8 | 0.3 | 0.3 | 2.7 | 0.0 | 0.0 | 0.5 |  |
| 9 | 7 | yes | no | 0.82 | 1.00 | 0.64 | 1.00/0.71 | 0.99 | 0.86 | 0.83 | 0.190 | 4.7 | 0.4 | 0.4 | 3.2 | 0.0 | 0.0 | 0.6 |  |

## Summary

- plans: 10
- pipeline_ok: 10
- failures_by_step: none
- closed_share: 0.800
- node_closure_mean: 0.960
- room_closure_mean: 1.000
- room_iou_mean: 0.923
- room_recall_mean: 0.938
- wall_f1_mean: 0.997
- door_recall_mean: 0.986
- window_recall_mean: 0.958
- room_area_error_mean: 0.020
- total_s_mean: 4.703
- total_s_max: 6.536
