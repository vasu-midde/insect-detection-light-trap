| Metric | Tiled (1/2 res, 1280 px tiles) | Whole frame (1/8 res) |
|---|---|---|
| Test frames / insects | 24 / 435 | 24 / 435 |
| Threshold (chosen on val) | 0.73 | 0.55 |
| AP50 | 0.860 | 0.935 |
| AP50:95 | 0.439 | 0.486 |
| Precision | 0.884 | 0.933 |
| Recall | 0.892 | 0.936 |
| F1 | 0.888 | 0.935 |
| Count MAE (insects/frame) | 0.83 | 0.54 |
| Count bias (pred - true) | +0.17 | +0.04 |
| Total count true / pred | 435 / 439 | 435 / 436 |
| Seconds per frame | 5.80 | 0.10 |
| Recall, small <100px | 0.91 (n=339) | 0.94 (n=339) |
| Recall, medium 100-300px | 0.75 (n=60) | 0.85 (n=60) |
| Recall, large >300px | 1.00 (n=36) | 1.00 (n=36) |
