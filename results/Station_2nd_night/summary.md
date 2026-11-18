| Metric | Tiled (1/2 res, 1280 px tiles) | Whole frame (1/8 res) |
|---|---|---|
| Test frames / insects | 24 / 435 | 24 / 435 |
| Threshold (chosen on val) | 0.65 | 0.52 |
| AP50 | 0.850 | 0.940 |
| AP50:95 | 0.427 | 0.421 |
| Precision | 0.894 | 0.925 |
| Recall | 0.853 | 0.938 |
| F1 | 0.873 | 0.932 |
| Count MAE (insects/frame) | 1.33 | 0.92 |
| Count bias (pred - true) | -0.83 | +0.25 |
| Total count true / pred | 435 / 415 | 435 / 441 |
| Seconds per frame | 5.09 | 0.11 |
| Recall, small <100px | 0.86 (n=339) | 0.94 (n=339) |
| Recall, medium 100-300px | 0.75 (n=60) | 0.90 (n=60) |
| Recall, large >300px | 1.00 (n=36) | 1.00 (n=36) |
