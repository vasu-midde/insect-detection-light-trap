| Metric | Tiled (1/2 res, 1280 px tiles) | Whole frame (1/8 res) |
|---|---|---|
| Test frames / insects | 24 / 526 | 24 / 526 |
| Threshold (chosen on val) | 0.75 | 0.59 |
| AP50 | 0.610 | 0.798 |
| AP50:95 | 0.285 | 0.368 |
| Precision | 0.821 | 0.918 |
| Recall | 0.540 | 0.787 |
| F1 | 0.651 | 0.847 |
| Count MAE (insects/frame) | 7.50 | 3.54 |
| Count bias (pred - true) | -7.50 | -3.12 |
| Total count true / pred | 526 / 346 | 526 / 451 |
| Seconds per frame | 5.92 | 0.10 |
| Recall, small <100px | 0.41 (n=319) | 0.67 (n=319) |
| Recall, medium 100-300px | 0.75 (n=116) | 0.96 (n=116) |
| Recall, large >300px | 0.74 (n=91) | 0.99 (n=91) |
