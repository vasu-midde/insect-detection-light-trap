| Metric | Tiled (1/2 res, 1280 px tiles) | Whole frame (1/8 res) |
|---|---|---|
| Test frames / insects | 24 / 526 | 24 / 526 |
| Threshold (chosen on val) | 0.60 | 0.86 |
| AP50 | 0.678 | 0.810 |
| AP50:95 | 0.307 | 0.356 |
| Precision | 0.895 | 0.900 |
| Recall | 0.650 | 0.736 |
| F1 | 0.753 | 0.810 |
| Count MAE (insects/frame) | 6.25 | 4.00 |
| Count bias (pred - true) | -6.00 | -4.00 |
| Total count true / pred | 526 / 382 | 526 / 430 |
| Seconds per frame | 5.46 | 0.10 |
| Recall, small <100px | 0.48 (n=319) | 0.61 (n=319) |
| Recall, medium 100-300px | 0.91 (n=116) | 0.93 (n=116) |
| Recall, large >300px | 0.92 (n=91) | 0.95 (n=91) |
