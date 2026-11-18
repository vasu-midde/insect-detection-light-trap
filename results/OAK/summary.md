| Metric | Tiled (1/2 res, 1280 px tiles) | Whole frame (1/8 res) |
|---|---|---|
| Test frames / insects | 24 / 246 | 24 / 246 |
| Threshold (chosen on val) | 0.49 | 0.84 |
| AP50 | 0.840 | 0.874 |
| AP50:95 | 0.492 | 0.520 |
| Precision | 0.902 | 0.878 |
| Recall | 0.789 | 0.874 |
| F1 | 0.842 | 0.876 |
| Count MAE (insects/frame) | 1.62 | 1.38 |
| Count bias (pred - true) | -1.29 | -0.04 |
| Total count true / pred | 246 / 215 | 246 / 245 |
| Seconds per frame | 4.89 | 0.10 |
| Recall, small <100px | 0.64 (n=130) | 0.76 (n=130) |
| Recall, medium 100-300px | 1.00 (n=57) | 1.00 (n=57) |
| Recall, large >300px | 0.92 (n=59) | 1.00 (n=59) |
