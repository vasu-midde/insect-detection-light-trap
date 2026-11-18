| Metric | Tiled (1/2 res, 1280 px tiles) | Whole frame (1/8 res) |
|---|---|---|
| Test frames / insects | 24 / 246 | 24 / 246 |
| Threshold (chosen on val) | 0.72 | 0.21 |
| AP50 | 0.831 | 0.853 |
| AP50:95 | 0.493 | 0.502 |
| Precision | 0.911 | 0.882 |
| Recall | 0.793 | 0.850 |
| F1 | 0.848 | 0.865 |
| Count MAE (insects/frame) | 1.50 | 1.46 |
| Count bias (pred - true) | -1.33 | -0.38 |
| Total count true / pred | 246 / 214 | 246 / 237 |
| Seconds per frame | 5.83 | 0.10 |
| Recall, small <100px | 0.67 (n=130) | 0.72 (n=130) |
| Recall, medium 100-300px | 1.00 (n=57) | 1.00 (n=57) |
| Recall, large >300px | 0.86 (n=59) | 1.00 (n=59) |
