# Pooled model (trained on all three sites) vs per-site models

Each site is scored on its own 24 test frames; thresholds chosen on that site's val frames.

## tiled

| Test metric | Station_1st_night per-site | Station_1st_night pooled | Station_2nd_night per-site | Station_2nd_night pooled | OAK per-site | OAK pooled |
|---|---|---|---|---|---|---|
| AP50 | 0.678 | 0.610 | 0.850 | 0.860 | 0.840 | 0.831 |
| F1 | 0.753 | 0.651 | 0.873 | 0.888 | 0.842 | 0.848 |
| Recall | 0.650 | 0.540 | 0.853 | 0.892 | 0.789 | 0.793 |
| Precision | 0.895 | 0.821 | 0.894 | 0.884 | 0.902 | 0.911 |
| Count MAE / frame | 6.25 | 7.50 | 1.33 | 0.83 | 1.62 | 1.50 |
| Count bias / frame | -6.00 | -7.50 | -0.83 | +0.17 | -1.29 | -1.33 |
| Recall small <100 px | 0.48 | 0.41 | 0.86 | 0.91 | 0.64 | 0.67 |

## whole frame

| Test metric | Station_1st_night per-site | Station_1st_night pooled | Station_2nd_night per-site | Station_2nd_night pooled | OAK per-site | OAK pooled |
|---|---|---|---|---|---|---|
| AP50 | 0.810 | 0.798 | 0.940 | 0.935 | 0.874 | 0.853 |
| F1 | 0.810 | 0.847 | 0.932 | 0.935 | 0.876 | 0.865 |
| Recall | 0.736 | 0.787 | 0.938 | 0.936 | 0.874 | 0.850 |
| Precision | 0.900 | 0.918 | 0.925 | 0.933 | 0.878 | 0.882 |
| Count MAE / frame | 4.00 | 3.54 | 0.92 | 0.54 | 1.38 | 1.46 |
| Count bias / frame | -4.00 | -3.12 | +0.25 | +0.04 | -0.04 | -0.38 |
| Recall small <100 px | 0.61 | 0.67 | 0.94 | 0.94 | 0.76 | 0.72 |

