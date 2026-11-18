# Species classifier (ResNet-18, ImageNet-pretrained)

Best epoch 8 of 25 (val macro-F1 0.987). Test: accuracy **0.971**, macro-F1 **0.968** on 140 crops.
Crops train / val / test: 575 / 153 / 140; 239 val/test crops dropped (moth also in another split).
Merged into 'other insect' (< 8 moths): Chiasmia, Drepaninae, Heliothis sp., Lomaspilis marginata, Morphospecies 1, Morphospecies 2, Morphospecies 3

| Class | Test crops | Precision | Recall | F1 |
|---|---|---|---|---|
| Deltote deceptoria | 19 | 0.95 | 1.00 | 0.97 |
| Eilema sp. | 6 | 0.86 | 1.00 | 0.92 |
| Plagodis | 9 | 1.00 | 1.00 | 1.00 |
| Thaumetopoea processionea | 47 | 1.00 | 0.96 | 0.98 |
| other insect | 59 | 0.97 | 0.97 | 0.97 |
