# Overnight booster summary

## Stage status
| stage               | status   |   seconds |
|:--------------------|:---------|----------:|
| A_enriched_metrics  | ok       |       781 |
| B_calibration       | ok       |       607 |
| C_selective_triage  | ok       |       171 |
| D_aug_baseline      | ok       |     11167 |
| E_finetune_restored | ok       |      6244 |

## Best restorer per cell (resnet50, by QWK)
| degradation   | level   | variant   |   accuracy |   kappa_qw |
|:--------------|:--------|:----------|-----------:|-----------:|
| blur          | high    | clahe     |   0.461364 |   0.251034 |
| blur          | low     | raw       |   0.663636 |   0.649646 |
| blur          | mid     | raw       |   0.590909 |   0.449021 |
| exposure      | high    | clahe     |   0.654545 |   0.589707 |
| exposure      | low     | raw       |   0.752273 |   0.806766 |
| exposure      | mid     | raw       |   0.718182 |   0.743534 |
| noise         | high    | ddpm_path |   0.577273 |   0.433158 |
| noise         | low     | raw       |   0.75     |   0.812776 |
| noise         | mid     | raw       |   0.709091 |   0.674351 |

_Descriptive only — chosen on the test split. A deployable policy must select on validation; see Stage C for the validation-safe confidence-based triage._

## Calibration (ECE)
| model           |   temperature |   ece_pre |   ece_post |      acc |
|:----------------|--------------:|----------:|-----------:|---------:|
| resnet50        |      0.999977 | 0.0698637 |  0.0698632 | 0.884091 |
| efficientnet_b3 |      0.935648 | 0.227796  |  0.219663  | 0.493182 |
| vit_base        |      0.857117 | 0.0707137 |  0.051476  | 0.836364 |
