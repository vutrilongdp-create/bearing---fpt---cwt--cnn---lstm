# Paper-ready FPT result presentation

This folder contains presentation-only tables and figures derived from existing result artifacts.
No model was retrained, no prediction was edited, and no outer-test threshold tuning was performed.

## Primary result

Primary analysis uses the fixed probability threshold P=0.5 for both models.

## Secondary analysis

Secondary analysis uses thresholds selected from the inner-validation bearing only, then frozen before outer-test evaluation.

## Key aggregate metrics

| Analysis | Model | Mean abs. error | Median abs. error | Std abs. error | Missed | Pre-FPT alarm runs |
|---|---|---:|---:|---:|---:|---:|
| primary_fixed_threshold_p_0_5 | CWT-CNN | 464.0 | 83.0 | 677.202 | 0 | 2 |
| primary_fixed_threshold_p_0_5 | CWT-CNN-LSTM | 567.833 | 369.0 | 667.836 | 0 | 2 |
| secondary_inner_val_calibrated | CWT-CNN | 254.833 | 58.5 | 322.008 | 0 | 1 |
| secondary_inner_val_calibrated | CWT-CNN-LSTM | 452.833 | 23.5 | 692.014 | 0 | 2 |

## Recommended paper framing

- Report the fixed P=0.5 results as the main comparison.
- Report inner-validation threshold calibration as a secondary sensitivity analysis.
- Use event-level FPT error before sample-level metrics when discussing model quality.
- Highlight early false alarms explicitly instead of hiding difficult bearings.