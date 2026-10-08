# Primary model metric tables

These tables are derived from existing `results/04_training_cnn_vs_cnn_lstm` artifacts only.
No model was retrained, no prediction was edited, and the fixed threshold remains `P=0.5`.

## Recommended use

- Use event-level FPT metrics as the primary evidence.
- Use sample-level metrics as supporting classification evidence.
- Treat accuracy as secondary because healthy/degraded samples are imbalanced.

## Aggregate summary

| Model | Mean abs. FPT error | Median abs. FPT error | Mean F1 | Mean recall | Mean balanced acc. | Mean AUC-PR | Mean accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| CWT-CNN | 464.0 | 83.0 | 0.66204 | 0.837689 | 0.831197 | 0.974107 | 0.73315 |
| CWT-CNN-LSTM | 567.833333 | 369.0 | 0.594361 | 0.772998 | 0.768173 | 0.887592 | 0.680859 |

## Artifact warnings

- fold_1_cnn: missing predictions.npz; used saved metric/confusion JSON only

## Leakage note

All metrics are computed from frozen outer-test artifacts after model selection.
No threshold calibration or model selection is performed in this script.