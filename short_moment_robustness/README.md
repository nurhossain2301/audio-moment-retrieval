# Short-Moment Robustness Variant

This experiment is isolated from the baseline implementation under:

- `lighthouse/lighthouse/common/short_moment_robust/qd_detr.py`
- `lighthouse/lighthouse/common/short_moment_robust/matcher.py`
- `lighthouse/lighthouse/common/short_moment_robust/dataset.py`
- `lighthouse/lighthouse/common/short_moment_robust/postprocessing.py`

Those files were copied from the original QD-DETR, matcher, dataset, and
postprocessor files before adding the short-moment changes.

## Additions

- MomentMix-style feature-space augmentation during training.
  - Replaces short target spans with donor short-span features.
  - Optionally inserts donor short-span features and shifts affected labels.
- Length-aware Hungarian matching.
- Length-aware span/GIoU loss weighting.
- Dedicated shortness head, `pred_short_logits`.
- Inference score calibration for short precise predictions and overlong
  predictions.

## Run

From `lighthouse/`:

```bash
python training/train.py --model short_moment_robust --dataset clotho-moment --feature clap
```
