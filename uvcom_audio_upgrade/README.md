# UVCOM-style DETR Backbone Upgrade

This folder documents the isolated CASTELLA/audio experiment added in
`lighthouse/lighthouse/common/uvcom_audio`.

## What changed

- `uvcom_audio` keeps Lighthouse's existing DETR I/O:
  text features plus audio-frame features in, `K` candidate segments out.
- The backbone adds text-conditioned multi-granularity context fusion before the
  existing UVCOM CIM module. Audio frames are pooled at `1,3,7,15` clip scales,
  gated by the query representation, and fused back into frame tokens.
- Training can use length-aware span weighting so short ground-truth moments
  receive larger L1/GIoU weight.
- A small per-query `pred_short_logits` head is available for short-vs-long
  calibration through `short_loss_coef`.

## How to run

From `lighthouse/`:

```bash
python training/train.py --model uvcom_audio --dataset castella --feature clap
```

For Clotho-Moment pretraining:

```bash
python training/train.py --model uvcom_audio --dataset clotho-moment --feature clap
```

The main config is `lighthouse/configs/model/uvcom_audio.yml`.
