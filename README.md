# dronio-custom-barcode-model

## Deep Transformer CTC Training Strategy (3-phase curriculum)

This repo contains a deep CNN + Transformer + CTC model for barcode recognition.

The recommended training workflow is a 3-phase curriculum:

- **Phase 1 (Day 1)**: Train on clean-ish data (**no augmentation**) until validation stops improving.
- **Phase 2 (Day 2)**: Fine-tune from the best checkpoint with **medium augmentation** (`--aug-prob 0.6`).
- **Phase 3 (Day 3)**: Fine-tune from the Phase 2 best checkpoint with **hard augmentation** (`--aug-prob 0.8` or `0.9`).

This is designed to:

- Learn stable CTC alignment on clean data first.
- Then progressively add robustness to motion blur, skew/rotation, perspective, and compression artifacts.


## Assumptions

- You are using a Python **venv** (recommended).
- Your dataset folder has this structure:

```
<DATASET_ROOT>/
  train/
    images/
    labels.csv
  val/
    images/
    labels.csv
  test/
    images/
    labels.csv
```

- You are training the **deep** model:

`transformer_model_deep/train_transformer_ctc.py`


## Common flags

- **TensorBoard logging**: `--tb`
- **Visualization collage (YOLO-like)**: `--viz --viz-n 8`
  - Logs a per-epoch collage showing validation images (clean) vs the same images passed through the training augmentation pipeline.
- **AMP (CUDA only)**: `--amp`
- **Augmentations**:
  - Enable: `--aug`
  - Probability: `--aug-prob <float>`


## Phase 1 (Day 1): Train without augmentation

Goal: learn clean decoding + stable CTC alignment.

Recommended:

- Keep `--aug` **off**.
- Use `--decode greedy` for speed.
- Use TensorBoard to watch validation loss / accuracy.

Command:

```bash
python3 transformer_model_deep/train_transformer_ctc.py \
  --data code128_dataset_v2 \
  --outdir checkpoints_transformer_deep_phase1 \
  --epochs 30 \
  --lr 2e-4 \
  --scheduler onecycle \
  --decode greedy \
  --tb --viz --viz-n 8 \
  --amp
```

When to stop Phase 1:

- Validation loss stops improving for several epochs.
- The model stops making progress in exact-match accuracy.

The best checkpoint is saved at:

- `checkpoints_transformer_deep_phase1/best_model.pt`


## Phase 2 (Day 2): Fine-tune with medium augmentation

Goal: learn robustness while preserving learned alignment.

Recommended changes vs Phase 1:

- Load Phase 1 best checkpoint using `--resume`.
- Turn augmentation on with **moderate probability**.
- Lower learning rate to avoid unlearning.

Command:

```bash
python3 transformer_model_deep/train_transformer_ctc.py \
  --data code128_dataset_v2 \
  --outdir checkpoints_transformer_deep_phase2 \
  --resume checkpoints_transformer_deep_phase1/best_model.pt \
  --epochs 10 \
  --lr 5e-5 \
  --scheduler cosine \
  --decode greedy \
  --aug --aug-prob 0.6 \
  --tb --viz --viz-n 8 \
  --amp
```

The best checkpoint from Phase 2 is saved at:

- `checkpoints_transformer_deep_phase2/best_model.pt`


## Phase 3 (Day 3): Fine-tune with hard augmentation

Goal: harden the model for motion blur, skew, rotation, and compression artifacts.

Recommended changes vs Phase 2:

- Load Phase 2 best checkpoint using `--resume`.
- Increase augmentation probability.
- Keep LR low.

Command (hard aug):

```bash
python3 transformer_model_deep/train_transformer_ctc.py \
  --data code128_dataset_v2 \
  --outdir checkpoints_transformer_deep_phase3 \
  --resume checkpoints_transformer_deep_phase2/best_model.pt \
  --epochs 5 \
  --lr 2e-5 \
  --scheduler cosine \
  --decode greedy \
  --aug --aug-prob 0.85 \
  --tb --viz --viz-n 8 \
  --amp
```

The final best checkpoint is:

- `checkpoints_transformer_deep_phase3/best_model.pt`


## Notes on LR + scheduler

- **Phase 1**:
  - `onecycle` works well to quickly reach a good solution.
  - Default `--lr 2e-4` is a reasonable starting point.
- **Phases 2-3**:
  - Prefer `cosine` (or `none`) with **lower LR**.
  - The main goal is fine-tuning robustness without destabilizing CTC alignment.


## Optional: inference sanity check

After training, you can run inference and print predicted codeword integers.

- `transformer_model_deep/inference.py`


## Tips

- If training is slow, increase `--num-workers` (e.g. `4` or `8`).
- Keep validation **clean** (no augmentation) so metrics remain comparable.
- If you see loss spikes when enabling augmentation, reduce `--aug-prob` or reduce blur intensity (augmentation config is in `transformer_model_deep/data.py`).
