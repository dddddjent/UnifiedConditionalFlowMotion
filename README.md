# Unified Motion

**A reimplementation** of [A Unified Conditional Flow for Motion Generation,
Editing, and Intra-Structural Retargeting](https://arxiv.org/abs/2604.13427)
by Junlin Li, Xinhao Song, Siqi Wang, Haibin Huang, and Yili Zhao.

One text- and skeleton-conditioned transformer supports motion generation,
zero-shot text editing, and retargeting between characters with the same
topology.

The training configuration targets **SMG + Mixamo with concatenated SMG and
AnyTop features**.

Inference and evaluation use the `--checkpoint` and `--data-root`
arguments to determine the checkpoint used and dataset root. 
Point them at the matching checkpoint and dataset
root; the loader identifies the checkpoint format and native feature width.

**Notice:**

Since we cannot redistribute Mixamo's data, we cannot provide the full dataset. SnapMoGen's dataset is directly available on HuggingFace. Therefore, we provide a model (`model.pt`) that was trained on SnapMoGen only in SnapMoGen feature space. This way, the model can be run easily on the existing data. Only text-based generation and edit will work on this model.

To obtain the full model, you need to first download the motions from Mixamo, and preprocess it and merge it with SnapMoGen dataset, according to the paper. The training code is for the full model. The inference code here handles the full or SnapMoGen only dataset/model at the same time.

## Installation

Use Python 3.11. PyTorch 2.8.0 and the remaining dependencies are pinned in
`pyproject.toml`. Install the PyTorch wheel appropriate for your machine before
installing this package if you need a specific CUDA build.

```bash
conda create -n unified-motion python=3.11 -y
conda activate unified-motion
python -m pip install -e .
```

Run the module commands below from this repository's root. T5 weights download
through Hugging Face on first use. Retargeting uses null text and does not need
to load T5 weights.

## Included checkpoints

| File | Purpose | Motion channels |
| --- | --- | ---: |
| `checkpoints/model.pt` | Pretrained research checkpoint; inference loads EMA weights | 296 |
| `checkpoints/evaluator.pt` | Frozen SnapMoGen text/motion embedding evaluator | 148 input channels |

The two files are in the release page. The copied training checkpoint
still contains its original optimizer and raw weights; inference uses only its
`ema_model`. No checkpoint was retrained, converted, or padded during this rewrite.

All checkpoints read their training normalization from `metadata/` under
the specified dataset root. For the supplied weights, provide:

```text
data/smg/metadata/
  motion_mean.npy  
  motion_std.npy   
  skeleton_mean.npy
  skeleton_std.npy 
```

Motion and skeleton input paths are relative to `--data-root`; absolute paths
are also accepted. Checkpoint and output paths are relative to the working
directory. 

## Generation

Supply a raw, unnormalized T-pose vector. Its joint ordering must
match the canonical skeleton (SMG).

```bash
python -m unified_motion.inference generate \
  --checkpoint checkpoints/model.pt \
  --data-root data/smg \
  --skeleton skeletons/smg.npy \
  --prompt "A person walks forward and then waves with their right hand." \
  --frames 320 --samples 4 --steps 100 --method rk4 \
  --device cuda --seed 42 --fps 30 --output outputs/generation
```

Each sample produces `.npy` features, `.npz` direct/FK positions and rotations,
and a `.bvh` animation. `inference.json` records the resolved settings and
checkpoint epoch. The `.npy` output can be used as the source of a subsequent edit.

To use a recovered original paper checkpoint, change only
`--checkpoint /path/to/checkpoint.pt` and `--data-root`.
Generation defaults to the guidance used in the original evaluations
(text 2.0 / skeleton 1.0 for the full model, 2.3 / 1.0 for the SnapMoGen-only model);
`--text-guidance` and `--skeleton-guidance` override it.

## Text editing

The source is an unnormalized motion feature array, not a BVH file. Describe
both the source motion and the intended target motion. The skeleton stays fixed.

```bash
python -m unified_motion.inference edit \
  --checkpoint checkpoints/model.pt \
  --data-root data/smg \
  --source "$PWD/outputs/generation/sample-000.npy" \
  --source-skeleton skeletons/smg.npy \
  --source-prompt "A person walks forward and then waves with their right hand." \
  --target-prompt "A person walks forward and then raises both arms overhead." \
  --source-text-guidance 1.5 --target-text-guidance 3.5 \
  --skeleton-guidance 1.0 --steps 100 --start-step 10 \
  --tail-steps 0 --averages 1 --device cuda --seed 42 --fps 30 \
  --output outputs/edit
```

`start-step / steps` is the initial flow time. A smaller start step gives a
stronger edit. `tail-steps` optionally finishes with ordinary target-conditioned
generation; its default is zero, retaining the shared-noise transport throughout.

## Retargeting

The source and target skeletons must have the same topology and joint order.
Retargeting changes only the skeleton condition and uses null text.

```bash
python -m unified_motion.inference retarget \
  --checkpoint checkpoints/model.pt \
  --data-root data/smg \
  --source "$PWD/outputs/generation/sample-000.npy" \
  --source-skeleton skeletons/smg.npy \
  --target-skeleton skeletons/target-character.npy \
  --skeleton-guidance 0.8 --steps 100 --start-step 10 \
  --tail-steps 0 --averages 1 --device cuda --seed 42 --fps 30 \
  --output outputs/retarget
```


## Dataset and training

Dataset assets are not distributed. The paper describes
how the paper combines SnapMoGen and Mixamo, the feature layout, captions,
splits, and the manifest expected by the loader. The importer can reorganize an
existing prepared research export without modifying feature values:

```bash
python -m unified_motion.prepare_data \
  --source /path/to/prepared-export --output data/paper --pair-end-offset 0 \
  --evaluator-metadata /path/to/SnapMoGen/meta_data --normalization-height 85.616624
```

```bash
torchrun --standalone --nproc_per_node=8 -m unified_motion.train \
  --config configs/default.json --data-root data/paper \
  --output runs/paper --device cuda
```


## Evaluation

Generation evaluation uses SMG test motions, one caption and one rounded motion
window per clip per replication. It reports FID, cosine R-Precision, CLIP score,
diversity, and optionally multimodality. Evaluation normalizes the leading 148
SMG channels with the original SnapMoGen evaluator's `mean.npy` and `std.npy`.
Place these under `<data-root>/metadata/evaluator/`. Each dataset root
used for generation evaluation includes this directory.

```bash
torchrun --standalone --nproc_per_node=8 -m unified_motion.evaluate generation \
  --checkpoint checkpoints/model.pt \
  --data-root data/smg \
  --evaluator checkpoints/evaluator.pt \
  --replications 20 --batch-size 16 --retrieval-pool 100 \
  --window 320 --min-frames 128 --unit-length 8 --steps 100 --method rk4 \
  --multimodality --mm-prompts 100 --mm-repeats 30 --mm-pairs 10 \
  --device cuda --seed 42 --output outputs/generation-metrics.json
```

Retargeting evaluation requires paired Mixamo test clips in the chosen root.
Every sweep candidate is saved; direct and FK metrics use separate
ground-truth-selected minima, with the selected start steps recorded explicitly.

The default pair file is `<data-root>/splits/retarget-test.jsonl`.
Store the dataset's positional scale in `<data-root>/metadata/evaluation.json`:

```json
{"normalization_height": 85.616624}
```

```bash
torchrun --standalone --nproc_per_node=8 -m unified_motion.evaluate retarget \
  --checkpoint checkpoints/model.pt \
  --data-root data/smg \
  --steps 100 --start-steps 5 10 15 20 25 30 35 40 \
  --skeleton-guidance 0.8 --repeats 1 --averages 1 \
  --device cuda --seed 42 --output outputs/retarget-metrics.json
```

The example height is the historical canonical normalization value; set the
metadata value in the same units as your prepared positions. Optional `--pairs`
and `--normalization-height` override those dataset defaults. Evaluation also
works with `python -m` on one device. 

Editing was evaluated through the paper's perceptual
user study; this repository provides the editing inference needed to create
comparison motions rather than inventing an automatic replacement metric.

## Layout

```text
unified_motion/
  model.py, attention.py    # shared joint/frame transformer
  text.py                   # frozen T5 encoding
  flow.py                   # generation and FlowEdit
  geometry.py               # SMG / AnyTop decoding and BVH export
  checkpoint.py             # strict EMA loading and checkpoint saving
  statistics.py, data.py    # normalization and manifest-based loading
  train.py                  # combined-dataset training
  inference.py              # generate / edit / retarget
  evaluate.py               # generation and retargeting evaluation
  evaluator.py, metrics.py  # frozen benchmark encoder and metrics
  prepare_data.py           # prepared-export importer
configs/default.json
checkpoints/
```

## Citation and license

```bibtex
@article{li2026unifiedmotion,
  title={A Unified Conditional Flow for Motion Generation, Editing, and Intra-Structural Retargeting},
  author={Li, Junlin and Song, Xinhao and Wang, Siqi and Huang, Haibin and Zhao, Yili},
  journal={arXiv preprint arXiv:2604.13427},
  year={2026}
}
```

Reimplementation source code is provided under the [MIT license](LICENSE).
