# SMILES-2026 ZO-Limited-ResNet Solution

Author: Artyom Manturov

The copy of description is available in `SOLUTION.md`.

---

## Reproducibility instructions

The final/default configuration is selected by the global variable `SOLUTION_PRESET` in `zo_optimizer.py`:

```python
SOLUTION_PRESET = 4
```

Run the official evaluator with:

```bash
pip install -r requirements.txt
python validate.py --data_dir ./data --batch_size 256 --n_batches 32 --output results.json
```

The run uses the official `validate.py` and `model.py` files unchanged. The editable files are:

- `head_init.py`
- `zo_optimizer.py`
- `train_data.py`
- `augmentation.py`

The archived best run before refactoring produced:

```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.6175,
  "val_accuracy_top1_finetuned": 0.6186,
  "n_batches": 32,
  "batch_size": 256,
  "layers_tuned": [
    "layer4.1.bn2.bias",
    "layer4.1.bn2.weight"
  ],
  "total_samples": 10000
}
```

Small numerical variation is possible because the zero-order optimizer samples random perturbation directions and the environment/BLAS/CUDA stack can differ.

## Final solution description

### High-level approach

The final/default preset, `SOLUTION_PRESET = 4`, uses a strong closed-form initialization for the CIFAR100 classification head and then performs a very conservative zero-order update of the final ResNet feature extractor.

The key observation is that ImageNet-pretrained ResNet18 already provides useful 512-dimensional features before the final fully connected layer. Instead of learning the 100-class CIFAR head from a random start, I fit a linear classifier directly on these frozen features. After that, the `fc` layer is kept fixed and the zero-order optimizer only adjusts the final BatchNorm affine parameters:

```text
layer4.1.bn2.weight
layer4.1.bn2.bias
```

This was the most stable configuration found. Larger backbone updates, convolutional tuning, aggressive class sampling, and mosaics tended to degrade the already strong fixed-head solution.

### Presets

`head_init.py` and `zo_optimizer.py` support four presets controlled by `SOLUTION_PRESET`:

| Preset | Description | Initialization samples | ZO optimization samples | Original init acc. | My init acc. | After ZO acc. |
|---|---|---|---|---|--------------|---------------|
| 1 | Xavier head, then zero-order optimization of `fc` and late BN | 0 | up to 8192 | 0.37% | 1.22%        | 1.17%         |
| 2 | ImageNet-to-CIFAR Hungarian matching, then zero-order optimization | 8192 | up to 8192 | 0.37% | 22.29%       | 22.31%        |
| 3 | Ridge/least-squares head on frozen ResNet features, then zero-order optimization | 8192 | up to 8192 | 0.37% | 59.10%       | 58.14%        |
| 4 | Ridge/least-squares head using all CIFAR100 train samples, fixed `fc`, minimal BN tuning | 50000 | up to 8192 | 0.37% | 61.75%       | 61.86%        |

Preset 1 is the cleanest baseline but is weak because the optimizer has to learn a 100-way classifier from a random head under a small zero-order budget.

Preset 2 reuses semantic information from the ImageNet classifier. I compute class-wise ImageNet logits on CIFAR100 samples and match CIFAR100 classes to ImageNet classes with a pure NumPy Hungarian matching implementation, without SciPy.

Preset 3 fits the final linear layer by solving a regularized least-squares problem on frozen 512-dimensional features:

```text
min_W ||XW - Y||^2 + lambda ||W||^2
```

Preset 4 is the final empirical choice. It uses the same ridge idea as Preset 3, but with all available CIFAR100 training samples. Since the resulting `fc` layer is already strong, it is frozen and only `layer4.1.bn2.weight/bias` are tuned.

### Zero-order optimizer

The optimizer uses SPSA-style antithetic perturbations. For selected parameters `theta`, it samples a random direction `u` and evaluates:

```text
loss(theta + eps * u)
loss(theta - eps * u)
```

The pseudo-gradient estimate is:

```text
(loss_plus - loss_minus) / (2 * eps) * u
```

Updates are smoothed with Adam-like first and second moments and clipped by per-group update norms. For the final preset, the important settings are:

```python
lr = 2e-3
eps = 3e-3
num_directions = 2
base_groups = ("layer4_1_bn2",)
freeze_fc = True
```

The active layer list in `results.json` should therefore be:

```json
[
  "layer4.1.bn2.bias",
  "layer4.1.bn2.weight"
]
```

### Data and confusion feedback

`head_init.py` computes an initial confusion matrix after the head initialization and exposes it through `get_initial_confusion_matrix()`. The optimizer can update this matrix online using logits captured by a forward hook and labels recorded by `train_data.py`.

`train_data.py` can mildly upweight hard classes based on this feedback. In the final preset this feedback is conservative. Mosaics are disabled by default because they introduced a train/validation distribution shift: training samples became collages, while validation samples remained normal single CIFAR images.

## Experiments and failed attempts

### Random/Xavier head only

This preset uses the whole formal budget only during optimization. It was a weak baseline: a random 100-class head starts too far from a useful classifier, and noisy zero-order updates were not enough to learn it well.

### ImageNet Hungarian matching

This produced a better initialization than random weights, but it is limited by the mismatch between ImageNet and CIFAR100 labels. It is useful as a robust transfer preset, but did not outperform the ridge head.

### Ridge head with trainable `fc`

The ridge/least-squares head was a strong initialization. Allowing `fc` to keep changing under SPSA-style updates often failed to improve it and sometimes reduced validation accuracy. Once the linear head is already fit to the frozen features, random zero-order updates to `fc` mostly add noise.

### Deeper backbone updates

Opening more BatchNorm groups or convolutional weights in `layer4` usually made optimization less stable. Changing the backbone alters the feature distribution expected by the fitted `fc` layer. The only consistently safe update was the final BatchNorm affine pair `layer4.1.bn2.weight/bias`.

### Budget interpretation

The official script enforces the optimization budget as:

```text
n_batches × batch_size ≤ 8192
```

This limit is enforced inside the fine-tuning loop. Data-dependent initialization in `head_init.py` is not explicitly limited by the repository structure, but it still uses samples and forward passes. For that reason, Presets 2, 3, and especially 4 are in a grey area if the intended rule is to count all pre-optimization computation as part of the same budget.

If such initialization is considered outside the intended rules, then the strictest interpretation of my result is:

```text
Use the 8192-sample budget for Preset 3 initialization only,
and ignore any later fine-tuning improvement.
```

Under this interpretation, Preset 3 is the most defensible final variant: it uses 8192 samples to compute a ridge head and does not rely on extra zero-order optimization gains. Preset 4 is the strongest empirical variant, but it uses all 50000 CIFAR100 training samples during initialization.

## How to switch presets

Edit the top of `zo_optimizer.py`:

```python
SOLUTION_PRESET = 4
```

Available values:

```text
1 = Xavier head + ZO over fc and late BN
2 = Hungarian-matched ImageNet head + ZO over fc and late BN
3 = Ridge head + ZO over fc and late BN
4 = Ridge head + fixed fc + ZO over layer4.1.bn2 only
```

Preset-specific knobs are in the `PRESET_CONFIGS` dictionary in `zo_optimizer.py`. Initialization sample counts are in `INIT_SAMPLES_PER_CLASS` in `head_init.py`.

## Conclusion

The main finding is that effective weight optimization with pseudo-gradients was not reliable under this budget. SPSA-style updates, Adam-like moments, different layer schedules, and limited backbone tuning were tested, but the updates were often too noisy to consistently improve a strong model.

The best gains came from initializing the final `fc` layer well. The pretrained ResNet18 backbone already provides useful features, and solving the final linear classification problem on top of those features was much more reliable than trying to learn many weights through noisy zero-order estimates.

Therefore, the strongest empirical solution is Preset 4. Under a stricter budget interpretation, the fairest final result is Preset 3 with only the initialized-head performance counted.
