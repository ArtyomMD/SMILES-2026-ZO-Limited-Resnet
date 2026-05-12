"""
zo_optimizer.py — preset-driven zero-order optimizer for SMILES-2026 ResNet18.

The active strategy is selected by the global SOLUTION_PRESET below.

Preset overview
---------------
1. Xavier head + ZO fine-tuning.
   The head is initialized with Xavier in head_init.py. The optimizer may update
   fc and a small late-BN group.

2. ImageNet-to-CIFAR Hungarian head + ZO fine-tuning.
   head_init.py runs the pretrained ImageNet ResNet18 on a deterministic CIFAR100
   subset, matches CIFAR100 classes to ImageNet logits with an in-file Hungarian
   implementation, copies the corresponding ImageNet fc rows into the 100-class
   head, and exposes an initial confusion matrix. The optimizer may update fc
   and a small late-BN group.

3. Ridge/least-squares head + ZO fine-tuning.
   head_init.py extracts frozen 512-dimensional features before fc, solves a
   ridge least-squares system for the 100-class head, and exposes an initial
   confusion matrix. The optimizer may update fc and late BN.

4. Final/default solution.
   head_init.py computes the ridge head in memory. The optimizer keeps fc fixed
   and makes a very small backbone-only ZO update on layer4.1.bn2.weight/bias.
   This corresponds to the best-performing configuration found in experiments.

"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


NUM_CLASSES = 100

# ---------------------------------------------------------------------------
# Main switch. Change this value to select the solution mode.
# ---------------------------------------------------------------------------
SOLUTION_PRESET = 4


@dataclass(frozen=True)
class ParamGroup:
    name: str
    params: tuple[str, ...]
    lr_scale: float = 1.0
    max_update_norm: float = 0.03


# These knobs are deliberately grouped by preset so each mode can be tuned
# independently without changing the optimizer logic.
PRESET_CONFIGS: dict[int, dict] = {
    1: {
        "name": "xavier_head_train_fc_plus_late_bn",
        "lr": 1.0e-3,
        "eps": 5.0e-3,
        "num_directions": 2,
        "beta1": 0.90,
        "beta2": 0.99,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "freeze_fc": False,
        "base_groups": ("fc", "layer4_1_bn2"),
        "scheduled_groups": (),
        "confusion_decay": 0.985,
        "hard_classes_per_refresh": 16,
        "top_confusers_per_class": 1,
    },
    2: {
        "name": "hungarian_head_train_fc_plus_late_bn",
        "lr": 8.0e-4,
        "eps": 4.0e-3,
        "num_directions": 2,
        "beta1": 0.90,
        "beta2": 0.99,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "freeze_fc": False,
        "base_groups": ("fc", "layer4_1_bn2"),
        "scheduled_groups": (),
        "confusion_decay": 0.985,
        "hard_classes_per_refresh": 20,
        "top_confusers_per_class": 1,
    },
    3: {
        "name": "ridge_head_train_fc_plus_late_bn",
        "lr": 5.0e-5,
        "eps": 3.0e-4,
        "num_directions": 2,
        "beta1": 0.90,
        "beta2": 0.99,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "freeze_fc": False,
        "base_groups": ("fc", "layer4_1_bn2"),
        "scheduled_groups": (),
        "confusion_decay": 0.990,
        "hard_classes_per_refresh": 20,
        "top_confusers_per_class": 1,
    },
    4: {
        "name": "ridge_head_fixed_fc_backbone_bn2_only",
        # These values match the best archived run: fc fixed, only layer4.1.bn2.
        "lr": 2.0e-3,
        "eps": 3.0e-3,
        "num_directions": 2,
        "beta1": 0.90,
        "beta2": 0.99,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "freeze_fc": True,
        "base_groups": ("layer4_1_bn2",),
        "scheduled_groups": (),
        "confusion_decay": 0.985,
        "hard_classes_per_refresh": 24,
        "top_confusers_per_class": 2,
    },
}


def _read_cli_int(flag: str, default: int) -> int:
    """Read validate.py command line args without modifying validate.py."""
    argv = sys.argv
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
        if item.startswith(flag + "="):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return default
    return default


class ZeroOrderOptimizer:
    """Gradient-free optimizer with preset-specific layer selection."""

    def __init__(
        self,
        model: nn.Module,
        lr: float | None = None,
        eps: float | None = None,
        perturbation_mode: str = "rademacher",
    ) -> None:
        if SOLUTION_PRESET not in PRESET_CONFIGS:
            raise ValueError(f"Unknown SOLUTION_PRESET={SOLUTION_PRESET}. Expected one of {sorted(PRESET_CONFIGS)}")

        self.model = model
        self.config = dict(PRESET_CONFIGS[SOLUTION_PRESET])
        self.preset = int(SOLUTION_PRESET)

        self.lr = float(self.config["lr"] if lr is None else lr)
        self.eps = float(self.config["eps"] if eps is None else eps)

        if perturbation_mode not in ("rademacher", "gaussian", "uniform"):
            raise ValueError(
                "perturbation_mode must be 'rademacher', 'gaussian', or 'uniform', "
                f"got {perturbation_mode!r}"
            )
        self.perturbation_mode = perturbation_mode

        self.total_steps = max(1, _read_cli_int("--n_batches", default=64))
        self.num_directions = int(self.config["num_directions"])
        self.beta1 = float(self.config["beta1"])
        self.beta2 = float(self.config["beta2"])
        self.adam_eps = float(self.config["adam_eps"])
        self.weight_decay = float(self.config["weight_decay"])

        self.confusion_decay = float(self.config["confusion_decay"])
        self.hard_classes_per_refresh = int(self.config["hard_classes_per_refresh"])
        self.top_confusers_per_class = int(self.config["top_confusers_per_class"])
        self.min_support_for_hardness = 1
        self._current_focus_classes: list[int] = list(range(NUM_CLASSES))

        self.step_idx = 0
        self.confusion = self._load_initial_confusion_matrix_from_head_init()
        self._last_logits_cpu: torch.Tensor | None = None
        self._last_batch_labels_cpu: torch.Tensor | None = None
        self._last_batch_preds_cpu: torch.Tensor | None = None
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._param_lr_scales: dict[str, float] = {}
        self._param_clip_norms: dict[str, float] = {}

        self.groups = self._make_param_groups()
        self.layer_names: list[str] = self._select_layer_names_for_step()

        # Make the intent visible for normal autograd users too. ZO itself uses
        # .data updates and the selected layer_names below.
        if bool(self.config["freeze_fc"]):
            for name, param in self.model.named_parameters():
                if name.startswith("fc."):
                    param.requires_grad_(False)

        self._forward_hook_handle = self.model.register_forward_hook(self._capture_logits_hook)
        self._publish_feedback_to_train_data()

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def _make_param_groups(self) -> dict[str, ParamGroup]:
        groups = [
            ParamGroup(
                name="fc",
                params=("fc.weight", "fc.bias"),
                lr_scale=1.00,
                max_update_norm=0.05,
            ),
            ParamGroup(
                name="layer4_1_bn2",
                params=("layer4.1.bn2.weight", "layer4.1.bn2.bias"),
                # Archived best config used global lr 2e-3 and this 1.5 scale.
                lr_scale=1.50 if self.preset == 4 else 1.00,
                max_update_norm=0.03,
            ),
            ParamGroup(
                name="layer4_1_bn_all",
                params=(
                    "layer4.1.bn1.weight",
                    "layer4.1.bn1.bias",
                    "layer4.1.bn2.weight",
                    "layer4.1.bn2.bias",
                ),
                lr_scale=0.75,
                max_update_norm=0.025,
            ),
            ParamGroup(
                name="layer4_0_bn",
                params=(
                    "layer4.0.bn1.weight",
                    "layer4.0.bn1.bias",
                    "layer4.0.bn2.weight",
                    "layer4.0.bn2.bias",
                    "layer4.0.downsample.1.weight",
                    "layer4.0.downsample.1.bias",
                ),
                lr_scale=0.50,
                max_update_norm=0.020,
            ),
            ParamGroup(
                name="layer4_1_conv2",
                params=("layer4.1.conv2.weight",),
                lr_scale=0.03,
                max_update_norm=0.005,
            ),
        ]
        return {g.name: g for g in groups}

    def _existing_names(self) -> set[str]:
        return {name for name, _ in self.model.named_parameters()}

    def _select_group_names_for_step(self) -> list[str]:
        group_names = list(self.config["base_groups"])
        progress = self.step_idx / max(1, self.total_steps - 1)

        # Optional scheduled groups are configured as tuples:
        # (progress_threshold, group_name). They are off by default for the final
        # preset but are useful for controlled experiments.
        for threshold, group_name in self.config.get("scheduled_groups", ()):  # type: ignore[assignment]
            if progress >= float(threshold):
                group_names.append(str(group_name))

        if bool(self.config["freeze_fc"]):
            group_names = [g for g in group_names if g != "fc"]

        return group_names

    def _select_layer_names_for_step(self) -> list[str]:
        existing = self._existing_names()
        layer_names: list[str] = []
        self._param_lr_scales = {}
        self._param_clip_norms = {}

        for group_name in self._select_group_names_for_step():
            group = self.groups[group_name]
            for param_name in group.params:
                if param_name in existing:
                    layer_names.append(param_name)
                    self._param_lr_scales[param_name] = group.lr_scale
                    self._param_clip_norms[param_name] = group.max_update_norm

        if not layer_names:
            raise RuntimeError("No active parameters were selected for ZO optimization.")

        return sorted(set(layer_names))

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: {missing}. "
                "Use [n for n, _ in model.named_parameters()] to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    # ------------------------------------------------------------------
    # Confusion matrix and train_data.py feedback
    # ------------------------------------------------------------------

    def _load_initial_confusion_matrix_from_head_init(self) -> np.ndarray:
        try:
            import head_init

            if hasattr(head_init, "get_initial_confusion_matrix"):
                cm = head_init.get_initial_confusion_matrix()
                if cm is not None and np.asarray(cm).shape == (NUM_CLASSES, NUM_CLASSES):
                    return np.asarray(cm, dtype=np.float64).copy()
        except Exception:
            pass
        return np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    def _capture_logits_hook(self, module: nn.Module, inputs, output) -> None:
        del module, inputs
        if torch.is_tensor(output) and output.ndim == 2 and output.shape[-1] == NUM_CLASSES:
            self._last_logits_cpu = output.detach().float().cpu()

    def _get_last_batch_labels_from_train_data(self) -> torch.Tensor | None:
        try:
            import train_data

            if hasattr(train_data, "get_last_batch_labels"):
                labels = train_data.get_last_batch_labels()
                if labels is not None:
                    return labels.detach().long().cpu()
        except Exception:
            pass
        return None

    def _update_confusion_from_last_forward(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        labels = self._get_last_batch_labels_from_train_data()
        logits = self._last_logits_cpu
        if labels is None or logits is None:
            return None, None

        n = min(labels.numel(), logits.shape[0])
        if n <= 0:
            return None, None

        labels = labels[:n]
        preds = logits[:n].argmax(dim=1).long()

        self.confusion *= self.confusion_decay
        for y, p in zip(labels.numpy(), preds.numpy()):
            y_i = int(y)
            p_i = int(p)
            if 0 <= y_i < NUM_CLASSES and 0 <= p_i < NUM_CLASSES:
                self.confusion[y_i, p_i] += 1.0

        self._last_batch_labels_cpu = labels
        self._last_batch_preds_cpu = preds
        return labels, preds

    def _class_accuracy_and_hardness(self) -> tuple[np.ndarray, np.ndarray]:
        support = self.confusion.sum(axis=1)
        correct = np.diag(self.confusion)

        acc = np.full(NUM_CLASSES, 0.5, dtype=np.float64)
        valid = support >= self.min_support_for_hardness
        acc[valid] = correct[valid] / np.maximum(support[valid], 1e-12)

        hardness = 1.0 - acc
        if support.max() > 0:
            evidence = np.log1p(support) / (np.log1p(support).max() + 1e-12)
            hardness = 0.80 * hardness + 0.20 * evidence
        return acc, hardness

    def _build_focus_classes(self, labels: torch.Tensor | None, preds: torch.Tensor | None) -> list[int]:
        focus: set[int] = set()

        if float(self.confusion.sum()) > 0.0:
            _, hardness = self._class_accuracy_and_hardness()
            hard = np.argsort(hardness)[::-1][: self.hard_classes_per_refresh]
            focus.update(int(c) for c in hard)

            for c in hard:
                row = self.confusion[int(c)].copy()
                row[int(c)] = 0.0
                confusers = np.argsort(row)[::-1][: self.top_confusers_per_class]
                for j in confusers:
                    if row[int(j)] > 0:
                        focus.add(int(j))

        if labels is not None:
            focus.update(int(x) for x in labels.tolist())
        if preds is not None:
            focus.update(int(x) for x in preds.tolist())
        if not focus:
            focus.update(range(NUM_CLASSES))

        return sorted(c for c in focus if 0 <= c < NUM_CLASSES)

    def _publish_feedback_to_train_data(self) -> None:
        try:
            import train_data

            if hasattr(train_data, "set_optimizer_feedback"):
                train_data.set_optimizer_feedback(
                    confusion_matrix=self.confusion.copy(),
                    focus_classes=list(self._current_focus_classes),
                    step_idx=int(self.step_idx),
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # SPSA pseudo-gradient
    # ------------------------------------------------------------------

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "rademacher":
            u = torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        elif self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:
            u = torch.rand_like(param).mul_(2.0).sub_(1.0)

        norm = u.norm()
        if norm > 0:
            u = u / norm
        return u

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        grads = {name: torch.zeros_like(param) for name, param in params.items()}

        with torch.no_grad():
            for _ in range(self.num_directions):
                directions = {name: self._sample_direction(param) for name, param in params.items()}

                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])
                f_plus = float(loss_fn())

                for name, param in params.items():
                    param.data.sub_(2.0 * self.eps * directions[name])
                f_minus = float(loss_fn())

                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])

                scale = (f_plus - f_minus) / (2.0 * self.eps)
                for name in params:
                    grads[name].add_(scale * directions[name])

            inv = 1.0 / float(max(1, self.num_directions))
            for name in grads:
                grads[name].mul_(inv)

        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        self.step_idx += 1
        t = float(self.step_idx)

        with torch.no_grad():
            for name, param in params.items():
                grad = grads[name]

                if self.weight_decay > 0.0:
                    grad = grad.add(self.weight_decay * param.data)

                if name not in self._m:
                    self._m[name] = torch.zeros_like(param)
                    self._v[name] = torch.zeros_like(param)

                self._m[name].mul_(self.beta1).add_((1.0 - self.beta1) * grad)
                self._v[name].mul_(self.beta2).add_((1.0 - self.beta2) * grad.square())

                m_hat = self._m[name] / (1.0 - self.beta1 ** t)
                v_hat = self._v[name] / (1.0 - self.beta2 ** t)

                lr_scale = self._param_lr_scales.get(name, 1.0)
                update = (self.lr * lr_scale) * m_hat / (v_hat.sqrt() + self.adam_eps)

                max_norm = self._param_clip_norms.get(name, 0.03)
                update_norm = update.norm()
                if update_norm > max_norm:
                    update.mul_(max_norm / (update_norm + 1e-12))

                param.data.sub_(update)

    # ------------------------------------------------------------------
    # Public API called by validate.py
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        self.layer_names = self._select_layer_names_for_step()
        params = self._active_params()

        with torch.no_grad():
            loss_before = float(loss_fn())

        labels, preds = self._update_confusion_from_last_forward()
        self._current_focus_classes = self._build_focus_classes(labels, preds)
        self._publish_feedback_to_train_data()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return loss_before
