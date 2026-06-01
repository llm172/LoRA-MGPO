from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear as LoraLinear
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Trainer
from transformers.data.data_collator import DataCollator
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.utils import is_sagemaker_mp_enabled


class LoraMGPOTrainer(Trainer):
    """LoRA-MGPO trainer aligned with the paper method section.

    The perturbation follows

        epsilon_l = -rho * v_l / ||v_l||_2 * (gbar_l)^-1

    where v_l is the optimizer momentum reconstructed in effective LoRA
    weight space and gbar_l is an EMA-smoothed layer gradient magnitude.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        rho: float = 0.05,
        mgpo_beta: float = 0.9,
        mgpo_eps: float = 1e-12,
        mgpo_normalize: bool = True,
        mgpo_log_steps: int = 50,
        mgpo_profile: str = "nlu",
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.rho = float(rho)
        self.mgpo_beta = float(mgpo_beta)
        self.mgpo_eps = float(mgpo_eps)
        self.mgpo_normalize = bool(mgpo_normalize)
        self.mgpo_log_steps = int(mgpo_log_steps)
        self.mgpo_profile = mgpo_profile
        self._mgpo_grad_ema: Dict[int, torch.Tensor] = {}
        self._mgpo_active_modules: List[LoraLinear] = []
        self._mgpo_last_stats = {"layers": 0, "mean_scale": 0.0, "mean_norm": 0.0}
        self._mgpo_last_logged_step = -1

        assert self.rho > 0, "rho must be positive for LoRA-MGPO"
        assert 0 <= self.mgpo_beta < 1, "mgpo_beta must satisfy 0 <= beta < 1"
        assert self.mgpo_eps > 0, "mgpo_eps must be positive"
        print(
            "-" * 40,
            "\n",
            "Using LoRA-MGPO",
            f"profile={self.mgpo_profile}",
            f"rho={self.rho}",
            f"beta={self.mgpo_beta}",
            "\n",
            "-" * 40,
        )

    def create_optimizer(self):
        if getattr(self.args, "loraplus_lr_ratio", None) is None:
            return super().create_optimizer()

        from lora_plus import create_loraplus_optimizer

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = create_loraplus_optimizer(
                opt_model,
                optimizer_cls,
                optimizer_kwargs,
                getattr(self.args, "loraplus_lr_ratio", None),
                getattr(self.args, "loraplus_lr_embedding", None),
            )
        return self.optimizer

    @staticmethod
    def _active_adapter(module: LoraLinear) -> Optional[str]:
        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, str):
            return active_adapter
        active_adapters = getattr(module, "active_adapters", None)
        if active_adapters:
            return active_adapters[0]
        if "default" in module.lora_A:
            return "default"
        return next(iter(module.lora_A.keys()), None)

    def _optimizer_momentum(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        if self.optimizer is None:
            return None
        state = self.optimizer.state.get(param, {})
        momentum = state.get("exp_avg", None)
        if momentum is None:
            momentum = state.get("momentum_buffer", None)
        return momentum

    @staticmethod
    def _lora_scaling(module: LoraLinear, adapter: str) -> float:
        scaling = module.scaling[adapter]
        if isinstance(scaling, torch.Tensor):
            return float(scaling.detach().item())
        return float(scaling)

    def _build_weight_space_momentum(
        self, module: LoraLinear
    ) -> Optional[Tuple[torch.Tensor, int]]:
        adapter = self._active_adapter(module)
        if adapter is None or adapter not in module.lora_A or adapter not in module.lora_B:
            return None

        lora_a = module.lora_A[adapter].weight
        lora_b = module.lora_B[adapter].weight
        momentum_a = self._optimizer_momentum(lora_a)
        momentum_b = self._optimizer_momentum(lora_b)
        if momentum_a is None or momentum_b is None:
            return None

        with torch.no_grad():
            direction = self._lora_scaling(module, adapter) * (
                momentum_b.detach().float() @ lora_a.detach().float()
                + lora_b.detach().float() @ momentum_a.detach().float()
            )
        if not torch.isfinite(direction).all():
            return None
        return direction, id(module)

    def _compute_mgpo_perturbation(
        self, module: LoraLinear
    ) -> Optional[Tuple[torch.Tensor, float, float]]:
        result = self._build_weight_space_momentum(module)
        if result is None:
            return None
        direction, layer_id = result
        direction_norm = torch.linalg.vector_norm(direction)
        if direction_norm <= self.mgpo_eps:
            return None

        gbar = self._mgpo_grad_ema.get(layer_id)
        if gbar is None or not self.mgpo_normalize:
            adaptive_scale = direction.new_tensor(1.0)
        else:
            adaptive_scale = torch.reciprocal(gbar + self.mgpo_eps)
        adaptive_scale = adaptive_scale.clamp(max=10.0)

        perturb = -self.rho * direction / (direction_norm + self.mgpo_eps)
        perturb = (perturb * adaptive_scale).to(
            device=module.weight.device, dtype=module.weight.dtype
        )
        return (
            perturb,
            float(adaptive_scale.detach().cpu()),
            float(direction_norm.detach().cpu()),
        )

    def _apply_mgpo_perturbation(self, model: nn.Module):
        self._mgpo_active_modules = []
        scales = []
        norms = []
        for module in model.modules():
            if not isinstance(module, LoraLinear):
                continue
            result = self._compute_mgpo_perturbation(module)
            if result is None:
                continue
            perturb, adaptive_scale, direction_norm = result
            with torch.no_grad():
                module.weight.add_(perturb)
            self._mgpo_active_modules.append(module)
            scales.append(adaptive_scale)
            norms.append(direction_norm)

        if scales:
            self._mgpo_last_stats = {
                "layers": len(scales),
                "mean_scale": float(np.mean(scales)),
                "mean_norm": float(np.mean(norms)),
            }

    def _remove_mgpo_perturbation(self):
        for module in reversed(self._mgpo_active_modules):
            result = self._compute_mgpo_perturbation(module)
            if result is None:
                continue
            perturb, _, _ = result
            with torch.no_grad():
                module.weight.sub_(perturb)
        self._mgpo_active_modules = []

    def _update_gradient_ema(self, model: nn.Module):
        for module in model.modules():
            if not isinstance(module, LoraLinear):
                continue
            adapter = self._active_adapter(module)
            if adapter is None or adapter not in module.lora_A or adapter not in module.lora_B:
                continue

            grad_a = module.lora_A[adapter].weight.grad
            grad_b = module.lora_B[adapter].weight.grad
            if grad_a is None or grad_b is None:
                continue

            with torch.no_grad():
                grad_delta_w = self._lora_scaling(module, adapter) * (
                    grad_b.detach().float() @ module.lora_A[adapter].weight.detach().float()
                    + module.lora_B[adapter].weight.detach().float() @ grad_a.detach().float()
                )
                grad_norm = torch.linalg.vector_norm(grad_delta_w)
                layer_id = id(module)
                prev = self._mgpo_grad_ema.get(layer_id)
                if prev is None:
                    self._mgpo_grad_ema[layer_id] = grad_norm
                else:
                    self._mgpo_grad_ema[layer_id] = (
                        self.mgpo_beta * prev + (1.0 - self.mgpo_beta) * grad_norm
                    )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        self._apply_mgpo_perturbation(model)
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
        finally:
            self._remove_mgpo_perturbation()

        self._update_gradient_ema(model)

        if (
            self.mgpo_log_steps > 0
            and self.state.global_step > 0
            and self.state.global_step % self.mgpo_log_steps == 0
            and self.state.global_step != self._mgpo_last_logged_step
            and self.is_world_process_zero()
        ):
            self.log(
                {
                    "mgpo/layers": self._mgpo_last_stats["layers"],
                    "mgpo/mean_adaptive_scale": self._mgpo_last_stats["mean_scale"],
                    "mgpo/mean_direction_norm": self._mgpo_last_stats["mean_norm"],
                }
            )
            self._mgpo_last_logged_step = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps
