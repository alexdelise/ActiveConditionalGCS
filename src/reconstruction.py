"""Single-run reconstruction routines for diffusion backpropagation."""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .config import RunConfig, run_config_to_dict, sampling_method_folder
from .diffusion import (
    decode_latents_to_unit_interval,
    encode_image_to_latents,
    make_latents,
    unrolled_image_from_latents,
)
from .metrics import calculate_psnr, calculate_ssim, chw_to_hwc_for_display
from .sampling import MeasurementOperator, build_sampling_pattern
from .utils import json_dump, safe_empty_cuda_cache, sha256_text


def load_gt_image_chw(gt_path: str, height: int, width: int) -> np.ndarray:
    """Load a saved ground-truth image from disk as a channel-first float array."""

    import matplotlib.pyplot as plt

    gt = plt.imread(gt_path)
    if gt.ndim == 2:
        gt = gt[:, :, None]
    if gt.shape[2] == 4:
        gt = gt[:, :, :3]
    if gt.shape[0] != height or gt.shape[1] != width:
        raise ValueError(f"Ground-truth image has shape {gt.shape[:2]}, expected {(height, width)}.")
    if gt.dtype != np.float32 and gt.dtype != np.float64:
        gt = gt.astype(np.float32)
    if gt.max() > 1.0:
        gt = gt / 255.0
    return np.transpose(gt, (2, 0, 1)).astype(np.float32, copy=False)


def timestep_batch_tensor(timestep_value, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a scheduler-timestep tensor with the desired batch size."""

    if torch.is_tensor(timestep_value):
        timestep = timestep_value.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
    else:
        timestep = torch.tensor([float(timestep_value)], device=device)
    if timestep.shape[0] == 1 and batch_size > 1:
        timestep = timestep.repeat(batch_size)
    return timestep


def noise_latents_at_timestep(pipe, latents_in: torch.Tensor, timestep_value) -> torch.Tensor:
    """Inject noise into a latent tensor at a specific scheduler timestep."""

    noise = torch.randn_like(latents_in)
    timestep_batch = timestep_batch_tensor(timestep_value, int(latents_in.shape[0]), latents_in.device)
    if hasattr(pipe.scheduler, "scale_noise"):
        try:
            return pipe.scheduler.scale_noise(latents_in, timestep_batch, noise)
        except Exception:
            pass
    if hasattr(pipe.scheduler, "add_noise"):
        try:
            return pipe.scheduler.add_noise(latents_in, noise, timestep_batch)
        except Exception:
            pass
    return latents_in + noise


def diffusion_backprop_learning_rate(iteration: int, *, config) -> float:
    """Return the active diffusion-backprop learning rate for a given iteration."""

    base_lr = max(0.0, float(config.learning_rate))
    schedule = str(getattr(config, "lr_schedule", "constant")).strip().lower()
    if schedule in {"", "constant", "none"}:
        return base_lr

    warmup_iterations = max(0, int(getattr(config, "lr_warmup_iterations", 0)))
    min_factor = max(0.0, min(1.0, float(getattr(config, "lr_min_factor", 0.0))))
    min_lr = base_lr * min_factor

    if warmup_iterations > 0 and iteration <= warmup_iterations:
        return max(min_lr, base_lr * float(iteration) / float(max(1, warmup_iterations)))

    if schedule in {"cosine", "cosine_decay"}:
        total_iterations = max(1, int(config.outer_iterations))
        if total_iterations <= warmup_iterations:
            return base_lr
        progress = float(iteration - warmup_iterations) / float(max(1, total_iterations - warmup_iterations))
        progress = min(1.0, max(0.0, progress))
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_weight

    raise ValueError(f"Unsupported reconstruction_solver.lr_schedule={schedule!r}.")


def measurement_loss(
    image_bchw: torch.Tensor,
    *,
    measurements: torch.Tensor,
    measurement_operator: MeasurementOperator,
    sigma_y: float,
    ls_weights: Optional[torch.Tensor],
    loss_reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weighted least squares and return raw and weighted residuals."""

    image_chw = image_bchw[0].to(dtype=torch.float32)
    raw_residual = measurement_operator.A(image_chw) - measurements
    weighted_residual = raw_residual if ls_weights is None else raw_residual * ls_weights
    squared = (
        torch.real(weighted_residual.conj() * weighted_residual)
        if torch.is_complex(weighted_residual)
        else weighted_residual.square()
    )
    if sigma_y > 0.0:
        squared = squared / float(sigma_y * sigma_y)
    reduction = str(loss_reduction).strip().lower()
    if reduction in {"mean", "measurement_mean", "legacy_mean"}:
        objective = 0.5 * torch.mean(squared)
    elif reduction in {"measurement_sum_channel_mean", "sum_measurements"}:
        objective = 0.5 * squared.reshape(measurement_operator.C, measurement_operator.m).sum(dim=1).mean()
    else:
        raise ValueError(f"Unsupported reconstruction_solver.loss_reduction={loss_reduction!r}.")
    return objective, raw_residual, weighted_residual


def residual_l2_norm(residual: torch.Tensor) -> float:
    """Return the Euclidean norm of a real or complex measurement residual."""

    if torch.is_complex(residual):
        return float(torch.linalg.vector_norm(residual).item())
    return float(residual.norm(p=2).item())


def atomic_save_optimization_trace(
    path: str | Path,
    *,
    iterations: list[int],
    losses: list[float],
    data_losses: list[float],
    reg_losses: list[float],
    grad_norms: list[float],
    raw_residuals: list[float],
    weighted_residuals: list[float],
    learning_rates: list[float],
    best_iteration: int,
    best_loss: float,
    target_iterations: int,
    configured_learning_rate: float,
    loss_reduction: str,
    complete: bool,
) -> None:
    """Atomically expose scalar optimizer progress to live analysis notebooks."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        bp_iter=np.asarray(iterations, dtype=np.int64),
        bp_loss=np.asarray(losses, dtype=np.float64),
        bp_data_loss=np.asarray(data_losses, dtype=np.float64),
        bp_reg_loss=np.asarray(reg_losses, dtype=np.float64),
        bp_grad_norm=np.asarray(grad_norms, dtype=np.float64),
        bp_raw_resid_l2=np.asarray(raw_residuals, dtype=np.float64),
        bp_weighted_resid_l2=np.asarray(weighted_residuals, dtype=np.float64),
        bp_learning_rate=np.asarray(learning_rates, dtype=np.float64),
        bp_best_iter=np.asarray(best_iteration, dtype=np.int64),
        bp_best_loss=np.asarray(best_loss, dtype=np.float64),
        bp_completed_iterations=np.asarray(len(iterations), dtype=np.int64),
        target_iterations=np.asarray(target_iterations, dtype=np.int64),
        learning_rate=np.asarray(configured_learning_rate, dtype=np.float64),
        loss_reduction=np.asarray(str(loss_reduction)),
        complete=np.asarray(int(bool(complete)), dtype=np.int8),
    )
    os.replace(temporary, destination)


def atomic_save_optimizer_checkpoint(
    path: str | Path,
    *,
    optimized_latents: torch.Tensor,
    best_latents: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    completed_iteration: int,
    target_iterations: int,
    best_iteration: int,
    best_loss: float,
    configured_learning_rate: float,
    loss_reduction: str,
) -> None:
    """Atomically save enough state for an exact future Adam continuation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "optimized_latents": optimized_latents.detach().cpu(),
            "best_latents": best_latents.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_iteration": int(completed_iteration),
            "target_iterations": int(target_iterations),
            "best_iteration": int(best_iteration),
            "best_loss": float(best_loss),
            "learning_rate": float(configured_learning_rate),
            "loss_reduction": str(loss_reduction),
        },
        temporary,
    )
    os.replace(temporary, destination)


def build_ls_weights(
    *,
    weighted_ls: bool,
    probabilities: Optional[np.ndarray],
    measurement_operator: MeasurementOperator,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build inverse-probability least-squares weights for active measurements."""

    if not bool(weighted_ls):
        return None
    if probabilities is None:
        prob_selected = np.full(
            int(measurement_operator.m),
            1.0 / float(measurement_operator.N),
            dtype=np.float64,
        )
    else:
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if values.shape != (int(measurement_operator.N),):
            raise ValueError(
                f"Sampling probability map has shape {values.shape}, "
                f"expected {(int(measurement_operator.N),)}."
            )
        prob_selected = values[measurement_operator.inds_t.detach().cpu().numpy()]
    if not np.all(np.isfinite(prob_selected)) or np.any(prob_selected <= 0.0):
        raise ValueError("Weighted least squares requires finite, strictly positive selected probabilities.")
    weights = 1.0 / np.sqrt(prob_selected)
    weights_t = torch.from_numpy(np.tile(weights, measurement_operator.C).astype(np.float32))
    return weights_t.to(device=device)


def _box_blur_hwc(image: np.ndarray) -> np.ndarray:
    """Apply a small 3x3 box blur to an HWC image using NumPy only."""

    if image.ndim == 2:
        image = image[:, :, None]

    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    neighbors = [
        padded[0:-2, 0:-2, :],
        padded[0:-2, 1:-1, :],
        padded[0:-2, 2:, :],
        padded[1:-1, 0:-2, :],
        padded[1:-1, 1:-1, :],
        padded[1:-1, 2:, :],
        padded[2:, 0:-2, :],
        padded[2:, 1:-1, :],
        padded[2:, 2:, :],
    ]
    return np.mean(np.stack(neighbors, axis=0), axis=0)


def grain_score(image_hwc: np.ndarray) -> float:
    """Estimate a simple grain score by comparing an image to a lightly smoothed version."""

    try:
        from scipy.ndimage import gaussian_filter

        if image_hwc.ndim == 2:
            blurred = gaussian_filter(image_hwc, sigma=1.0)
        else:
            blurred = gaussian_filter(image_hwc, sigma=(1.0, 1.0, 0.0))
    except Exception:
        blurred = _box_blur_hwc(image_hwc)
    return float(np.mean(np.abs(image_hwc - blurred)))


def prepare_timesteps(pipe, latents: torch.Tensor, num_steps: int):
    """Prepare scheduler timesteps for a latent tensor and step count."""

    pipe.scheduler.set_timesteps(int(num_steps), device=latents.device)
    return pipe.scheduler.timesteps


def run_diffusion_backprop_reconstruction(
    pipe,
    *,
    prompt_embeddings,
    generation,
    measurement_operator: MeasurementOperator,
    measurements: torch.Tensor,
    config,
    optim_cfg,
    weighted_ls: bool,
    probabilities: Optional[np.ndarray],
    trace_snapshot_path: Optional[str | Path] = None,
    optimizer_checkpoint_path: Optional[str | Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Optimize the initial latent by backpropagating through the complete denoising chain."""

    random_latents = make_latents(pipe, measurement_operator.H, measurement_operator.W, batch_size=1, dtype=torch.float32)
    timesteps_full = prepare_timesteps(pipe, random_latents, generation.num_steps)
    initial_latents = random_latents.detach()
    init_mode = "random"
    init_timestep_value = float(timesteps_full[0].item()) if len(timesteps_full) > 0 and torch.is_tensor(timesteps_full[0]) else (
        float(timesteps_full[0]) if len(timesteps_full) > 0 else -1.0
    )

    if bool(config.init_from_meas_backproj):
        # A zero-filled inverse FFT can be mixed into the initial latent to warm
        # start optimization; paper configs can disable this for random starts.
        backproj = measurement_operator.zero_filled(measurements)
        backproj = backproj.to(dtype=torch.float32).clamp(0.0, 1.0)
        backproj_latents = encode_image_to_latents(pipe, backproj)
        if len(timesteps_full) > 0:
            noised_backproj = noise_latents_at_timestep(pipe, backproj_latents, timesteps_full[0])
            blend = float(max(0.0, min(1.0, float(config.backproj_init_strength))))
            initial_latents = ((1.0 - blend) * initial_latents + blend * noised_backproj).detach()
            init_mode = "backproj_noised_mix"
        else:
            initial_latents = backproj_latents.detach()
            init_mode = "backproj_latent"

    resume_latents_path = str(getattr(config, "resume_latents_path", "") or "").strip()
    resume_optimizer_path = str(getattr(config, "resume_optimizer_path", "") or "").strip()
    if resume_latents_path and resume_optimizer_path:
        raise ValueError("Specify only one of resume_latents_path and resume_optimizer_path.")
    resumed_optimizer_payload: Optional[Dict[str, Any]] = None
    if resume_optimizer_path:
        checkpoint_path = Path(resume_optimizer_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Optimizer continuation checkpoint does not exist: {checkpoint_path}")
        resumed_optimizer_payload = torch.load(checkpoint_path, map_location=initial_latents.device, weights_only=True)
        if not isinstance(resumed_optimizer_payload, dict) or "optimized_latents" not in resumed_optimizer_payload:
            raise TypeError(f"Invalid optimizer continuation checkpoint: {checkpoint_path}")
        resumed_latents = resumed_optimizer_payload["optimized_latents"]
        if not torch.is_tensor(resumed_latents):
            raise TypeError(f"Optimizer checkpoint latent must be a tensor: {checkpoint_path}")
        resumed_latents = resumed_latents.to(device=initial_latents.device, dtype=initial_latents.dtype)
        if tuple(resumed_latents.shape) != tuple(initial_latents.shape):
            raise ValueError(
                f"Optimizer checkpoint latent has shape {tuple(resumed_latents.shape)}, "
                f"expected {tuple(initial_latents.shape)}."
            )
        initial_latents = resumed_latents.detach().clone()
        init_mode = "saved_current_latent_resumed_adam"
    if resume_latents_path:
        resume_path = Path(resume_latents_path).expanduser()
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"Continuation latent does not exist: {resume_path}")
        resumed_latents = torch.load(resume_path, map_location=initial_latents.device, weights_only=True)
        if not torch.is_tensor(resumed_latents):
            raise TypeError(f"Continuation latent must be a tensor: {resume_path}")
        resumed_latents = resumed_latents.to(device=initial_latents.device, dtype=initial_latents.dtype)
        if tuple(resumed_latents.shape) != tuple(initial_latents.shape):
            raise ValueError(
                f"Continuation latent has shape {tuple(resumed_latents.shape)}, "
                f"expected {tuple(initial_latents.shape)}."
            )
        initial_latents = resumed_latents.detach().clone()
        init_mode = "saved_best_latent_fresh_adam"

    ls_weights = build_ls_weights(
        weighted_ls=bool(weighted_ls),
        probabilities=probabilities,
        measurement_operator=measurement_operator,
        device=measurements.device,
    )

    optimized_latents = torch.nn.Parameter(initial_latents.clone())
    optimizer = torch.optim.Adam(
        [optimized_latents],
        lr=float(config.learning_rate),
        betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
        eps=float(optim_cfg.adam_eps),
    )
    if resumed_optimizer_payload is not None:
        optimizer.load_state_dict(resumed_optimizer_payload["optimizer_state_dict"])

    iter_ids: list[int] = []
    total_losses: list[float] = []
    data_losses: list[float] = []
    reg_losses: list[float] = []
    grad_norms: list[float] = []
    residual_l2: list[float] = []
    raw_residual_l2: list[float] = []
    weighted_residual_l2: list[float] = []
    learning_rates: list[float] = []

    best_loss = math.inf
    best_iter = -1
    best_latents = initial_latents.clone()
    no_improve_iters = 0
    early_stop_iter = -1
    early_stop_reason = ""
    early_stop_patience = max(0, int(getattr(config, "early_stop_patience", 0)))
    early_stop_min_rel_improvement = max(0.0, float(getattr(config, "early_stop_min_rel_improvement", 0.0)))
    iteration_offset = max(0, int(getattr(config, "iteration_offset", 0)))
    if resumed_optimizer_payload is not None:
        checkpoint_iteration = int(resumed_optimizer_payload.get("completed_iteration", -1))
        if checkpoint_iteration != iteration_offset:
            raise ValueError(
                f"Optimizer checkpoint ends at iteration {checkpoint_iteration}, "
                f"but iteration_offset={iteration_offset}."
            )
        checkpoint_best = resumed_optimizer_payload.get("best_latents")
        if torch.is_tensor(checkpoint_best):
            best_latents = checkpoint_best.to(device=initial_latents.device, dtype=initial_latents.dtype).detach().clone()
        best_loss = float(resumed_optimizer_payload.get("best_loss", math.inf))
        best_iter = int(resumed_optimizer_payload.get("best_iteration", -1))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    target_iterations = iteration_offset + int(config.outer_iterations)
    for local_iteration in range(1, int(config.outer_iterations) + 1):
        iteration = iteration_offset + local_iteration
        # Each outer iteration unrolls the full denoising chain, compares the
        # decoded image to Fourier measurements, and updates the initial latent.
        current_lr = diffusion_backprop_learning_rate(local_iteration, config=config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(current_lr)
        optimizer.zero_grad(set_to_none=True)
        image_bchw = unrolled_image_from_latents(
            pipe,
            optimized_latents,
            prompt_embeddings,
            generation,
            use_checkpoint=bool(config.checkpoint_denoiser),
        )
        data_loss, raw_residual, weighted_residual = measurement_loss(
            image_bchw,
            measurements=measurements,
            measurement_operator=measurement_operator,
            sigma_y=float(config.sigma_y),
            ls_weights=ls_weights,
            loss_reduction=str(getattr(config, "loss_reduction", "mean")),
        )
        reg_loss = 0.5 * float(config.latent_l2_penalty) * torch.mean((optimized_latents - initial_latents).square())
        total_loss = data_loss + reg_loss
        total_loss.backward()

        grad_norm = 0.0
        with torch.no_grad():
            if optimized_latents.grad is not None:
                gradient = optimized_latents.grad
                grad_norm = float(gradient.norm(p=2).item())
                if bool(config.normalize_grad) and grad_norm > 0.0:
                    # Normalization and clipping are optional guards for very
                    # high-gradient runs; they are controlled entirely by config.
                    gradient.div_(float(grad_norm + 1e-12))
                if float(config.grad_clip) > 0.0:
                    clipped_norm = float(gradient.norm(p=2).item())
                    if clipped_norm > float(config.grad_clip):
                        gradient.mul_(float(config.grad_clip) / float(clipped_norm + 1e-12))

        optimizer.step()

        total_loss_value = float(total_loss.item())
        data_loss_value = float(data_loss.item())
        reg_loss_value = float(reg_loss.item())
        raw_residual_value = residual_l2_norm(raw_residual.detach())
        weighted_residual_value = residual_l2_norm(weighted_residual.detach())
        residual_value = weighted_residual_value

        iter_ids.append(int(iteration))
        total_losses.append(total_loss_value)
        data_losses.append(data_loss_value)
        reg_losses.append(reg_loss_value)
        grad_norms.append(float(grad_norm))
        residual_l2.append(float(residual_value))
        raw_residual_l2.append(float(raw_residual_value))
        weighted_residual_l2.append(float(weighted_residual_value))
        learning_rates.append(float(current_lr))

        min_improvement = 0.0 if not math.isfinite(best_loss) else abs(best_loss) * early_stop_min_rel_improvement
        if total_loss_value < best_loss - min_improvement:
            # Keep the best latent rather than the last latent so noisy
            # optimization tails do not degrade the reported reconstruction.
            best_loss = total_loss_value
            best_iter = int(iteration)
            best_latents = optimized_latents.detach().clone()
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        trace_save_every = max(0, int(getattr(config, "trace_save_every", 0)))
        if (
            trace_snapshot_path is not None
            and trace_save_every > 0
            and (local_iteration == 1 or iteration % trace_save_every == 0)
        ):
            atomic_save_optimization_trace(
                trace_snapshot_path,
                iterations=iter_ids,
                losses=total_losses,
                data_losses=data_losses,
                reg_losses=reg_losses,
                grad_norms=grad_norms,
                raw_residuals=raw_residual_l2,
                weighted_residuals=weighted_residual_l2,
                learning_rates=learning_rates,
                best_iteration=best_iter,
                best_loss=best_loss,
                target_iterations=target_iterations,
                configured_learning_rate=float(config.learning_rate),
                loss_reduction=str(getattr(config, "loss_reduction", "mean")),
                complete=False,
            )
            if optimizer_checkpoint_path is not None:
                atomic_save_optimizer_checkpoint(
                    optimizer_checkpoint_path,
                    optimized_latents=optimized_latents,
                    best_latents=best_latents,
                    optimizer=optimizer,
                    completed_iteration=iteration,
                    target_iterations=target_iterations,
                    best_iteration=best_iter,
                    best_loss=best_loss,
                    configured_learning_rate=float(config.learning_rate),
                    loss_reduction=str(getattr(config, "loss_reduction", "mean")),
                )

        if int(config.log_every) > 0 and (local_iteration == 1 or iteration % int(config.log_every) == 0):
            stale_text = (
                f" | stale {no_improve_iters:03d}/{early_stop_patience:03d}"
                if early_stop_patience > 0
                else ""
            )
            print(
                f"  opt {iteration:04d}/{target_iterations:04d} | "
                f"lr {current_lr:9.3e} | "
                f"loss {total_loss_value:9.3e} | "
                f"data {data_loss_value:9.3e} | "
                f"resid {residual_value:9.3e} | "
                f"grad {grad_norm:9.3e} | "
                f"best {best_loss:9.3e} @ {best_iter:03d}"
                f"{stale_text}"
            )

        del image_bchw, raw_residual, weighted_residual, total_loss, data_loss, reg_loss
        if local_iteration % 5 == 0:
            gc.collect()
            safe_empty_cuda_cache()

        if early_stop_patience > 0 and no_improve_iters >= early_stop_patience:
            early_stop_iter = int(iteration)
            early_stop_reason = f"no_best_improvement_for_{early_stop_patience}_iters"
            print(
                f"  early stop @ {early_stop_iter:03d}: "
                f"best loss {best_loss:.3e} has not improved enough since iter {best_iter:03d}"
            )
            break

    if trace_snapshot_path is not None and int(getattr(config, "trace_save_every", 0)) > 0:
        atomic_save_optimization_trace(
            trace_snapshot_path,
            iterations=iter_ids,
            losses=total_losses,
            data_losses=data_losses,
            reg_losses=reg_losses,
            grad_norms=grad_norms,
            raw_residuals=raw_residual_l2,
            weighted_residuals=weighted_residual_l2,
            learning_rates=learning_rates,
            best_iteration=best_iter,
            best_loss=best_loss,
            target_iterations=target_iterations,
            configured_learning_rate=float(config.learning_rate),
            loss_reduction=str(getattr(config, "loss_reduction", "mean")),
            complete=True,
        )
        if optimizer_checkpoint_path is not None:
            atomic_save_optimizer_checkpoint(
                optimizer_checkpoint_path,
                optimized_latents=optimized_latents,
                best_latents=best_latents,
                optimizer=optimizer,
                completed_iteration=iter_ids[-1],
                target_iterations=target_iterations,
                best_iteration=best_iter,
                best_loss=best_loss,
                configured_learning_rate=float(config.learning_rate),
                loss_reduction=str(getattr(config, "loss_reduction", "mean")),
            )

    with torch.no_grad():
        # Decode the selected best latent once without checkpointing for the
        # final image and metrics.
        image_rec = unrolled_image_from_latents(
            pipe,
            best_latents,
            prompt_embeddings,
            generation,
            use_checkpoint=False,
        )[0].to(dtype=torch.float32)
        latents = best_latents.detach()
        final_data_loss, final_raw_residual, final_weighted_residual = measurement_loss(
            image_rec.unsqueeze(0),
            measurements=measurements,
            measurement_operator=measurement_operator,
            sigma_y=float(config.sigma_y),
            ls_weights=ls_weights,
            loss_reduction=str(getattr(config, "loss_reduction", "mean")),
        )

    if ls_weights is None:
        ls_weight_min = 1.0
        ls_weight_max = 1.0
        ls_weights_rgb = np.ones(
            int(measurement_operator.C) * int(measurement_operator.m),
            dtype=np.float32,
        )
    else:
        ls_weight_min = float(ls_weights.min().item())
        ls_weight_max = float(ls_weights.max().item())
        ls_weights_rgb = ls_weights.detach().to(dtype=torch.float32, device="cpu").numpy()
    operator_row_weights_rgb = ls_weights_rgb / math.sqrt(
        float(measurement_operator.m)
    )

    traces = {
        # These arrays are saved into run_data.npz and later consumed by the
        # analysis notebooks for convergence diagnostics.
        "reconstruction_solver": "sd15_backprop",
        "sigma_y": float(config.sigma_y),
        "init_mode": init_mode,
        "init_timestep": float(init_timestep_value),
        "outer_iterations": int(config.outer_iterations),
        "learning_rate": float(config.learning_rate),
        "lr_schedule": str(getattr(config, "lr_schedule", "constant")),
        "lr_warmup_iterations": int(getattr(config, "lr_warmup_iterations", 0)),
        "lr_min_factor": float(getattr(config, "lr_min_factor", 0.0)),
        "loss_reduction": str(getattr(config, "loss_reduction", "mean")),
        "trace_save_every": int(getattr(config, "trace_save_every", 0)),
        "resume_latents_path": resume_latents_path,
        "resume_optimizer_path": resume_optimizer_path,
        "iteration_offset": iteration_offset,
        "optimizer_state_resumed": int(resumed_optimizer_payload is not None),
        "latent_l2_penalty": float(config.latent_l2_penalty),
        "normalize_grad": int(bool(config.normalize_grad)),
        "grad_clip": float(config.grad_clip),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_rel_improvement": float(early_stop_min_rel_improvement),
        "early_stop_iter": int(early_stop_iter),
        "early_stop_reason": early_stop_reason,
        "checkpoint_denoiser": int(bool(config.checkpoint_denoiser)),
        "weighted_ls": int(bool(weighted_ls)),
        "fft_normalization": str(measurement_operator.fft_normalization),
        "ls_weight_min": float(ls_weight_min),
        "ls_weight_max": float(ls_weight_max),
        "ls_weights_selected": np.asarray(
            ls_weights_rgb[: int(measurement_operator.m)],
            dtype=np.float32,
        ),
        "ls_weights_rgb": np.asarray(ls_weights_rgb, dtype=np.float32),
        "operator_row_weight_min": float(operator_row_weights_rgb.min()),
        "operator_row_weight_max": float(operator_row_weights_rgb.max()),
        "operator_row_weights_selected": np.asarray(
            operator_row_weights_rgb[: int(measurement_operator.m)],
            dtype=np.float32,
        ),
        "operator_row_weights_rgb": np.asarray(
            operator_row_weights_rgb,
            dtype=np.float32,
        ),
        "final_data_loss": float(final_data_loss.item()),
        "final_raw_resid_l2": residual_l2_norm(final_raw_residual),
        "final_weighted_resid_l2": residual_l2_norm(final_weighted_residual),
        "bp_iter": np.asarray(iter_ids, dtype=np.int64),
        "bp_loss": np.asarray(total_losses, dtype=np.float64),
        "bp_data_loss": np.asarray(data_losses, dtype=np.float64),
        "bp_reg_loss": np.asarray(reg_losses, dtype=np.float64),
        "bp_learning_rate": np.asarray(learning_rates, dtype=np.float64),
        "bp_grad_norm": np.asarray(grad_norms, dtype=np.float64),
        "bp_resid_l2": np.asarray(residual_l2, dtype=np.float64),
        "bp_raw_resid_l2": np.asarray(raw_residual_l2, dtype=np.float64),
        "bp_weighted_resid_l2": np.asarray(weighted_residual_l2, dtype=np.float64),
        "bp_completed_iterations": int(len(iter_ids)),
        "bp_best_iter": int(best_iter),
        "bp_best_loss": float(best_loss),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    return image_rec, latents, traces


def run_single_reconstruction(
    cfg: RunConfig,
    pipe,
    dataset_item: Dict[str, Any],
    samp_method: int,
    samp_perc: float,
    repeat_id: int,
    parent_run_dir: str,
    *,
    prompt_text: str,
    prompt_embeddings,
    probabilities: Optional[np.ndarray],
    ktilde_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single diffusion-backprop reconstruction."""

    height = int(cfg.image.height)
    width = int(cfg.image.width)
    num_pixels = int(height * width)

    if "gt_png_path" not in dataset_item:
        raise ValueError("Dataset items must include gt_png_path for reconstruction.")
    image_true_np = load_gt_image_chw(dataset_item["gt_png_path"], height, width)
    model_device = next(pipe.unet.parameters()).device if hasattr(pipe, "unet") else torch.device("cpu")
    image_true = torch.from_numpy(image_true_np).to(device=model_device, dtype=torch.float32)
    channels = int(image_true.shape[0])

    m_coeffs = int(np.round(float(samp_perc) * num_pixels))
    m_coeffs = max(1, min(m_coeffs, num_pixels))
    # Fold item id, repeat id, sampling percentage, and sampler id into the seed
    # so every leaf run is deterministic and distinct.
    repeat_seed = int(
        cfg.repro.seed
        + 10_000 * int(dataset_item["item_id"])
        + 100 * int(repeat_id)
        + int(1e6 * float(samp_perc))
        + 1_000_000 * int(samp_method)
    )
    np.random.seed(repeat_seed)
    torch.manual_seed(repeat_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(repeat_seed)

    indices_np, _, prob_used, sampling_metadata = build_sampling_pattern(
        N=num_pixels,
        m=m_coeffs,
        samp_method=int(samp_method),
        prob=probabilities,
        H=height,
        W=width,
        vd_params=cfg.sampling.vd_params,
        return_metadata=True,
    )
    measurement_operator = MeasurementOperator(
        inds_np=indices_np,
        N=num_pixels,
        C=channels,
        H=height,
        W=width,
        fft_normalization=str(cfg.sampling.fft_normalization),
    )
    # Measurements are generated directly from the saved ground-truth image.
    measurements = measurement_operator.A(image_true)

    sigma_y = float(cfg.reconstruction_solver.sigma_y)
    if sigma_y > 0.0:
        # Optional synthetic measurement noise follows the configured sigma_y.
        if torch.is_complex(measurements):
            noise = torch.randn_like(measurements.real) + 1j * torch.randn_like(measurements.real)
            noise = noise / math.sqrt(2.0)
            measurements = measurements + sigma_y * noise
        else:
            measurements = measurements + sigma_y * torch.randn_like(measurements)

    method_folder = sampling_method_folder(samp_method)
    item_id = int(dataset_item["item_id"])
    sample_tag = f"samp_{float(samp_perc):.5f}".replace(".", "p")
    repeat_tag = f"rep_{int(repeat_id):02d}"
    run_dir = Path(parent_run_dir) / f"item_{item_id:03d}" / sample_tag / repeat_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_snapshot_path = (
        run_dir / "optimization_trace.npz"
        if int(getattr(cfg.reconstruction_solver, "trace_save_every", 0)) > 0
        else None
    )
    optimizer_checkpoint_path = (
        run_dir / "optimizer_checkpoint.pt"
        if int(getattr(cfg.reconstruction_solver, "trace_save_every", 0)) > 0
        else None
    )

    start_time = time.time()
    print(
        f"\n[recon start] prompt={prompt_text or '<unprompted>'!r} "
        f"item={int(dataset_item['item_id']):03d} samp={float(samp_perc):.5f} "
        f"rep={int(repeat_id):02d} m={int(m_coeffs)}"
    )
    image_rec_t, latents_rec, traces = run_diffusion_backprop_reconstruction(
        pipe,
        prompt_embeddings=prompt_embeddings,
        generation=cfg.gen_recon,
        measurement_operator=measurement_operator,
        measurements=measurements,
        config=cfg.reconstruction_solver,
        optim_cfg=cfg.optim,
        weighted_ls=bool(cfg.sampling.weighted_ls),
        probabilities=prob_used,
        trace_snapshot_path=trace_snapshot_path,
        optimizer_checkpoint_path=optimizer_checkpoint_path,
    )
    runtime = time.time() - start_time

    image_rec = image_rec_t.detach().to(dtype=torch.float32, device="cpu").numpy()
    image_true_display = chw_to_hwc_for_display(np.nan_to_num(image_true_np, nan=0.0, posinf=1.0, neginf=0.0))
    image_rec_display = chw_to_hwc_for_display(np.nan_to_num(image_rec, nan=0.0, posinf=1.0, neginf=0.0))
    psnr_value = calculate_psnr(255.0 * image_true_display, 255.0 * image_rec_display, max_value=255.0)
    try:
        from skimage.metrics import structural_similarity as skimage_ssim

        try:
            ssim_value = float(skimage_ssim(image_true_display, image_rec_display, multichannel=True, data_range=1.0))
        except TypeError:
            ssim_value = float(skimage_ssim(image_true_display, image_rec_display, channel_axis=2, data_range=1.0))
    except Exception:
        ssim_value = calculate_ssim(255.0 * image_true_display, 255.0 * image_rec_display, max_value=255.0)
    grain_value = grain_score(image_rec_display)
    pixel_mae_value = float(np.mean(np.abs(image_true_display - image_rec_display)))

    conditioning_mode = "unconditioned" if str(prompt_text) == "" else "prompt"
    # The directory names are intentionally stable because scripts and notebooks
    # use them to find completed runs.

    # The zero-filled reconstruction is not optimized; it is saved as a simple
    # Fourier-adjoint baseline for each exact mask.
    zero_filled_t = measurement_operator.zero_filled(measurements)
    zero_filled_display = chw_to_hwc_for_display(
        np.nan_to_num(
            zero_filled_t.detach().to(dtype=torch.float32, device="cpu").numpy(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    )
    zero_filled_display = np.clip(zero_filled_display, 0.0, 1.0)
    zf_psnr_value = calculate_psnr(255.0 * image_true_display, 255.0 * zero_filled_display, max_value=255.0)
    try:
        zf_ssim_value = float(skimage_ssim(image_true_display, zero_filled_display, channel_axis=2, data_range=1.0))
    except Exception:
        zf_ssim_value = calculate_ssim(255.0 * image_true_display, 255.0 * zero_filled_display, max_value=255.0)
    zf_grain_value = grain_score(zero_filled_display)
    zf_pixel_mae_value = float(np.mean(np.abs(image_true_display - zero_filled_display)))

    run_data: Dict[str, Any] = {
        # Keep scalar metadata and metrics together so npz/csv tables are enough
        # for most analysis without reopening images.
        "item_id": item_id,
        "prompt_text": prompt_text,
        "prompt_sha256": sha256_text(prompt_text),
        "conditioning_mode": conditioning_mode,
        "H": height,
        "W": width,
        "C": channels,
        "N": num_pixels,
        "samp_perc": float(samp_perc),
        "samp_method": int(samp_method),
        "m_coeffs": int(m_coeffs),
        "sampled_indices": indices_np.astype(np.int64),
        "sampled_probabilities": np.asarray(prob_used, dtype=np.float64)[indices_np],
        "sampling_probability_min": float(np.min(prob_used)),
        "sampling_probability_max": float(np.max(prob_used)),
        "sampling_probability_sum": float(np.sum(prob_used)),
        "probability_regularization_zeta": float(cfg.sampling.probability_regularization_zeta),
        "fft_normalization": str(cfg.sampling.fft_normalization),
        "ktilde_name": str(cfg.ktilde.name),
        "ktilde_max_samples": int((ktilde_metadata or {}).get("max_samples", 0)),
        "repeat_id": int(repeat_id),
        "rep_seed": int(repeat_seed),
        "runtime_sec": float(runtime),
        "psnr_db": float(psnr_value),
        "ssim": float(ssim_value),
        "pixel_mae": float(pixel_mae_value),
        "grain": float(grain_value),
        "zero_filled_psnr_db": float(zf_psnr_value),
        "zero_filled_ssim": float(zf_ssim_value),
        "zero_filled_pixel_mae": float(zf_pixel_mae_value),
        "zero_filled_grain": float(zf_grain_value),
        "recon_num_steps": int(cfg.gen_recon.num_steps),
        "method": method_folder,
    }
    for key, value in sampling_metadata.items():
        if isinstance(value, (np.number, float, int, str, bool)):
            run_data[f"sampling_design_{key}"] = value
    run_data.update(traces)

    if bool(cfg.output.save_json) and bool(cfg.sweep.save_per_run_artifacts):
        json_dump(run_dir / "run_config.json", run_config_to_dict(cfg))
        json_dump(run_dir / "dataset_item.json", dataset_item)
        json_dump(run_dir / "sampling_pattern.json", sampling_metadata)

    torch.save(latents_rec.detach().cpu(), str(run_dir / "z_rec.pt"))
    if bool(cfg.output.save_npz) and bool(cfg.sweep.save_per_run_artifacts):
        # Only scalar/array/string fields are serializable in the compact npz.
        np.savez_compressed(
            str(run_dir / "run_data.npz"),
            **{key: value for key, value in run_data.items() if isinstance(value, (np.ndarray, np.number, float, int, str))},
        )
    if bool(cfg.output.save_mat) and bool(cfg.sweep.save_per_run_artifacts):
        import scipy.io as sio

        sio.savemat(
            str(run_dir / "run_data.mat"),
            {
                "psnr_db": np.array([[psnr_value]], dtype=np.float64),
                "ssim": np.array([[ssim_value]], dtype=np.float64),
                "pixel_mae": np.array([[pixel_mae_value]], dtype=np.float64),
                "prompt_text": np.array([prompt_text], dtype=object),
                "samp_perc": np.array([[float(samp_perc)]], dtype=np.float64),
            },
        )
    if bool(cfg.output.save_images):
        import matplotlib.pyplot as plt

        plt.imsave(
            str(run_dir / f"recon_{method_folder}.png"),
            image_rec_display if image_rec_display.ndim == 3 else image_rec_display,
            cmap=None if image_rec_display.ndim == 3 else "gray",
        )
        plt.imsave(
            str(run_dir / "zero_filled_ifft.png"),
            zero_filled_display if zero_filled_display.ndim == 3 else zero_filled_display,
            cmap=None if zero_filled_display.ndim == 3 else "gray",
        )
    if bool(cfg.output.plot_images):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 4))
        plt.imshow(image_rec_display if image_rec_display.ndim == 3 else image_rec_display, cmap=None if image_rec_display.ndim == 3 else "gray")
        plt.axis("off")
        plt.show()

    with open(run_dir / "run_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("=== Reconstruction Run Summary ===\n\n")
        handle.write(f"method: {method_folder}\n")
        handle.write("reconstruction_solver: sd15_backprop\n")
        handle.write(f"item_id: {item_id}\n")
        handle.write(f"conditioning_mode: {conditioning_mode}\n")
        handle.write(f"prompt_text: {prompt_text}\n")
        handle.write(f"samp_perc: {float(samp_perc)}\n")
        handle.write(f"repeat_id: {int(repeat_id)}\n")
        handle.write(f"ktilde_name: {cfg.ktilde.name}\n")
        handle.write(f"ktilde_max_samples: {int((ktilde_metadata or {}).get('max_samples', 0))}\n")
        handle.write(f"fft_normalization: {cfg.sampling.fft_normalization}\n")
        handle.write(f"weighted_ls: {bool(cfg.sampling.weighted_ls)}\n")
        handle.write(f"probability_regularization_zeta: {float(cfg.sampling.probability_regularization_zeta)}\n")
        handle.write(f"sampling_probability_min: {float(np.min(prob_used)):.12e}\n")
        handle.write(f"sampling_probability_max: {float(np.max(prob_used)):.12e}\n")
        handle.write(f"sampling_law: {sampling_metadata.get('sampling_law', '')}\n")
        for key in (
            "disk_target_count",
            "disk_count",
            "outside_count",
            "disk_radius_pixels",
            "inverse_square_count",
            "uniform_count",
        ):
            if key in sampling_metadata:
                handle.write(f"{key}: {sampling_metadata[key]}\n")
        handle.write(f"ls_weight_min: {float(run_data['ls_weight_min']):.12e}\n")
        handle.write(f"ls_weight_max: {float(run_data['ls_weight_max']):.12e}\n")
        handle.write(
            "operator_row_weight_min: "
            f"{float(run_data['operator_row_weight_min']):.12e}\n"
        )
        handle.write(
            "operator_row_weight_max: "
            f"{float(run_data['operator_row_weight_max']):.12e}\n"
        )
        handle.write(f"runtime_sec: {runtime:.4f}\n")
        handle.write(f"psnr_db: {psnr_value:.4f}\n")
        handle.write(f"ssim: {ssim_value:.6f}\n")
        handle.write(f"pixel_mae: {pixel_mae_value:.6f}\n")
        handle.write(f"zero_filled_psnr_db: {zf_psnr_value:.4f}\n")
        handle.write(f"zero_filled_ssim: {zf_ssim_value:.6f}\n")
        handle.write(f"zero_filled_pixel_mae: {zf_pixel_mae_value:.6f}\n")
        handle.write(f"final_raw_resid_l2: {float(run_data['final_raw_resid_l2']):.12e}\n")
        handle.write(f"final_weighted_resid_l2: {float(run_data['final_weighted_resid_l2']):.12e}\n")

    del latents_rec, measurements, measurement_operator, image_true, image_rec_t
    gc.collect()
    safe_empty_cuda_cache()
    return run_data
