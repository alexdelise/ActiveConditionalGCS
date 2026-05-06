"""Single-run reconstruction routines for diffusion backpropagation."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .config import RunConfig, active_dc_method, run_config_to_dict, sampling_method_folder
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

    raise ValueError(f"Unsupported diffusion_backprop.lr_schedule={schedule!r}.")


def measurement_loss(
    image_bchw: torch.Tensor,
    *,
    measurements: torch.Tensor,
    measurement_operator: MeasurementOperator,
    sigma_y: float,
    ls_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a weighted least-squares measurement loss and residual."""

    image_chw = image_bchw[0].to(dtype=torch.float32)
    residual = measurement_operator.A(image_chw) - measurements
    if ls_weights is not None:
        # Optional importance weights correct for nonuniform sampling when a
        # weighted least-squares objective is requested.
        residual = residual * ls_weights
    squared = torch.real(residual.conj() * residual) if torch.is_complex(residual) else residual.square()
    if sigma_y > 0.0:
        squared = squared / float(sigma_y * sigma_y)
    return 0.5 * torch.mean(squared), residual


def residual_l2_norm(residual: torch.Tensor) -> float:
    """Return the Euclidean norm of a real or complex measurement residual."""

    if torch.is_complex(residual):
        return float(torch.linalg.vector_norm(residual).item())
    return float(residual.norm(p=2).item())


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
        prob_selected = np.ones(int(measurement_operator.m), dtype=np.float32)
    else:
        # Pull the probabilities only for the active Fourier coefficients and
        # clamp them away from zero before forming inverse-probability weights.
        prob_selected = np.maximum(
            probabilities[measurement_operator.inds_t.detach().cpu().numpy()].astype(np.float32),
            1e-18,
        )
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
        backproj = (float(measurement_operator.m) / float(measurement_operator.N)) * measurement_operator.At(measurements)
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

    iter_ids: list[int] = []
    total_losses: list[float] = []
    data_losses: list[float] = []
    reg_losses: list[float] = []
    grad_norms: list[float] = []
    residual_l2: list[float] = []
    learning_rates: list[float] = []

    best_loss = math.inf
    best_iter = -1
    best_latents = initial_latents.clone()
    no_improve_iters = 0
    early_stop_iter = -1
    early_stop_reason = ""
    early_stop_patience = max(0, int(getattr(config, "early_stop_patience", 0)))
    early_stop_min_rel_improvement = max(0.0, float(getattr(config, "early_stop_min_rel_improvement", 0.0)))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for iteration in range(1, int(config.outer_iterations) + 1):
        # Each outer iteration unrolls the full denoising chain, compares the
        # decoded image to Fourier measurements, and updates the initial latent.
        current_lr = diffusion_backprop_learning_rate(iteration, config=config)
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
        data_loss, residual = measurement_loss(
            image_bchw,
            measurements=measurements,
            measurement_operator=measurement_operator,
            sigma_y=float(config.sigma_y),
            ls_weights=ls_weights,
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
        residual_value = residual_l2_norm(residual.detach())

        iter_ids.append(int(iteration))
        total_losses.append(total_loss_value)
        data_losses.append(data_loss_value)
        reg_losses.append(reg_loss_value)
        grad_norms.append(float(grad_norm))
        residual_l2.append(float(residual_value))
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

        if int(config.log_every) > 0 and (iteration == 1 or iteration % int(config.log_every) == 0):
            stale_text = (
                f" | stale {no_improve_iters:03d}/{early_stop_patience:03d}"
                if early_stop_patience > 0
                else ""
            )
            print(
                f"  opt {iteration:03d}/{int(config.outer_iterations):03d} | "
                f"lr {current_lr:9.3e} | "
                f"loss {total_loss_value:9.3e} | "
                f"data {data_loss_value:9.3e} | "
                f"resid {residual_value:9.3e} | "
                f"grad {grad_norm:9.3e} | "
                f"best {best_loss:9.3e} @ {best_iter:03d}"
                f"{stale_text}"
            )

        del image_bchw, residual, total_loss, data_loss, reg_loss
        if iteration % 5 == 0:
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

    traces = {
        # These arrays are saved into run_data.npz and later consumed by the
        # analysis notebooks for convergence diagnostics.
        "recon_method": "diffusion_backprop",
        "sigma_y": float(config.sigma_y),
        "init_mode": init_mode,
        "init_timestep": float(init_timestep_value),
        "outer_iterations": int(config.outer_iterations),
        "learning_rate": float(config.learning_rate),
        "lr_schedule": str(getattr(config, "lr_schedule", "constant")),
        "lr_warmup_iterations": int(getattr(config, "lr_warmup_iterations", 0)),
        "lr_min_factor": float(getattr(config, "lr_min_factor", 0.0)),
        "latent_l2_penalty": float(config.latent_l2_penalty),
        "normalize_grad": int(bool(config.normalize_grad)),
        "grad_clip": float(config.grad_clip),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_rel_improvement": float(early_stop_min_rel_improvement),
        "early_stop_iter": int(early_stop_iter),
        "early_stop_reason": early_stop_reason,
        "checkpoint_denoiser": int(bool(config.checkpoint_denoiser)),
        "weighted_ls": int(bool(weighted_ls)),
        "bp_iter": np.asarray(iter_ids, dtype=np.int64),
        "bp_loss": np.asarray(total_losses, dtype=np.float64),
        "bp_data_loss": np.asarray(data_losses, dtype=np.float64),
        "bp_reg_loss": np.asarray(reg_losses, dtype=np.float64),
        "bp_learning_rate": np.asarray(learning_rates, dtype=np.float64),
        "bp_grad_norm": np.asarray(grad_norms, dtype=np.float64),
        "bp_resid_l2": np.asarray(residual_l2, dtype=np.float64),
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

    indices_np, _, prob_used = build_sampling_pattern(
        N=num_pixels,
        m=m_coeffs,
        samp_method=int(samp_method),
        prob=probabilities,
        H=height,
        W=width,
    )
    measurement_operator = MeasurementOperator(inds_np=indices_np, N=num_pixels, C=channels, H=height, W=width)
    # Measurements are generated directly from the saved ground-truth image.
    measurements = measurement_operator.A(image_true)

    dc_method = active_dc_method(cfg)
    sigma_y = float(cfg.dc_methods.diffusion_backprop.sigma_y)
    if sigma_y > 0.0:
        # Optional synthetic measurement noise follows the configured sigma_y.
        if torch.is_complex(measurements):
            noise = torch.randn_like(measurements.real) + 1j * torch.randn_like(measurements.real)
            noise = noise / math.sqrt(2.0)
            measurements = measurements + sigma_y * noise
        else:
            measurements = measurements + sigma_y * torch.randn_like(measurements)

    start_time = time.time()
    print(
        f"\n[recon start] method={dc_method} prompt={prompt_text or '<unprompted>'!r} "
        f"item={int(dataset_item['item_id']):03d} samp={float(samp_perc):.5f} "
        f"rep={int(repeat_id):02d} m={int(m_coeffs)}"
    )
    image_rec_t, latents_rec, traces = run_diffusion_backprop_reconstruction(
        pipe,
        prompt_embeddings=prompt_embeddings,
        generation=cfg.gen_recon,
        measurement_operator=measurement_operator,
        measurements=measurements,
        config=cfg.dc_methods.diffusion_backprop,
        optim_cfg=cfg.optim,
        weighted_ls=bool(cfg.sampling.weighted_ls),
        probabilities=prob_used,
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

    method_folder = sampling_method_folder(samp_method)
    item_id = int(dataset_item["item_id"])
    conditioning_mode = "unconditioned" if str(prompt_text) == "" else "prompt"
    # The directory names are intentionally stable because scripts and notebooks
    # use them to find completed runs.
    sample_tag = f"samp_{float(samp_perc):.5f}".replace(".", "p")
    repeat_tag = f"rep_{int(repeat_id):02d}"
    run_dir = Path(parent_run_dir) / f"item_{item_id:03d}" / sample_tag / repeat_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # The zero-filled reconstruction is not optimized; it is saved as a simple
    # Fourier-adjoint baseline for each exact mask.
    zero_filled_t = (float(measurement_operator.m) / float(measurement_operator.N)) * measurement_operator.At(measurements)
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
    run_data.update(traces)

    if bool(cfg.output.save_json) and bool(cfg.sweep.save_per_run_artifacts):
        json_dump(run_dir / "run_config.json", run_config_to_dict(cfg))
        json_dump(run_dir / "dataset_item.json", dataset_item)

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
        handle.write(f"dc_method: {dc_method}\n")
        handle.write(f"item_id: {item_id}\n")
        handle.write(f"conditioning_mode: {conditioning_mode}\n")
        handle.write(f"prompt_text: {prompt_text}\n")
        handle.write(f"samp_perc: {float(samp_perc)}\n")
        handle.write(f"repeat_id: {int(repeat_id)}\n")
        handle.write(f"runtime_sec: {runtime:.4f}\n")
        handle.write(f"psnr_db: {psnr_value:.4f}\n")
        handle.write(f"ssim: {ssim_value:.6f}\n")
        handle.write(f"pixel_mae: {pixel_mae_value:.6f}\n")
        handle.write(f"zero_filled_psnr_db: {zf_psnr_value:.4f}\n")
        handle.write(f"zero_filled_ssim: {zf_ssim_value:.6f}\n")
        handle.write(f"zero_filled_pixel_mae: {zf_pixel_mae_value:.6f}\n")

    del latents_rec, measurements, measurement_operator, image_true, image_rec_t
    gc.collect()
    safe_empty_cuda_cache()
    return run_data
