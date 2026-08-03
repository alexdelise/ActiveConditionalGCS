"""Stable Diffusion 1.5 helpers used by dataset, k-tilde, and reconstruction code."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch

from .config import MODEL_ID, GenerationConfig, RuntimeConfig
from .constants import DEVICE
from .utils import safe_empty_cuda_cache


def _freeze_module(module) -> None:
    """Put a module in eval mode and disable gradients when it exists."""

    if module is None:
        return
    try:
        module.eval()
    except Exception:
        pass
    try:
        module.requires_grad_(False)
    except Exception:
        pass


def offload_text_encoder(pipe) -> None:
    """Move the text encoder back to CPU to reduce steady-state GPU memory pressure."""

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        text_encoder.to("cpu")
        safe_empty_cuda_cache()


def ensure_text_encoder_on_device(pipe, device=DEVICE) -> None:
    """Move the text encoder onto the active device before prompt encoding."""

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        text_encoder.to(device)


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Map a dtype name from config into the matching PyTorch dtype."""

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _scheduler_step(pipe, noise_pred: torch.Tensor, timestep, latents: torch.Tensor, *, eta: float) -> torch.Tensor:
    """Advance the active scheduler while passing only supported keyword arguments."""

    signature = inspect.signature(pipe.scheduler.step)
    kwargs: Dict[str, Any] = {}
    # Diffusers scheduler signatures vary across versions; only pass eta when
    # the installed scheduler exposes it.
    if "eta" in signature.parameters:
        kwargs["eta"] = float(eta)
    return pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False, **kwargs)[0]


def load_sd15_pipeline(runtime: RuntimeConfig):
    """Load the fixed Stable Diffusion 1.5 pipeline configured for this repo."""

    torch_dtype = resolve_torch_dtype(runtime.torch_dtype)

    from diffusers import DDIMScheduler, StableDiffusionPipeline  # type: ignore

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
    }
    # The pinned and fallback diffusers versions differ slightly in accepted
    # kwargs, so loading is intentionally defensive while preserving the model id.
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=False,
            **load_kwargs,
        )
    except TypeError:
        load_kwargs.pop("requires_safety_checker", None)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=False,
            **load_kwargs,
        )
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, **load_kwargs)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

    try:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    # Keep the denoiser/VAE on the active device, but offload the text encoder
    # after prompt encoding because reconstruction optimizes latents only.
    try:
        pipe.to(DEVICE)
    except Exception:
        pass

    unet = getattr(pipe, "unet", None)
    vae = getattr(pipe, "vae", None)
    text_encoder = getattr(pipe, "text_encoder", None)

    if runtime.gradient_checkpointing and unet is not None and hasattr(unet, "enable_gradient_checkpointing"):
        try:
            unet.enable_gradient_checkpointing()
        except Exception:
            pass

    # These memory-saving switches are no-ops when unsupported by the installed
    # diffusers version.
    if runtime.attention_slicing:
        try:
            pipe.enable_attention_slicing(runtime.attention_slicing)
        except Exception:
            pass

    if vae is not None and hasattr(vae, "enable_slicing"):
        try:
            vae.enable_slicing()
        except Exception:
            pass
    elif hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    _freeze_module(unet)
    _freeze_module(vae)
    _freeze_module(text_encoder)
    offload_text_encoder(pipe)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    safe_empty_cuda_cache()
    return pipe


def validate_guidance_scale(value: float, *, field_name: str = "guidance_scale") -> float:
    """Validate an SD-style guidance scale."""

    guidance_scale = float(value)
    if guidance_scale < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return guidance_scale


def encode_prompt(
    pipe,
    prompt: str,
    *,
    guidance_scale: float,
    negative_prompt: str = "",
    offload_after_encode: bool = True,
) -> Dict[str, Any]:
    """Encode one prompt for SD1.5 and keep the tensors needed for CFG denoising."""

    validate_guidance_scale(float(guidance_scale), field_name="generation.guidance_scale")
    ensure_text_encoder_on_device(pipe)
    do_cfg = float(guidance_scale) > 1.0
    with torch.no_grad():
        # Keep prompt encoding outside the optimization graph. Gradients flow
        # through the denoising chain with these embeddings held fixed.
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=str(prompt or ""),
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=bool(do_cfg),
            negative_prompt=None if not do_cfg else str(negative_prompt or ""),
        )
    if offload_after_encode:
        offload_text_encoder(pipe)
    return {
        "prompt_embeds": prompt_embeds.to(device=DEVICE),
        "negative_prompt_embeds": None if negative_prompt_embeds is None else negative_prompt_embeds.to(device=DEVICE),
        "do_classifier_free_guidance": bool(do_cfg),
    }


def _prompt_embeds_for_unet(prompt_embeddings) -> torch.Tensor:
    """Return the encoder hidden states expected by the UNet call."""

    prompt_embeds = prompt_embeddings["prompt_embeds"]
    if not bool(prompt_embeddings.get("do_classifier_free_guidance", False)):
        return prompt_embeds
    negative_prompt_embeds = prompt_embeddings.get("negative_prompt_embeds")
    if negative_prompt_embeds is None:
        raise ValueError("Classifier-free guidance requires negative_prompt_embeds.")
    return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)


@torch.no_grad()
def make_latents(pipe, height: int, width: int, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Create random latent noise with the spatial shape expected by SD1.5."""

    latent_channels = int(pipe.unet.config.in_channels)
    if hasattr(pipe, "prepare_latents"):
        return pipe.prepare_latents(
            batch_size,
            latent_channels,
            height,
            width,
            dtype,
            DEVICE,
            None,
            None,
        ).to(device=DEVICE, dtype=dtype)

    latent_height = int(height) // int(pipe.vae_scale_factor)
    latent_width = int(width) // int(pipe.vae_scale_factor)
    latents = torch.randn((batch_size, latent_channels, latent_height, latent_width), device=DEVICE, dtype=dtype)
    sigma = float(getattr(pipe.scheduler, "init_noise_sigma", 1.0))
    return latents * sigma


@torch.no_grad()
def encode_image_to_latents(pipe, image_chw: torch.Tensor) -> torch.Tensor:
    """Encode an image estimate into the VAE latent space."""

    image_chw = image_chw.to(device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
    image_input = (2.0 * image_chw - 1.0).unsqueeze(0).to(dtype=pipe.vae.dtype)
    latent_dist = pipe.vae.encode(image_input).latent_dist
    if hasattr(latent_dist, "mode"):
        encoded = latent_dist.mode()
    elif hasattr(latent_dist, "mean"):
        encoded = latent_dist.mean
    else:
        encoded = latent_dist.sample()
    latents = encoded * float(pipe.vae.config.scaling_factor)
    return latents.to(device=DEVICE, dtype=torch.float32)


def decode_latents_to_unit_interval(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Decode SD1.5 latents back into `[0, 1]` images."""

    latents_dec = latents.to(dtype=pipe.vae.dtype) / float(pipe.vae.config.scaling_factor)
    image = pipe.vae.decode(latents_dec, return_dict=False)[0]
    return ((image / 2.0) + 0.5).clamp(0.0, 1.0)


def denoise_one_step(
    pipe,
    latents: torch.Tensor,
    prompt_embeddings,
    timestep,
    *,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    step_index_hint: Optional[int] = None,
) -> torch.Tensor:
    """Advance the SD1.5 scheduler by one denoising step."""

    del step_index_hint
    guidance_scale = validate_guidance_scale(float(guidance_scale), field_name="generation.guidance_scale")
    do_cfg = bool(prompt_embeddings.get("do_classifier_free_guidance", False)) and guidance_scale > 1.0
    unet_device = next(pipe.unet.parameters()).device
    latents = latents.to(device=unet_device, dtype=torch.float32)

    if torch.is_tensor(timestep):
        timestep_model = timestep.to(device=unet_device)
    else:
        timestep_model = torch.tensor(timestep, device=unet_device)

    latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
    if hasattr(pipe.scheduler, "scale_model_input"):
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep_model)

    encoder_hidden_states = _prompt_embeds_for_unet(prompt_embeddings).to(device=unet_device)
    # The UNet may run in fp16/bfloat16, but the optimized latent state is kept
    # in fp32 for stable Adam updates.
    noise_pred = pipe.unet(
        latent_model_input.to(device=unet_device, dtype=pipe.unet.dtype),
        timestep_model,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0].float()

    if do_cfg:
        # Classifier-free guidance combines unconditional and conditional noise
        # predictions using the configured guidance scale.
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return _scheduler_step(
        pipe,
        noise_pred.to(dtype=torch.float32),
        timestep_model,
        latents.to(device=unet_device, dtype=torch.float32),
        eta=float(eta),
    )


def unrolled_latents_from_init(
    pipe,
    latents_init: torch.Tensor,
    prompt_embeddings,
    generation: GenerationConfig,
    *,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Run the SD1.5 denoising process from an initial latent tensor."""

    latents = latents_init.to(device=DEVICE, dtype=torch.float32)
    pipe.scheduler.set_timesteps(int(generation.num_steps), device=latents.device)
    timesteps = pipe.scheduler.timesteps.to(device=latents.device)

    for timestep in timesteps:
        if use_checkpoint and latents.requires_grad:
            from torch.utils.checkpoint import checkpoint as torch_checkpoint

            # Checkpointing trades extra UNet forward passes for lower memory
            # during diffusion-backprop sweeps.
            def step_fn(latents_in: torch.Tensor, timestep_in: torch.Tensor) -> torch.Tensor:
                return denoise_one_step(
                    pipe,
                    latents_in,
                    prompt_embeddings,
                    timestep_in,
                    guidance_scale=float(generation.guidance_scale),
                    eta=float(getattr(generation, "eta", 0.0)),
                )

            # Pass the timestep explicitly.
            latents = torch_checkpoint(
                step_fn,
                latents,
                timestep,
                use_reentrant=False,
            )
        else:
            latents = denoise_one_step(
                pipe,
                latents,
                prompt_embeddings,
                timestep,
                guidance_scale=float(generation.guidance_scale),
                eta=float(getattr(generation, "eta", 0.0)),
            )
    return latents


def unrolled_image_from_latents(
    pipe,
    latents_init: torch.Tensor,
    prompt_embeddings,
    generation: GenerationConfig,
    *,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Decode the fully denoised image from an initial latent tensor."""

    latents = unrolled_latents_from_init(
        pipe,
        latents_init,
        prompt_embeddings,
        generation,
        use_checkpoint=bool(use_checkpoint),
    )
    return decode_latents_to_unit_interval(pipe, latents)


@torch.no_grad()
def generate_from_latents(
    pipe,
    latents_init: torch.Tensor,
    prompt_embeddings,
    generation: GenerationConfig,
) -> torch.Tensor:
    """Run SD1.5 generation from an initial latent tensor and return one image."""

    decoded = unrolled_image_from_latents(
        pipe,
        latents_init,
        prompt_embeddings,
        generation,
        use_checkpoint=False,
    )
    return decoded[0]
