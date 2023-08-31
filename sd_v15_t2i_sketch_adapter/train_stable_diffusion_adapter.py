import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import typer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionAdapterPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.models import MultiAdapter, T2IAdapter, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from packaging import version
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

if is_wandb_available():
    import wandb

from dataloader import create_webdataset_reader

logger = get_logger(__name__)

app = typer.Typer(pretty_exceptions_show_locals=False)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(
    pretrained_model_name_or_path,
    vae,
    text_encoder,
    tokenizer,
    unet,
    adapter,
    accelerator,
    weight_dtype,
    step,
    seed=None,
    validation_prompt=None,
    validation_image=None,
    num_validation_images=4,
    revision=None,
):
    logger.info("Running validation... ")

    adapter = accelerator.unwrap_model(adapter)

    pipeline = StableDiffusionAdapterPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        adapter=adapter,
        safety_checker=None,
        revision=revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    if len(validation_image) == len(validation_prompt):
        validation_images = validation_image
        validation_prompts = validation_prompt
    elif len(validation_image) == 1:
        validation_images = validation_image * len(validation_prompt)
        validation_prompts = validation_prompt
    elif len(validation_prompt) == 1:
        validation_images = validation_image
        validation_prompts = validation_prompt * len(validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        # load as grayscale for sketch
        validation_image = Image.open(validation_image).convert("L")

        images = []

        for _ in range(num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt,
                    validation_image,
                    num_inference_steps=20,
                    generator=generator,
                    height=512,
                    width=512,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "validation_image": validation_image,
                "images": images,
                "validation_prompt": validation_prompt,
            }
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Adapter conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    # for log in image_logs:
    #     images = log["images"]
    #     validation_prompt = log["validation_prompt"]
    #     validation_image = log["validation_image"]

    #     image_grid_image = make_image_grid(images, 1, len(images))
    #     image_grid_image.save(os.path.join(accelerator.logging_dir, f"validation_{validation_prompt}_{step}.png"))

    return image_logs


@app.command()
def main(
    input_file_list_path: str = typer.Option(..., help="The input file path."),
    output_dir: str = typer.Option(".", help="The output directory."),
    logging_dir: str = typer.Option(".", help="The logging directory."),
    gradient_accumulation_steps: Optional[int] = typer.Option(1, help="Gradient accumulation steps."),
    mixed_precision: str = typer.Option("no", help="Mixed precision training."),
    report_to: str = typer.Option("none", help="The destination of the reports."),
    seed: int = typer.Option(42, help="Random seed."),
    tokenizer_name_or_path: str = typer.Option(None, help="The tokenizer name or path."),
    pretrained_model_name_or_path: str = typer.Option(None, help="The pretrained model name or path."),
    revision: str = typer.Option(None, help="Revision of pretrained model identifier from huggingface.co/models"),
    adapter_name_or_path: str = typer.Option(None, help="The adapter name or path."),
    adapter_config: str = typer.Option(None, help="The adapter config."),
    gradient_checkpointing: bool = typer.Option(False, help="Gradient checkpointing."),
    allow_tf32: bool = typer.Option(False, help="Enable TF32 for faster training on Ampere GPUs,"),
    scale_lr: bool = typer.Option(False, help="Scale the learning rate"),
    learning_rate: float = typer.Option(1e-5, help="Initial learning rate (after the potential warmup period) to use."),
    use_8bit_adam: bool = typer.Option(False, help="Use 8-bit adam"),
    adam_beta1: float = typer.Option(0.9, help="The beta1 parameter for the Adam optimizer."),
    adam_beta2: float = typer.Option(0.999, help="The beta2 parameter for the Adam optimizer."),
    adam_weight_decay: float = typer.Option(1e-02, help="Weight decay to use."),
    adam_epsilon: float = typer.Option(1e-08, help="Epsilon for the Adam optimizer."),
    max_grad_norm: float = typer.Option(1.0, help="Max gradient norm."),
    max_train_steps: int = typer.Option(500, help="Max number of training steps."),
    lr_scheduler: str = typer.Option("linear", help="The learning rate scheduler."),
    lr_warmup_steps: int = typer.Option(0, help="The number of warmup steps for the learning rate scheduler."),
    lr_num_cycles: int = typer.Option(1, help="The number of cycles for the learning rate scheduler."),
    lr_power: float = typer.Option(1.0, help="The power for the learning rate scheduler."),
    tracker_project_name: str = typer.Option("diffusion", help="The tracker project name."),
    train_batch_size: int = typer.Option(1, help="The batch size for training."),
    resume_from_checkpoint: str = typer.Option(None, help="The path to a checkpoint to resume from."),
    set_grads_to_none: bool = typer.Option(False, help="setting grads to None instead of zero"),
    checkpoints_total_limit: int = typer.Option(None, help="The maximum number of checkpoints to keep."),
    checkpointing_steps: int = typer.Option(500, help="The number of steps between checkpoints."),
    validation_steps: int = typer.Option(500, help="The number of steps between validation."),
    validation_prompt: Optional[List[str]] = typer.Option(None, help="The prompt to use for validation."),
    validation_image: Optional[List[str]] = typer.Option(None, help="The image to use for validation."),
):
    # load list of file paths from input_file_list_path
    with open(input_file_list_path, "r") as f:
        input_file_list = f.readlines()
    input_file_list = [x.strip() for x in input_file_list]
    input_file_list = [x for x in input_file_list if x != ""]
    input_file_list = [x for x in input_file_list if not x.startswith("#")]

    input_file_list = [os.path.join("/mnt/data1/ayushman/laion2B-en-aesthetic-data", x) for x in input_file_list]
    assert [os.path.exists(x) for x in input_file_list]

    logging_dir = Path(output_dir, logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(seed)

    logger.info(validation_prompt)

    if accelerator.is_local_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    if tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, revision=revision, use_fast=False)
    elif pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    if adapter_name_or_path:
        logger.info("Loading existing adapter weights")
        adapter = T2IAdapter.from_pretrained(adapter_name_or_path)
    else:
        assert adapter_config is not None, "adapter_config must be provided if adapter_name_or_path is not provided."
        logger.info("Creating new adapter")
        adapter = T2IAdapter.from_config(adapter_config)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "adapter"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = T2IAdapter.from_pretrained(input_dir, subfolder="adapter")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    adapter.train()

    if gradient_checkpointing:
        adapter.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(adapter).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(adapter).dtype}. {low_precision_error_string}"
        )

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = adapter.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataloader = create_webdataset_reader(
        tokenizer,
        input_file_list,
        batch_size=train_batch_size,
        num_prepro_workers=8,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
    )

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    # Prepare everything with our `accelerator`.
    adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        adapter, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(tracker_project_name)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Mixed precision training = {accelerator.mixed_precision}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step, first_epoch = 0, 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist")
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    data_iter = iter(train_dataloader)

    for step in range(initial_global_step, max_train_steps):
        batch = next(data_iter)

        if step < 5:
            pixel_values = batch["pixel_values"]
            save_image(pixel_values, f"input_examples/{step}_img.png", nrow=4, normalize=True)
            conditioning_pixel_values = batch["conditioning_pixel_values"]
            save_image(conditioning_pixel_values, f"input_examples/{step}_cond.png", nrow=4, normalize=True)

        with accelerator.accumulate(adapter):
            batch["pixel_values"] = batch["pixel_values"].to(accelerator.device)
            batch["conditioning_pixel_values"] = batch["conditioning_pixel_values"].to(accelerator.device)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device)

            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            adapter_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
            adapter_state = adapter(adapter_image)

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[state.to(dtype=weight_dtype) for state in adapter_state],
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = adapter.parameters()
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=set_grads_to_none)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if checkpoints_total_limit is not None:
                        checkpoints = os.listdir(output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if validation_prompt is not None and global_step % validation_steps == 0:
                    image_logs = log_validation(
                        pretrained_model_name_or_path,
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        adapter,
                        accelerator,
                        weight_dtype,
                        global_step,
                        seed,
                        validation_prompt,
                        validation_image,
                        num_validation_images=4,
                        revision=revision,
                    )

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter = accelerator.unwrap_model(adapter)
        adapter.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    app()
