from datasets import load_from_disk
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from rich.console import Console
import json
import os
from pathlib import Path
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from models import Capsule_CoAdapter
import accelerate
from rich.console import Console
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder, notebook_login
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
# from t2i import Adapter
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    #T2IAdapter,
)
# from diffusers.models.adapter import LightAdapter
from adapter import T2IAdapter, CoAdapterFuser
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from pipeline_xl_adapter import StableDiffusionXLAdapterPipeline
from torch.utils.data import DataLoader, Dataset
from datasets.fingerprint import Hasher



import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from rich.console import Console
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
import logging
from typing import Any, Optional, Union, Dict, List, Tuple
import random
import functools
from transformers import AutoTokenizer
from transformers import AutoTokenizer, DPTFeatureExtractor, DPTForDepthEstimation, PretrainedConfig






# ==========================================================================
#                             setup logger                                  
# ==========================================================================

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='dataset.log', filemode='a', format=log_format, level=logging.DEBUG)
logger = logging.getLogger(__name__)








class prepare_dataset:
    def __init__(self,
                 dataset_name,
                 dataset_dir,
                 image_column,
                 caption_column,
                 conditioning_image_column,
                 conditioning_image_column2,
                 device = "cuda:3",
                 resolution = 1024,
                 shuffle = True,
                 max_train_samples = 10000):
        
        
        self.resolution = resolution
        self.caption_column = caption_column
        self.image_column = image_column
        self.conditioning_image_column = conditioning_image_column
        self.conditioning_image_column2 = conditioning_image_column2
        self.max_train_samples = max_train_samples
        
        if dataset_name is not None and dataset_dir is not None:
            raise ValueError("You can't specify both a dataset name and a dataset directory.")
        
        if dataset_name:
            dataset = load_dataset(dataset_name)
        else:
            if dataset_dir:
                dataset = load_from_disk(dataset_dir)
        Console().log(f"dataset has been loaded", style='red')
    
        column_names = dataset.column_names
        Console().print(f"column names are {column_names}", style='red')
        
        # split the dataset
        dataset = dataset.train_test_split(test_size=0.01)
        Console().print(f"dataset has been split", style='red')
        
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        
        if shuffle:
            train_dataset = train_dataset.shuffle(seed=100)
            Console().print(f"dataset has been shuffled", style='red')
        if max_train_samples is not None:
            train_dataset = train_dataset.select(range(max_train_samples))
            Console().print(f"max train samples --> {max_train_samples}", style='red')

        
        with Console().status(f"[cyan]Loading [red]Tokenizers [cyan]and [red]Text-Encoders", spinner='bouncingBall') as status:
            self.init_models()
        
        if not hasattr(self, "tokenizers") or not hasattr(self, "text_encoders"):
            raise ValueError("tokenizers and text encoders not loaded successfully")
                
        self.process_train_ds(train_dataset, device)
    
    
    def get_dataset(self):
        return self.processed_dataset
    
    
    def import_model_class_from_model_name_or_path(self, 
                                    pretrained_model_name_or_path: str, 
                                    revision: str, 
                                    subfolder: str = "text_encoder"
                                    ):
        
        '''
        helper function to import the correct text encoder class
        '''        
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision, cache='./cache'
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")
    
    
    
    def init_models(self):
        '''
        load the tokenizers and text encoders for sdxl, there are two of each
        attributes tokenizers and text_encoders are set
        '''        
        # set the tokenizers and text encoders
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
            cache='./cache'
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
            cache='./cache'
        )
        
        
        
        # import correct text encoder classes
        text_encoder_cls_one = self.import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, None
        )
        text_encoder_cls_two = self.import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, None, subfolder="text_encoder_2"
        )
        
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=None,
            
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=None,
            
        )

        self.tokenizers = [tokenizer_one, tokenizer_two]
        Console().print(f"tokenizers have been loaded", style='red')
        
        self.text_encoders = [text_encoder_one, text_encoder_two]
        Console().print(f"text encoders have been loaded", style='red')

    
    def process_train_ds(self, train_ds, proportion_empty_prompts=0.0, device = "cuda:3"):
        '''
        take train dataset and compute embeddings for unet and both text encoders to save the VRAM
        apply the transformation and preprocessing
        '''
        compute_embeddings_fn = functools.partial(
                                                self.compute_embeddings,
                                                text_encoders=self.text_encoders,
                                                tokenizers=self.tokenizers,
                                                proportion_empty_prompts=proportion_empty_prompts,
                                                device = device
                                                )
        
        with Console().status(f"[cyan]Pre-computing the [red]embeddings", spinner='bouncingBall') as status:
            
            new_fingerprint = Hasher.hash(self.max_train_samples)
            train_dataset = train_ds.map(
                                            compute_embeddings_fn,
                                            batched=True,
                                            batch_size=20,
                                            new_fingerprint=new_fingerprint,
                                        )
            status.update(status=f"[red]deleting ✂️\tthe tokenizers and text encoders", spinner='earth')
            del self.text_encoders, self.tokenizers
            gc.collect()
            torch.cuda.empty_cache()

            Console().log(f"embeddings created", style='red')
            status.update(status=f"Transforming train dataset", spinner='aesthetic')
            # 🔴 Call the preparation function
            
            
            # ⬇️ Dump the dataset
            train_dataset.save_to_disk("./train_dataset_inter")
            
            
            self.processed_dataset = self.prepare_train_dataset(train_dataset)

    def compute_embeddings(
                            self,
                            batch, 
                            proportion_empty_prompts, 
                            text_encoders, 
                            tokenizers, 
                            is_train=True,
                            device = "cuda:3"
                        ):
        original_size = (self.resolution, self.resolution)
        target_size = (self.resolution, self.resolution)
        
        crops_coords_top_left = (0, 0)
        prompt_batch = batch[self.caption_column]

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                                                                    prompt_batch, 
                                                                    text_encoders, 
                                                                    tokenizers, 
                                                                    proportion_empty_prompts, 
                                                                    is_train,
                                                                    device
                                                                )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        unet_added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}
    
    
    
    
    def prepare_train_dataset(self, dataset):
        '''
        apply the transformations on image and conditioning image

        Parameters
        ----------
        dataset : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''        
        p = 0.05
        image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # ye wali transform donoon images pr lgey gi
        # no single channel image
        conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(
                self.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(self.resolution),
            # transforms.Grayscale(), # 👈 donot need the grayscale image
            transforms.ToTensor(),
        ]
        )
        
        def preprocess_train(examples):
            '''
            Apply transformations on a single record

            Parameters
            ----------
            examples : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            '''            
            #augment = transforms.TrivialAugmentWide()
            # read the image and both conditioning images
            images = [image.convert("RGB") for image in examples[self.image_column]]
            conditioning_images = [
                image.convert("RGB") for image in examples[self.conditioning_image_column]
            ]
            conditioning_images2 = [
                image.convert("RGB") for image in examples[self.conditioning_image_column2]
            ]
            
            # 🔴change the transformations. All adapter images are 3 channels
            images = [image_transforms(image) for image in images]
            conditioning_images = [
                conditioning_image_transforms(image) for image in conditioning_images
            ]
            conditioning_images2 = [
                conditioning_image_transforms(image) for image in conditioning_images2
            ]
            
            #combined = torch.stack(images + conditioning_images)
            #combined = augment(combined.to(torch.uint8))
            #images = [combined[i] for i in range(len(images))]
            #conditioning_images = [combined[len(images)+i] for i in range(len(conditioning_images))]
            examples["pixel_values"] = images
            examples["conditioning_pixel_values"] = conditioning_images
            examples["conditioning_pixel_values2"] = conditioning_images2
            
            return examples

        # 🔵 See the datasets lib. docs for this purpose
        dataset = dataset.with_transform(preprocess_train)

        return dataset

    
    def encode_prompt(
        self,
        prompt_batch, 
        text_encoders, 
        tokenizers, 
        proportion_empty_prompts, 
        is_train=True,
        device = "cuda:3"
    ):
        '''
        get intermiediate embeddings form both the text encoders for passing to unet, this is only in case of sdxl
        

        Parameters
        ----------
        prompt_batch : _type_
            _description_
        text_encoders : List[text encoders]
            CLIPTextModel, CLIPTextModelWithProjection
        tokenizers : List[CLIPTokenizer]
            _description_
        proportion_empty_prompts : _type_
            _description_
        is_train : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        '''        
        prompt_embeds_list = []

        captions = []
        for caption in prompt_batch:
            if random.random() < float(proportion_empty_prompts):
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        with torch.no_grad():
            # put text encoders on gpu
            Console().log(f"[cyan]using [green]cuda:0 [cyan]for text encoders to get intermediate embeddings", style='red')
            for k in range(len(text_encoders)):
                text_encoders[k] = text_encoders[k].to(device)
            
            
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        
        return prompt_embeds, pooled_prompt_embeds

        
    def train_collate_fn(self, examples):
        
        # target images ka batch
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # conditioning images sketch ka batch
        conditioning_pixel_values = torch.stack(
            [example["conditioning_pixel_values"] for example in examples]
        )
        
        conditioning_pixel_values2 = torch.stack(
            [example["conditioning_pixel_values2"] for example in examples]
        )
        
        # push the conditioning images to gpu
        conditioning_pixel_values = conditioning_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        conditioning_pixel_values2 = conditioning_pixel_values2.to(
            memory_format=torch.contiguous_format
        ).float()
        
        
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        prompt_ids = torch.stack(
            [torch.tensor(example["prompt_embeds"]) for example in examples]
        )

        add_text_embeds = torch.stack(
            [torch.tensor(example["text_embeds"]) for example in examples]
        )
        add_time_ids = torch.stack(
            [torch.tensor(example["time_ids"]) for example in examples]
        )

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "conditioning_pixel_values2": conditioning_pixel_values2,
            "prompt_ids": prompt_ids,
            "unet_added_conditions": {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            },
        }


    def get_dataloader(self, batchsize=2, workers=4, shuffle=True):
        '''
        Get the dataloader for training
        '''
        train_dataset = self.get_dataset()
        
        train_dataloader = DataLoader(
                                        train_dataset,
                                        collate_fn=self.train_collate_fn,
                                        batch_size=batchsize,
                                        num_workers=workers,
                                        shuffle=shuffle,
                                    )
        
        return train_dataloader




# ==========================================================================
#                             lightning module                                  
# ==========================================================================
class sdxl_pl_model(pl.LightningModule):
    def __init__(self, dataset_name, n_adapters, adapter_names, learning_rate,adam_weight_decay, 
                 dataset_dir="dress_pose_depth_scribble_captions_v1.0_3_million",
                 shuffle=True,
                 image_column="image_path",
                 caption_column="caption",
                 conditioning_image_column = "scribble_path",
                 conditioning_image_column2 = "pose_path",
                 max_train_samples = 10000,
                 dataset_cache_device = "cuda:3",
                 workers = 12,
                 batchsize = 2,
                 pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
                 pretrained_vae_model_name_or_path = None
                 ) -> None:
        super().__init__()
        self.n_adapters = n_adapters
        self.adapter_names = adapter_names
        self.learning_rate = learning_rate
        self.adam_weight_decay = adam_weight_decay
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained_vae_model_name_or_path = pretrained_vae_model_name_or_path
        
        self.dataset_name = dataset_name,
        self.dataset_dir = dataset_dir, 
        self.shuffle = shuffle,
        self.image_column = image_column,
        self.caption_column = caption_column,
        self.conditioning_image_column = conditioning_image_column,
        self.conditioning_image_column2 = conditioning_image_column2,
        self.max_train_samples = max_train_samples
        self.dataset_cache_device = dataset_cache_device
        self.workers = workers
        self.batchsize = batchsize
        
        # save hyper parameters
        self.save_hyperparameters()        
        
        # with Console().status(f"loading components ...", spinner='bouncingBall') as status:
        self.tokenizer_one, self.tokenizer_two, self.text_encoder_one, self.text_encoder_two, self.vae, self.unet, self.noise_scheduler = self._initialize_components()

        # put models to correct devices
        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder_one.to(self.device)
        self.text_encoder_two.to(self.device)
        # status.update(status=f"🔵\tModels are put on correct device", spinner='aesthetic')
        Console().log(f"device is {self.device}")
        
        
        
        Console().log(f"sdxl pipeline components have been initialized...")
        # status.update(status=f"loading adapters - coadapters ...", spinner='aesthetic')
        self.model = self.setup_coadapter_adapters_model(n_adapters = self.hparams.n_adapters, adapter_names = self.hparams.adapter_names)
        Console().log(f"adapter-coadapters have been setup")
        # status.update(status=f"⚡ Changing grad status of model components....", spinner='earth')
        self.vae.requires_grad_(False)
        Console().log(f"changed vae grad to False")
        self.unet.requires_grad_(False)
        Console().log(f"changed unet grad to False")
        self.text_encoder_one.requires_grad_(False)
        Console().log(f"changed text_encoder_one grad to False")
        self.text_encoder_two.requires_grad_(False)
        Console().log(f"changed text_encoder_two grad to False")
        
        self.model.requires_grad_(True)
        Console().log(f"model-adapter-coadapter grad set to True")
        self.model.train()
        Console().log(f"model-adapter-coadapter set to training mode")
        
            
            
    def forward(self, *inputs_):
        
        return self.model(*inputs_)

    def training_step(self, batch, batch_idx):
        # self.log("device", self.device)
        
        # Console().log(f"🔴\tVAE is on device ===> {self.vae.device}")
        
        pixel_values = batch["pixel_values"].to(self.device)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
                                    0,
                                    self.noise_scheduler.config.num_train_timesteps,
                                    (bsz,),
                                    device=latents.device,
                                )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # t2i_adapter conditioning.
        t2i_adapter_image = batch["conditioning_pixel_values"] # sketch
        t2i_adapter_image2 = batch["conditioning_pixel_values2"]  # pose
        
        # # log the images to wandb
        # to_pil = transforms.ToPILImage()
        # t2i_adapter_image_wandb = [to_pil(k) for k in t2i_adapter_image]
        # t2i_adapter_image2_wandb = [to_pil(k) for k in t2i_adapter_image2]
        

        # down_block_res_samples = scribble_adapter(t2i_adapter_image)
        down_block_res_samples = self.model(t2i_adapter_image, t2i_adapter_image2)
        
        # # reverse the list
        # up_block_res_samples = down_block_res_samples[::-1]
        
        
        # for k in down_block_res_samples:
        #     Console().print(k.shape)
        
        # for k, v in enumerate(down_block_res_samples):
        #    down_block_res_samples[k] = torch.cat([v] * 2, dim=0)
        #print(unet.down_blocks)
        #print([x.shape for x in down_block_res_samples])

        # 🟩 Predict the noise residual
        # if batch_idx == 0:
        #     Console().print(f" batch prompt_ids shape [cyan] {batch['prompt_ids'].shape}")
        #     Console().print(f" batch unet_added_conditions shape [cyan] {[(k,v.shape)  for k,v in batch['unet_added_conditions'].items()]}")
        #     Console().print(f" down block res samples shape [cyan] {[k.shape for k in down_block_res_samples]}")
        
        # Console().log(f"💀\tUNET model is on device {self.unet.device}")
        
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=batch["prompt_ids"],
            added_cond_kwargs=batch["unet_added_conditions"],
            down_block_additional_residuals=down_block_res_samples,
        ).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        
        # 🔴 Orignal loss that was used in controlnet
        
        loss = F.mse_loss(
                            model_pred.float(), 
                            target.float(), 
                            reduction="mean"
                        )
        # log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        
        return {'loss' : loss}
    
    
    def configure_optimizers(self):
        params_to_optimize = self.model.parameters()
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.adam_weight_decay,
        )
        Console().print(f"optimizer defined")
        
        # Define learning rate scheduler (for example, a StepLR)
        # self.trainer.max_steps
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6),
            'interval': 'epoch',  # or 'step' for step-wise learning rate updates
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
        
    
    def train_dataloader(self):
        dataset_ = prepare_dataset(
                            dataset_name = self.hparams.dataset_name,
                            dataset_dir = self.hparams.dataset_dir, 
                            shuffle = self.hparams.shuffle,
                            image_column = self.hparams.image_column,
                            caption_column = self.hparams.caption_column,
                            conditioning_image_column = self.hparams.conditioning_image_column,
                            conditioning_image_column2 = self.hparams.conditioning_image_column2,
                            max_train_samples = self.hparams.max_train_samples,
                            device = self.hparams.dataset_cache_device
                        )

        dl = dataset_.get_dataloader(batchsize=self.hparams.batchsize, workers=self.hparams.workers, shuffle=self.hparams.shuffle)
        return dl

        
        
        
        
    def setup_coadapter_adapters_model(self, n_adapters : int = 2, adapter_names : List[str] = ['sketch', 'openpose']):
        
        
        all_adapters = []
        for k in range(n_adapters):
            adapter =  T2IAdapter(channels=[320, 640, 1280], in_channels=3) 
            adapter.requires_grad_(True)
            
            all_adapters.append(adapter)
        
        # adapter_sketch = T2IAdapter(channels=[320, 640, 1280], in_channels=3)
        # adapter_sketch.requires_grad_(True)
        
        
        # adapter_pose = T2IAdapter(channels=[320,640,1280], in_channels=3)
        # adapter_pose.requires_grad_(True)
        
        # 💀  define coadapter
        coadapter_fuser = CoAdapterFuser()
        coadapter_fuser.requires_grad_(True)
        
        #  💀 define the forward pass fuser (its purpose is only to encapsulate the forward pass)
        fuser = Capsule_CoAdapter(*all_adapters, coadapter_fuser=coadapter_fuser, adapter_order=['sketch','openpose'])
        fuser.requires_grad_(True)
        
        
        
        return fuser
        
        
        
        
        
    
        
    def import_model_class_from_model_name_or_path(self,
                                                pretrained_model_name_or_path: str, 
                                                revision: str, 
                                                subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision, cache='./cache'
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")
    
    
    
    def _initialize_components(self):
        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
            cache='./cache'
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
            cache='./cache'
        )

        # import correct text encoder classes
        text_encoder_cls_one = self.import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, None
        )
        text_encoder_cls_two = self.import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, None, subfolder="text_encoder_2"
        )
        
        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=None,
            
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=None,
            
        )
        vae_path = (
            self.pretrained_model_name_or_path
            if self.pretrained_vae_model_name_or_path is None
            else self.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if self.pretrained_vae_model_name_or_path is None else None,
            revision=None,
            cache='./cache'
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet", revision=None, cache='./cache'
        )
        
        
        # return all 
        return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler



if __name__ == '__main__':
    
    # dataset_ = prepare_dataset(
    #                         dataset_name = None,
    #                         dataset_dir = "dress_pose_depth_scribble_captions_v1.0_3_million", 
    #                         shuffle = True,
    #                         image_column = "image_path",
    #                         caption_column = "caption",
    #                         conditioning_image_column = "scribble_path",
    #                         conditioning_image_column2 = "pose_path",
    #                         max_train_samples = 100
    #                     )

    # dl = dataset_.get_dataloader(batchsize=16, workers=8, shuffle=True)
    
    
    
    # for k in dl:
    #     Console().print(f"➡️ BATCH: {k.keys()}", style='red')
    #     break
    
    
    model = sdxl_pl_model(dataset_name = None, n_adapters = 2, adapter_names = ['sketch', 'openpose'], learning_rate = 1e-5, adam_weight_decay = 5e-5,
                          max_train_samples=10000, batchsize=4)
    
    wandb_logger = WandbLogger(name="sdxl_adapter_coadapter-pl-lightning", save_dir="./pl_lightning_ckpts", log_model="all")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=1500, dirpath="sdxl_ckpts_pl", filename='{epoch}-{step}', verbose=True, save_last = True, save_weights_only=False, save_on_train_epoch_end =True)
    trainer = pl.Trainer(
                            logger=wandb_logger,
                            callbacks=[checkpoint_callback],
                            accelerator="auto",
                            # precision=16,
                            devices=[3],
                            # strategy='ddp',
                            max_epochs=5,
                            accumulate_grad_batches=4,
                        )
    
    trainer.fit(model)
    











# class HF_Dataset_Module(pl.LightningModule):
#     def __init__(self, 
#                  dmax_train_samplesataset_name : str,
#                  train_data_dir : str,
#                  caption_column : str,
#                 #  text_encoders : List,
#                 #  tokenizers : List,
#                  conditioning_image_column : Union[str, List[str]],
#                  image_column : str,
#                  batch_size : int = 2,
#                  dataloader_num_workers : int =4,
#                  max_train_samples : Optional[int] = None,
#                  shuffle : bool = True,
#                  resolution : int = 1024,
#                  proportion_empty_prompts : Optional[float] = 0.0,
#                  split_ratio : float = 0.1):
#         super().__init__()
#         self.dataset_name = dataset_name
#         self.train_data_dir = train_data_dir
#         self.split_ratio = split_ratio
#         self.shuffle = shuffle
#         self.max_train_samples = max_train_samples
#         self.resolution = resolution
#         # get the columns for the dataset
#         self.caption_column = caption_column
#         self.conditioning_image_column = conditioning_image_column
#         self.image_column = image_column
        
        
#         self.batch_size = batch_size
#         self.dataloader_num_workers = dataloader_num_workers
#         self.proportion_empty_prompts = proportion_empty_prompts
#         # self.text_encoders = text_encoders
#         # self.tokenizers = tokenizers
    
#         logger.info(f"device: ===> {self.device}")
    
#         # save the hyperparameters
#         self.save_hyperparameters()
#         logger.info(f"params dataset are {self.hparams}")
    
    
#         # ==========================================================================
#         #                             initialize models                                  
#         # ==========================================================================

#         pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"


#         # Load the tokenizers
#         tokenizer_one = AutoTokenizer.from_pretrained(
#             pretrained_model_name_or_path,
#             subfolder="tokenizer",
#             revision=None,
#             use_fast=False,
#             cache='./cache'
#         )
#         tokenizer_two = AutoTokenizer.from_pretrained(
#             pretrained_model_name_or_path,
#             subfolder="tokenizer_2",
#             revision=None,
#             use_fast=False,
#             cache='./cache'
#         )

#         # import correct text encoder classes
#         text_encoder_cls_one = import_model_class_from_model_name_or_path(
#             pretrained_model_name_or_path, None
#         )
#         text_encoder_cls_two = import_model_class_from_model_name_or_path(
#             pretrained_model_name_or_path, None, subfolder="text_encoder_2"
#         )
#         #%%
#         # Load scheduler and models
#         # noise_scheduler = DDPMScheduler.from_pretrained(
#         #     self.pretrained_model_name_or_path, subfolder="scheduler"
#         # )
#         text_encoder_one = text_encoder_cls_one.from_pretrained(
#             pretrained_model_name_or_path,
#             subfolder="text_encoder",
#             revision=None,
            
#         ).to(self.device)
#         text_encoder_two = text_encoder_cls_two.from_pretrained(
#             pretrained_model_name_or_path,
#             subfolder="text_encoder_2",
#             revision=None,
            
#         ).to(self.device)
#         # vae_path = (
#         #     self.pretrained_model_name_or_path
#         #     if args.pretrained_vae_model_name_or_path is None
#         #     else args.pretrained_vae_model_name_or_path
#         # )
#         # vae = AutoencoderKL.from_pretrained(
#         #     vae_path,
#         #     subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
#         #     revision=None,
#         #     cache='./cache'
#         # )
#         # unet = UNet2DConditionModel.from_pretrained(
#         #     self.pretrained_model_name_or_path, subfolder="unet", revision=None, cache='./cache'
#         # )

#         self.text_encoders = [text_encoder_one, text_encoder_two]
#         self.tokenizers = [tokenizer_one, tokenizer_two]

#         self.model = torch.nn.Module() 
        
        
        
    
#     def forward(self, x):
#         return x

#     def training_step(self, batch, batch_idx):
#         Console().print(batch.keys(), style='red')
#         # Normally you'd do some operations here
#         loss = torch.tensor(0)  # Placeholder loss
#         return {'loss': loss}

#     def configure_optimizers(self):
#         return torch.optim.SGD(self.parameters(), lr=1e-5)
    
    
    
    
    
    
    
    
#     # ==========================================================================
#     #                             data related hooks                                  
#     # ==========================================================================
    
    
    
    
    
    
#     def prepare_data(self) -> None:
#         # download the dataset , runs on 1 GPU
#         # assign train/val datasets for use in dataloaders
#         logger.info(f'inside prepare_data single gpu')
#         train_dataset, test_dataset, column_names = self.get_train_dataset(self.split_ratio)
        
#         # save the test dataset locally for later use
#         # TODO: need to apply transforms
#         test_dataset.save_to_disk("test_dataset")
#         logger.info("Test dataset saved to disk for later use") # inside setup()
        
#         logger.info("Dataset setup complete")
#         # log dataset info
#         logger.info("Train Dataset info: {}".format(train_dataset.info))
#         logger.info("Test Dataset info: {}".format(test_dataset.info))
        
#         # if shuffle is True, shuffle the dataset
#         if self.shuffle:
#             train_dataset = train_dataset.shuffle()
        
#         if self.max_train_samples is not None:
#             logger.info("Limiting the number of training samples to {}".format(self.max_train_samples))
#             train_dataset = train_dataset.select(range(self.max_train_samples))
        
        
#         # save the train dataset
#         train_dataset.save_to_disk("train_dataset")
#         logger.info(f"saved train dataset raw")
        
        
        
#         # # ==========================================================================
#         # #                        save vram by precomputing embeddings                                  
#         # # ==========================================================================
        
#         # compute_embeddings_fn = functools.partial(
#         #                                             self.compute_embeddings,
#         #                                             text_encoders=self.text_encoders,
#         #                                             tokenizers=self.tokenizers,
#         #                                             proportion_empty_prompts=self.proportion_empty_prompts,
#         #                                         )
        
#         # train_dataset = train_dataset.map(
#         #                                             compute_embeddings_fn,
#         #                                             batched=True,
#         #                                             batch_size=20,
#         #                                             # new_fingerprint=new_fingerprint,
#         #                                         )
        
#         # Console().print(f"embeddings have been computed", style='red')
        
#         # # apply transformations on dataset, both target transforms 🖼️ and conditional image transforms ⚙️
#         # train_dataset = self.apply_transforms(train_dataset)
#         # # TODO: apply the same transforms on the test dataset
        
        
#         # # save the train dataset locally for later use
#         # train_dataset.save_to_disk("train_dataset")
    
    
    
#     def setup(self, stage: Optional[str] = None) -> None:
#         """
#         assign the states here
#         """
#         logger.info(f"inside the setup, called on specified GPUs")
        
#         train_dataset = load_from_disk("train_dataset")
#         # ==========================================================================
#         #                        save vram by precomputing embeddings                                  
#         # ==========================================================================
        
#         compute_embeddings_fn = functools.partial(
#                                                     self.compute_embeddings,
#                                                     text_encoders=self.text_encoders,
#                                                     tokenizers=self.tokenizers,
#                                                     proportion_empty_prompts=self.proportion_empty_prompts,
#                                                 )
        
#         train_dataset = train_dataset.map(
#                                                     compute_embeddings_fn,
#                                                     batched=True,
#                                                     batch_size=20,
#                                                     # new_fingerprint=new_fingerprint,
#                                                 )
        
#         Console().print(f"embeddings have been computed", style='red')
        
#         # apply transformations on dataset, both target transforms 🖼️ and conditional image transforms ⚙️
#         train_dataset = self.apply_transforms(train_dataset)
#         # TODO: apply the same transforms on the test dataset
        
        
#         # save the train dataset locally for later use
#         # train_dataset.save_to_disk("train_dataset")
    
        
        
        
        
#         # self.train_dataset = load_from_disk("train_dataset")
#         self.train_dataset = train_dataset
#         self.test_dataset = load_from_disk("test_dataset")
    
    
#     def train_dataloader(self):
        
#         # define the collator to make batch
#         def train_collate_fn(examples):
        
#             # target images ka batch
#             pixel_values = torch.stack([example["pixel_values"] for example in examples])
#             # conditioning images sketch ka batch
#             conditioning_pixel_values = torch.stack(
#                 [example["conditioning_pixel_values"] for example in examples]
#             )
            
#             conditioning_pixel_values2 = torch.stack(
#                 [example["conditioning_pixel_values2"] for example in examples]
#             )
            
#             # push the conditioning images to gpu
#             conditioning_pixel_values = conditioning_pixel_values.to(
#                 memory_format=torch.contiguous_format
#             ).float()
#             conditioning_pixel_values2 = conditioning_pixel_values2.to(
#                 memory_format=torch.contiguous_format
#             ).float()
            
            
#             pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
#             prompt_ids = torch.stack(
#                 [torch.tensor(example["prompt_embeds"]) for example in examples]
#             )

#             add_text_embeds = torch.stack(
#                 [torch.tensor(example["text_embeds"]) for example in examples]
#             )
#             add_time_ids = torch.stack(
#                 [torch.tensor(example["time_ids"]) for example in examples]
#             )

#             return {
#                 "pixel_values": pixel_values,
#                 "conditioning_pixel_values": conditioning_pixel_values,
#                 "conditioning_pixel_values2": conditioning_pixel_values2,
#                 "prompt_ids": prompt_ids,
#                 "unet_added_conditions": {
#                     "text_embeds": add_text_embeds,
#                     "time_ids": add_time_ids,
#                 },
#             }
        
        
#         return DataLoader(self.train_dataset, 
#                           shuffle=False, 
#                           collate_fn=train_collate_fn, 
#                           batch_size=self.batch_size, 
#                           num_workers=self.dataloader_num_workers)


#     def test_dataloader(self):
        
#         # TODO: fix it according to dataset keys
        
#         # define collator to make batch
#         def test_collate_fn(examples):
#             pixel_values = torch.stack([example["pixel_values"] for example in examples])
#             conditioning_pixel_values = torch.stack(
#                 [example["conditioning_pixel_values"] for example in examples]
#             )
#             conditioning_pixel_values = conditioning_pixel_values.to(
#                 memory_format=torch.contiguous_format
#             ).float()
#             pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            
#             return {
#                 "pixel_values": pixel_values,
#                 "conditioning_pixel_values": conditioning_pixel_values,
#                 "caption": [example["caption"] for example in examples],
#             }

        
#         return DataLoader(self.test_dataset, 
#                           shuffle=False, 
#                           collate_fn=test_collate_fn, 
#                           batch_size=self.batch_size, 
#                           num_workers=self.dataloader_num_workers)
    
#     def encode_prompt(self,
#                     prompt_batch, 
#                     text_encoders, 
#                     tokenizers, 
#                     proportion_empty_prompts, 
#                     is_train=True):
#         prompt_embeds_list = []

#         captions = []
#         for caption in prompt_batch:
#             if random.random() < proportion_empty_prompts:
#                 captions.append("")
#             elif isinstance(caption, str):
#                 captions.append(caption)
#             elif isinstance(caption, (list, np.ndarray)):
#                 # take a random caption if there are multiple
#                 captions.append(random.choice(caption) if is_train else caption[0])

#         with torch.no_grad():
#             for tokenizer, text_encoder in zip(tokenizers, text_encoders):
#                 text_inputs = tokenizer(
#                     captions,
#                     padding="max_length",
#                     max_length=tokenizer.model_max_length,
#                     truncation=True,
#                     return_tensors="pt",
#                 )
#                 text_input_ids = text_inputs.input_ids
#                 Console().print(f"[green]device : [red]{self.device}")
#                 text_encoder.to(self.device)
#                 prompt_embeds = text_encoder(
#                     text_input_ids.to(text_encoder.device),
#                     output_hidden_states=True,
#                 )

#                 # We are only ALWAYS interested in the pooled output of the final text encoder
#                 pooled_prompt_embeds = prompt_embeds[0]
#                 prompt_embeds = prompt_embeds.hidden_states[-2]
#                 bs_embed, seq_len, _ = prompt_embeds.shape
#                 prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
#                 prompt_embeds_list.append(prompt_embeds)

#         prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
#         pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
#         return prompt_embeds, pooled_prompt_embeds

        
    
    
    
#     def compute_embeddings(self,
#         batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
#     ):
#         original_size = (self.resolution, self.resolution)
#         target_size = (self.resolution, self.resolution)
#         crops_coords_top_left = (0, 0)
#         prompt_batch = batch[self.caption_column]

#         prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
#             prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
#         )
#         add_text_embeds = pooled_prompt_embeds

#         # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
#         add_time_ids = list(original_size + crops_coords_top_left + target_size)
#         add_time_ids = torch.tensor([add_time_ids])
#         add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
#         unet_added_cond_kwargs = {
#             "text_embeds": add_text_embeds,
#             "time_ids": add_time_ids,
#         }

#         return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    
    
    
    
#     def get_train_dataset(self, split_ratio):
#         # 🔴 download the dataset from HuggingFace Datasets / Load Locally
#         # if both self.dataset_name and self.train_data_dir are given raise an error
#         if self.dataset_name is not None and self.train_data_dir is not None:
#             raise ValueError("Both dataset_name and train_data_dir cannot be given at the same time")
        
#         if self.dataset_name is not None:
#             logger.info("Loading dataset from HuggingFace Datasets")
#             dataset = load_dataset(
#                                         self.dataset_name,
#                                         # cache_dir=self.cache_dir,
#                                     )

#         else:
#             # load a locally stored dataset which is in HF-format
#             if self.train_data_dir is not None:
#                 logger.info("Loading dataset from local storage")
#                 dataset = load_from_disk(
#                                             self.train_data_dir
#                                         )
#         # Preprocessing the datasets.
#         # We need to tokenize inputs and targets.
#         column_names = dataset.column_names

#         # 6. Get the column names for input/target.
        
#         # making split of the dataset so as to avoid changing the train key elsewhere
#         # test split istimaal nhin karna
#         dataset = dataset.train_test_split(test_size=split_ratio)
        
#         train_dataset = dataset["train"]
#         test_dataset = dataset["test"]
        
#         return train_dataset, test_dataset, column_names
    
#     def apply_transforms(self, dataset):
        
        
#         p = 0.05
#         # 🟢 ye wali transform target image pr lagey gi
#         image_transforms = transforms.Compose(
#             [
#                 transforms.Resize(
#                     self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
#                 ),
#                 transforms.CenterCrop(self.resolution),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )

#         # 🔴 ye wali transform donoon conditional images pr lgey gi
#         # no single channel image
#         conditioning_image_transforms = transforms.Compose(
#             [
#                 transforms.Resize(
#                     self.resolution,
#                     interpolation=transforms.InterpolationMode.BILINEAR,
#                 ),
#                 transforms.CenterCrop(self.resolution),
#                 # transforms.Grayscale(), # 👈 donot need the grayscale image
#                 transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
#             ]
#         )
        
#         def preprocess_train_multi_condition(examples, columns_names):
#             """
#             Apply the transformations on the dataset columns
#             """
#             Console().print(f"type examples: [red]{examples.__class__}")
#             Console().print(f"[red]columns names [green]{columns_names}")
#             # read the image and both conditioning images
#             images = [image.convert("RGB") for image in examples[self.image_column]]
#             images = [image_transforms(image) for image in images]
#             examples["pixel_values"] = images
            
#             for idx, cond_column in enumerate(columns_names):
#                 logger.info(f"transforming {cond_column} column")
#                 conditioning_images = [
#                                         image.convert("RGB") for image in examples[cond_column]
#                                     ]
#                 conditioning_images = [
#                                         conditioning_image_transforms(image) for image in conditioning_images
#                                     ]
#                 examples[f"conditioning_pixel_values{idx+1}"] = conditioning_images
            
#             return examples

        
#         if not isinstance(self.image_column, str):
#             raise ValueError("image_column must be a string")
        
#         # if the conditioning image column is a list, then we have multiple conditioning images columns
#         _condition_columns_names = []
#         # if isinstance(self.conditioning_image_column, list):
#         for conditioning_image_column in self.conditioning_image_column:
#             _condition_columns_names.append(conditioning_image_column)

#         self.columns_names=_condition_columns_names
#         logger.info("Conditioning image columns to be subjected to transforms: {}".format(self.columns_names))
    
    
        
#         _process_func = functools.partial(preprocess_train_multi_condition,
#                                                columns_names=self.columns_names)
        
            
    
#         # dataset = dataset.map(lambda record : preprocess_train_multi_condition(record, columns_names=self.columns_names))
#         dataset = dataset.map(_process_func, batched=True, batch_size=20)

#         return dataset



# def import_model_class_from_model_name_or_path(
#     pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
# ):
#     text_encoder_config = PretrainedConfig.from_pretrained(
#         pretrained_model_name_or_path, subfolder=subfolder, revision=revision, cache='./cache'
#     )
#     model_class = text_encoder_config.architectures[0]

#     if model_class == "CLIPTextModel":
#         from transformers import CLIPTextModel

#         return CLIPTextModel
#     elif model_class == "CLIPTextModelWithProjection":
#         from transformers import CLIPTextModelWithProjection

#         return CLIPTextModelWithProjection
#     else:
#         raise ValueError(f"{model_class} is not supported.")



# if __name__ == '__main__':
        


#     # pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"


#     # # Load the tokenizers
#     # tokenizer_one = AutoTokenizer.from_pretrained(
#     #     pretrained_model_name_or_path,
#     #     subfolder="tokenizer",
#     #     revision=None,
#     #     use_fast=False,
#     #     cache='./cache'
#     # )
#     # tokenizer_two = AutoTokenizer.from_pretrained(
#     #     pretrained_model_name_or_path,
#     #     subfolder="tokenizer_2",
#     #     revision=None,
#     #     use_fast=False,
#     #     cache='./cache'
#     # )

#     # # import correct text encoder classes
#     # text_encoder_cls_one = import_model_class_from_model_name_or_path(
#     #     pretrained_model_name_or_path, None
#     # )
#     # text_encoder_cls_two = import_model_class_from_model_name_or_path(
#     #     pretrained_model_name_or_path, None, subfolder="text_encoder_2"
#     # )
#     # #%%
#     # # Load scheduler and models
#     # # noise_scheduler = DDPMScheduler.from_pretrained(
#     # #     self.pretrained_model_name_or_path, subfolder="scheduler"
#     # # )
#     # text_encoder_one = text_encoder_cls_one.from_pretrained(
#     #     pretrained_model_name_or_path,
#     #     subfolder="text_encoder",
#     #     revision=None,
        
#     # )
#     # text_encoder_two = text_encoder_cls_two.from_pretrained(
#     #     pretrained_model_name_or_path,
#     #     subfolder="text_encoder_2",
#     #     revision=None,
        
#     # )
#     # # vae_path = (
#     # #     self.pretrained_model_name_or_path
#     # #     if args.pretrained_vae_model_name_or_path is None
#     # #     else args.pretrained_vae_model_name_or_path
#     # # )
#     # # vae = AutoencoderKL.from_pretrained(
#     # #     vae_path,
#     # #     subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
#     # #     revision=None,
#     # #     cache='./cache'
#     # # )
#     # # unet = UNet2DConditionModel.from_pretrained(
#     # #     self.pretrained_model_name_or_path, subfolder="unet", revision=None, cache='./cache'
#     # # )

#     # text_encoders = [text_encoder_one, text_encoder_two]
#     # tokenizers = [tokenizer_one, tokenizer_two]
#     logger.info(f"text encoders and tokenizers ready")


#     model = HF_Dataset_Module(
#                     dataset_name = None,
#                     train_data_dir = "HR_VITON",
#                     caption_column = "caption",
#                     # text_encoders = text_encoders,
#                     # tokenizers = tokenizers,
#                     conditioning_image_column = ["cloth", "openpose_img"],
#                     image_column = "image",
#                     batch_size = 2,
#                     dataloader_num_workers = 4,
#                     max_train_samples = 10,
#                     shuffle = True,
#                     resolution = 1024,
#                     proportion_empty_prompts = 0.0,
#                     split_ratio = 0.1
#     )

    

#     # Use the Lightning Trainer just to check data loading
#     trainer = pl.Trainer(
#                             # fast_dev_run=True, 
#                             devices=[0,1], accelerator="gpu",
#                             precision=16,
#                          )  # fast_dev_run just runs 1 batch of train to check everything is in order
#     trainer.fit(model)










