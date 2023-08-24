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
    
)
# from diffusers.models.adapter import LightAdapter
from adapter import T2IAdapter, CoAdapterFuser, MultiAdapter
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

# ==========================================================================
#                          Dataset class making function                                  
# ==========================================================================

class prepare_dataset:
    def __init__(self,
                 dataset_name,
                 dataset_dir,
                 image_column,
                 caption_column,
                 conditioning_image_column : List[str],
                 device = "cuda:3",
                 resolution = 1024,
                 shuffle = True,
                 max_train_samples = 10000):
        
        
        self.resolution = resolution
        self.caption_column = caption_column
        self.image_column = image_column
        self.conditioning_image_column = conditioning_image_column
        # self.conditioning_image_column2 = conditioning_image_column2
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
        import datasets
        if isinstance(dataset, datasets.arrow_dataset.Dataset):
            # if dataset has no predefined split then split it.
            dataset = dataset.train_test_split(test_size=0.01)
            Console().print(f"dataset has been split", style='red')
            train_dataset = dataset["train"]
            test_dataset = dataset["test"]
        if isinstance(dataset, datasets.dataset_dict.DatasetDict):
            Console().print(f"dataset has predefined split...⚡", style='red')
            train_dataset = dataset["train"]
        
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
        # train_ds, proportion_empty_prompts=0.0, device = "cuda:3"
        self.process_train_ds(train_dataset, device=device)
    
    
    def get_dataset(self):
        '''
        Will be called by pl-data module to fetch the dataset
        '''
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
        take train dataset and compute embeddings from both text encoders to feed the unet at training time and 
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
      
        def preprocess_train(examples, columns_names):
            images = [image.convert("RGB") for image in examples[self.image_column]]
            images = [image_transforms(image) for image in images]
            examples["pixel_values"] = images
            
            for idx, cond_column in enumerate(columns_names):
                logger.info(f"transforming {cond_column} column")
                conditioning_images = [
                                        image.convert("RGB") for image in examples[cond_column]
                                    ]
                conditioning_images = [
                                        conditioning_image_transforms(image) for image in conditioning_images
                                    ]
                examples[f"conditioning_pixel_values{idx+1}"] = conditioning_images
            
            return examples

        # # 🔵 See the datasets lib. docs for this purpose
        # dataset = dataset.with_transform(preprocess_train)

        # return dataset
        
        if not isinstance(self.image_column, str):
            raise ValueError("image_column must be a string")
        
        # if the conditioning image column is a list, then we have multiple conditioning images columns
        _condition_columns_names = []
        # if isinstance(self.conditioning_image_column, list):
        for conditioning_image_column in self.conditioning_image_column:
            _condition_columns_names.append(conditioning_image_column)

        self.columns_names=_condition_columns_names
        logger.info("Conditioning image columns to be subjected to transforms: {}".format(self.columns_names))
    
    
        
        _process_func = functools.partial(preprocess_train,
                                               columns_names=self.columns_names)
        
            
    
        # dataset = dataset.map(lambda record : preprocess_train_multi_condition(record, columns_names=self.columns_names))
        # dataset = dataset.map(_process_func, batched=True, batch_size=20)
        dataset = dataset.with_transform(_process_func)

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
        '''
        create as many conditioning images batches as there are conditioning images columns specified by the user
        it will be passed to the dataloader to create a batch
        
        Parameters
        ----------
        examples :  a single record

        '''        
        
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # ============================================================
        _condition = {}
        for k in range(len(self.conditioning_image_column)):
            # conditioning images sketch ka batch
            conditioning_pixel_values = torch.stack(
                [example[f"conditioning_pixel_values{k+1}"] for example in examples]
            )
            conditioning_pixel_values = conditioning_pixel_values.to(
            memory_format=torch.contiguous_format
            ).float()
            _condition[f'conditioning_pixel_values{k+1}'] = conditioning_pixel_values
        
        
        
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
            "condition" : _condition,
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
    def __init__(self, 
                 with_coadapter,
                 dataset_name, 
                 n_adapters, 
                 adapter_names, 
                 learning_rate,
                 adam_weight_decay, 
                 dataset_dir="dress_pose_depth_scribble_captions_v1.0_3_million",
                 shuffle=True,
                 conditioning_image_column = ["scribble_path", "pose_path"],
                 workers = 12,
                 batchsize = 2,
                 pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
                 pretrained_vae_model_name_or_path = None
                 ) -> None:
        super().__init__()
        self.with_coadapter = with_coadapter
        self.n_adapters = n_adapters
        self.adapter_names = adapter_names
        self.learning_rate = learning_rate
        self.adam_weight_decay = adam_weight_decay
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained_vae_model_name_or_path = pretrained_vae_model_name_or_path
        
        self.dataset_name = dataset_name,
        self.dataset_dir = dataset_dir, 
        self.shuffle = shuffle,
        
        self.conditioning_image_column = conditioning_image_column,
        
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
        
        if self.with_coadapter == True and self.n_adapters > 1:
            # fuse individual adapters using the CoAdapters approach
            Console().log(f"📢\t[red]t2i adapters will be fused using [green]co-adapter")
            self.model = self.setup_coadapter_adapters_model(n_adapters = self.hparams.n_adapters, adapter_names = self.hparams.adapter_names)
        if self.with_coadapter == False and self.n_adapters > 1:
            # fuse individual adapters using the MultiAdapter approach
            self.model = self.setup_multiadapter_adapters_model(n_adapters = self.hparams.n_adapters)
            Console().log(f"📢\t[red]t2i adapters will be fused using [green]multi-adapter")
        if self.n_adapters == 1:
            # fuse individual adapters using the MultiAdapter approach
            self.model = self.setup_single_adapter_model()
        
        
        
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
        # get all the conditions
        # Console().print(f"[red] {self.conditioning_image_column}[green] 🔴 debugger")
        
        _condition_batches = []
        for k in range(len(self.conditioning_image_column[0])):
            _condition_batches.append(batch['condition'][f'conditioning_pixel_values{k+1}'])
            
        
        # # log the images to wandb
        # to_pil = transforms.ToPILImage()
        # t2i_adapter_image_wandb = [to_pil(k) for k in t2i_adapter_image]
        # t2i_adapter_image2_wandb = [to_pil(k) for k in t2i_adapter_image2]
        

        if self.with_coadapter == True and self.n_adapters > 1:
            # for co-adapter fusion approach
            down_block_res_samples = self.model(*_condition_batches)
        
        elif self.with_coadapter == False and self.n_adapters > 1:
            # for the multi-adapter fusion approach
            # concat the batches in channel dim. since its the format multi-adapter is expecting
            down_block_res_samples = self.model(torch.concat(_condition_batches, dim=1))
            
        elif self.n_adapters == 1:
            # for single adapter, apply the conditioning image to the adapter
            # _condition_batches will be a list of 1 
            down_block_res_samples = self.model(*_condition_batches)
        
        
        # 🔴 For debugging, in ko ghair-comment krna hai. yahan concat feature k masail aatey hain
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
        
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6),
            'interval': 'epoch',  # or 'step' for step-wise learning rate updates
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
        
    def setup_multiadapter_adapters_model(self, n_adapters : int)-> MultiAdapter:
        '''
        takes mutliple individual T2I adapters and fuses them using the MultiAdapter approach

        Parameters
        ----------
        n_adapters : int
            no. of T2I adapters to fuse

        Returns
        -------
        _type_
            _description_
        '''        
        # fuse the individual t2idapters using Multiadapter
        all_adapters = []
        for k in range(n_adapters):
            adapter =  T2IAdapter(channels=[320, 640, 1280], in_channels=3) 
            adapter.requires_grad_(True)
            
            all_adapters.append(adapter)
        
        multi_adapter = MultiAdapter(all_adapters)
        multi_adapter.requires_grad_(True)
        Console().log(f'[yellow]multi adapter has been setup, grad set to [red] True')
        
        return multi_adapter
    
    
    def setup_single_adapter_model(self,)->T2IAdapter:
        '''
        sutup single T2I adapter and return it

        Returns
        -------
        T2IAdapter
            
        '''        
        adapter =  T2IAdapter(channels=[320, 640, 1280], in_channels=3)
        adapter.requires_grad_(True)
        Console().log(f'[yellow]single T2I adapter has been setup, grad set to [red] True')
        
        return adapter
    
    

    def setup_coadapter_adapters_model(self, n_adapters : int = 2, adapter_names : List[str] = ['sketch', 'openpose']):
        
        
        all_adapters = []
        for k in range(n_adapters):
            adapter =  T2IAdapter(channels=[320, 640, 1280], in_channels=3) 
            adapter.requires_grad_(True)
            
            all_adapters.append(adapter)
        
        
        # 💀  define coadapter
        coadapter_fuser = CoAdapterFuser()
        coadapter_fuser.requires_grad_(True)
        
        #  💀 define the forward pass fuser (its purpose is only to encapsulate the forward pass)
        fuser = Capsule_CoAdapter(*all_adapters, coadapter_fuser=coadapter_fuser, adapter_order=adapter_names)
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


# ==========================================================================
#                         yeh Data Module related cheezain hain                                  
# ==========================================================================

class dm(pl.LightningDataModule):
    def __init__(self, 
                 dataset_name, 
                 dataset_dir, 
                 shuffle, 
                 image_column, 
                 caption_column, 
                 conditioning_image_column, 
                 max_train_samples, 
                 dataset_cache_device, 
                 batchsize, 
                 workers):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.image_column = image_column
        self.caption_column = caption_column
        self.conditioning_image_column = conditioning_image_column
        
        self.max_train_samples = max_train_samples
        self.dataset_cache_device = dataset_cache_device
        self.batchsize = batchsize
        self.workers = workers
        
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        self.dataset_ = prepare_dataset(
                            dataset_name=self.dataset_name,
                            dataset_dir=self.dataset_dir, 
                            shuffle=self.shuffle,
                            image_column=self.image_column,
                            caption_column=self.caption_column,
                            conditioning_image_column=self.conditioning_image_column,
                            # conditioning_image_column2=self.conditioning_image_column2,
                            max_train_samples=self.max_train_samples,
                            device=self.dataset_cache_device
                        )
        Console().log(f"🟩 setup stage called .... ", style='red')

    def train_dataloader(self):

        dl = self.dataset_.get_dataloader(batchsize=self.batchsize, workers=self.workers, shuffle=self.shuffle)
        return dl



if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='SDXL-1.0 model with multiple conditions')

    # Arguments for the first function
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset from hugging face, It should be public')
    parser.add_argument('--n_adapters', type=int, required=True, help='Number of adapters, should be equal to the number of adapter names given')
    parser.add_argument('--adapter_names', type=str, nargs='+', default=['sketch', 'openpose'], help='List of strings for adapter names, it can be sketch, keypose, seg, depth, canny, style, color, openpose')
    parser.add_argument('--learning_rate', type=float,  default=1e-5, help='Learning rate')
    parser.add_argument('--adam_weight_decay', type=float, default=5e-5, help='Weight decay for Adam optimizer')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Directory of the dataset locally stored in huggingface datasets format')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset')
    parser.add_argument('--conditioning_image_column', nargs='+', default=["scribble_path", "pose_path"], help='names of columns from the dataset that need to be used as conditioning images, should correspond to adapter names argument given and equal to n_adapters')
    parser.add_argument('--workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--batchsize', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help='Path or name of the pretrained SDXL model')
    parser.add_argument('--pretrained_vae_model_name_or_path', type=str, default=None, help='Path or name of the pretrained VAE model')
    parser.add_argument('--image_column', type=str, default='image_path', help='column name from dataset which corresponds to the target Image')
    parser.add_argument('--caption_column', type=str, default="caption", help='column name from dataset which corresponds to the caption')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use for training')
    parser.add_argument('--dataset_cache_device', type=str, default='cuda:0', help='Device for caching the caption encodings for both text encoders to save time during training, default is cuda:0')

    # WandbLogger arguments
    parser.add_argument('--wandb_name', type=str, default="sdxl_adapter_coadapter-pl-lightning", help='Name for WandbLogger')
    parser.add_argument('--wandb_save_dir', type=str, default="./pl_lightning_ckpts", help='Save directory for WandbLogger')
    parser.add_argument('--log_model', type=str, default="all", help='Logging option for the model, wandb logger')
    parser.add_argument('--project', type=str, default="sdxl_codadpters", help='Project name for WandbLogger')

    # ModelCheckpoint arguments
    parser.add_argument('--checkpoint_every_n_train_steps', type=int, default=1000, help='Save checkpoint every n train steps')
    parser.add_argument('--checkpoint_dirpath', type=str, default="sdxl_ckpts_pl", help='Directory path for saving checkpoints')
    
    
    # Trainer
    parser.add_argument("--max_epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--accumulate_grad_batches", default=4, type=int, help="no. of gradient batches to accumulate")
    parser.add_argument('--devices', type=int, nargs='+', default=[0,1], help='list of gpu devices to use for training')


    # 1. if n_adapters 1 train a single adapter
    # 2. if n_adapters > 1 train with coadapter
    # 3. if n_adapters > 1 and --use_codapter False then Multi Adapter
    parser.add_argument("--with_coadapter",  action='store_true'  , help="if this flag is given then multiple adapters will be fused using the codadapter approach")


    args = parser.parse_args()
    
    
    
    with Console().status(f"preparing things ....", spinner='material') as status:
            
        model = sdxl_pl_model(
                            with_coadapter = args.with_coadapter,
                            dataset_name = args.dataset_name, 
                            dataset_dir=args.dataset_dir,
                            n_adapters = args.n_adapters, 
                            conditioning_image_column = args.conditioning_image_column,
                            adapter_names = args.adapter_names, 
                            learning_rate = args.learning_rate,
                            adam_weight_decay = args.adam_weight_decay,
                            # dataset_cache_device='cuda:0',
                            # max_train_samples=1000, 
                            batchsize=args.batchsize)
      
        status.update("model class initailized....", spinner='aesthetic')    
    
        data_module = dm(dataset_dir=args.dataset_dir,
                        dataset_name=args.dataset_name,
                        shuffle=args.shuffle,
                        image_column=args.image_column,
                        caption_column=args.caption_column,
                        conditioning_image_column = args.conditioning_image_column,
                            # conditioning_image_column2 = "pose_path",
                        max_train_samples = args.max_train_samples,
                        dataset_cache_device = args.dataset_cache_device,
                        workers = args.workers,
                        batchsize = args.batchsize,
                        )
    
        status.update("Data class initailized....", spinner='runner')   
        
    
        
        wandb_logger = WandbLogger(name="sdxl_adapter_coadapter-pl-lightning", 
                                   save_dir="./pl_lightning_ckpts", 
                                   log_model=False, 
                                   project="sdxl_codadpters",
                                   )
        
        wandb_logger.watch(model, log="all", log_freq=50, log_graph=True)
        
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every_n_train_steps, 
                                                           dirpath=args.checkpoint_dirpath, 
                                                           filename='{epoch}-{step}', 
                                                           verbose=True, 
                                                           save_weights_only=False, 
                                                           save_last=True,
                                                           save_on_train_epoch_end =True)
        status.update(f"callbacks defined....", spinner="dots")
        status.update(f"[red]Training Started....\n\n", spinner="pong")
        trainer = pl.Trainer(
                                logger=wandb_logger,
                                callbacks=[checkpoint_callback, lr_monitor],
                                accelerator="auto",
                                precision='bf16',
                                devices=args.devices,
                                strategy='ddp',
                                max_epochs=args.max_epochs,
                                accumulate_grad_batches=args.accumulate_grad_batches,
                                log_every_n_steps = 5,
                            )
        
        trainer.fit(model, data_module)
