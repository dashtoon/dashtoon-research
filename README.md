
# training start


```bash
accelerate launch t2i_trainer.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="ckpts" --dataset_name="SaffalPoosh/scribble_controlnet_dataset" --resolution=1024  --learning_rate=1e-5 --train_batch_size=2  --gradient_accumulation_steps=4 --num_train_epochs=5 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='image'  --conditioning_image_column="scribble"  --caption_column="caption"

```



Multiple CoAdapters
```bash
accelerate launch t2i_trainer.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="ckpts_joint_sketch_pose" --dataset_name="dress_code_dataset_scribbles" --resolution=1024  --learning_rate=1e-5 --train_batch_size=2  --gradient_accumulation_steps=4 --num_train_epochs=5 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='images'  --conditioning_image_column="scribble" --conditioning_image_column2="skeletons"  --caption_column="captions"


```


# train small controlnet
 **Dataset** : dress_code_dataset_scribbles
```bash

accelerate launch  train_small_controlnet.py    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"    --output_dir=small_controlnet   --dataset_name="dress_code_dataset_scribbles" --resolution=1024  --learning_rate=1e-5 --train_batch_size=2   --num_train_epochs=4 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='images'  --conditioning_image_column="scribble" --caption_column="captions"  --gradient_accumulation_steps=4


```


 **Dataset** : dress_pose_depth_scribble_captions_v1.0_3_million
```bash
accelerate launch  train_small_controlnet.py    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"    --output_dir=small_controlnet_scrapped_data   --dataset_name="dress_pose_depth_scribble_captions_v1.0_3_million" --resolution=1024  --learning_rate=1e-5 --train_batch_size=4   --num_train_epochs=4 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='image_path'  --conditioning_image_column="scribble_path" --caption_column="caption"


```




# Generator + T2I + Coadapter train

## **HR-VTON dataset**


```bash
accelerate launch vton_train.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="ckpts_vton_gen_t2i_sketch_pose" --dataset_name="HR_VITON" --resolution=1024  --learning_rate=1e-5 --train_batch_size=2  --gradient_accumulation_steps=4 --num_train_epochs=10 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='image'  --conditioning_image_column="cloth" --conditioning_image_column2="openpose_img"  --caption_column="caption"
```



## Long training

```bash
accelerate launch --config_file   vton_t2i_sketch_pose_acclerate.yaml  vton_train.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="ckpts_vton_gen_t2i_sketch_pose_10_epochs" --dataset_name="HR_VITON" --resolution=1024  --learning_rate=1e-5 --train_batch_size=4  --gradient_accumulation_steps=4 --num_train_epochs=10 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='image'  --conditioning_image_column="cloth" --conditioning_image_column2="openpose_img"  --caption_column="caption"   --report_to  wandb  
```




## **3 million scrapped images**

```bash
accelerate launch vton_train.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="ckpts_vton_gen_t2i_sketch_pose_3_million" --dataset_name="dress_pose_depth_scribble_captions_v1.0_3_million" --resolution=1024  --learning_rate=1e-5 --train_batch_size=2  --gradient_accumulation_steps=4 --num_train_epochs=20 --checkpointing_steps=500 --checkpoints_total_limit=10   --image_column='image_path'  --conditioning_image_column="scribble_path" --conditioning_image_column2="pose_path"  --caption_column="caption"
```



| Image | Scribble | Caption |
|-------|----------|---------|
| image1      | condition1         |  caption1       |
| image2      | condition2         |  caption2       |
| image3      | condition3         |  caption3       |



# How to split/analyze the weights of coadapter and individual adapters


```bash 
python split_weights.py
```


# inference pipeline single adapter Gradio

```bash
python inference_capsule_coadpter.py
```

# inference pipeline Multi adapter Gradio

```bash
python multi_inference_capsule_coadpter.py
```



# dress code dataset - generate scribbles
load the dress code dataset (already has skeleton), so generating scribble images and pushing to hub.

```bash
python process_dress_code_dataset.py
```



# Sample scribble images:

following dir. has sample scribble images
```bash
sample_scribbles/*
```


# Datasets 

## Dress code dataset prepared for joint training

following dir.

```bash
from datasets import load_dataset
ds = load_dataset(dress_code_dataset_scribbles)
```
## COCO scribble dataset for single coadapter training
following dir.

```bash
from datasets import load_dataset
ds = load_dataset(coco_scribble_captions)
```

# Adapter / Coadapter

see following script:
```bash
adapter.py
```


# Generator-With-T2I Adapter

See following

```
/workspace/generator_model_adapter.py

```

<!-- 
# T2I-Adapters & Training code for SDXL in Diffusers

This is the initial code to make T2I-Adapters work in SDXL with Diffusers. The UNet has changed in SDXL making changes necessary to the diffusers library to make T2IAdapters work. I intend to upstream the code to diffusers once I get it more settled. A training script is also included.

## Sample Input and Output

Here is a sample input mask:

![Input Mask](demo_mask.png)

And here is the corresponding output:

![Sample Output](demo.png)

An iPython notebook is included to show how to use the the pipeline. Please note that the segmentation adapter is a 1 channel adapter, so the input mask should be a single channel mask if you are going to use it. You can train an adapter to use any given number of channels. -->