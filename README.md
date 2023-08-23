

# Install requirements

```bash
pip install -r requirements.txt
```


# Train

Devices corresponds to the gpus to be used for training, if machine has 4 gpus then devices can be 0 1 2 3
 
```bash

python train.py --n_adapters 2   --adapter_names sketch openpose  --learning_rate  1e-5  --dataset_dir  dress_pose_depth_scribble_captions_v1.0_3_million   --shuffle True  --conditioning_image_column   scribble_path  pose_path  --batchsize  2  --image_column image_path  --caption_column  caption  --max_train_samples  30  --dataset_cache_device "cuda:0"  --checkpoint_every_n_train_steps  5  --checkpoint_dirpath  sdxl_ckpts_dummy  --max_epochs 5  --accumulate_grad_batches 4  --devices 0 1 2
```


# Details of the dataset

In above example Dataset is **Huggingface datasets** format dataset stored locally.

NAMES OF THE DATASET COLUMNS ARE:

 - Conditioning image columns: ['scribble_path', 'pose_path']
 - Target image column: ['image_path']
 - Caption column: ['caption']


# 📢 Note

All conditional images must be RGB images.






