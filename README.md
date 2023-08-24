

# Install requirements

```bash
pip install -r requirements.txt
```


# Train

Devices corresponds to the gpus to be used for training, if machine has 4 gpus then devices can be 0 1 2 3
 


# Traininng Adapters



## multiple T2I adapters + CoAdapter fusion

```bash

python train.py --dataset_name SaffalPoosh/deep_fashion_mask_pose_overlay  --n_adapters 2   --adapter_names sketch openpose  --learning_rate  1e-5   --shuffle True  --conditioning_image_column   mask_respective  pose  --batchsize  4 --image_column image  --caption_column  caption  --max_train_samples  5000  --dataset_cache_device "cuda:0"  --checkpoint_every_n_train_steps  200  --checkpoint_dirpath  sdxl_ckpts_dummy  --max_epochs 5  --accumulate_grad_batches 4  --devices 0 1 2 3  --with_coadapter
```



## multiple T2I adapters + MultiAdapter fusion

```bash

python train.py --dataset_name SaffalPoosh/deep_fashion_mask_pose_overlay  --n_adapters 2   --adapter_names sketch openpose  --learning_rate  1e-5   --shuffle True  --conditioning_image_column   mask_respective  pose  --batchsize  4 --image_column image  --caption_column  caption  --max_train_samples  5000  --dataset_cache_device "cuda:0"  --checkpoint_every_n_train_steps  200  --checkpoint_dirpath  sdxl_ckpts_dummy  --max_epochs 5  --accumulate_grad_batches 4  --devices 0 1 2 3 
```


## Single T2I adapter

```bash

python train.py --dataset_name SaffalPoosh/deep_fashion_mask_pose_overlay  --n_adapters 1   --adapter_names openpose  --learning_rate  1e-5   --shuffle True  --conditioning_image_column  pose  --batchsize  4 --image_column image  --caption_column  caption  --max_train_samples  5000  --dataset_cache_device "cuda:0"  --checkpoint_every_n_train_steps  200  --checkpoint_dirpath  sdxl_ckpts_dummy  --max_epochs 5  --accumulate_grad_batches 4  --devices 0 1 2 3 
```




> 📢 *Note*: All conditional images must be RGB images.




![Alt text](image-1.png)