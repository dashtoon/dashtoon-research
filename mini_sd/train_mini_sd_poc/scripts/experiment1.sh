export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="/home/ayushman/code/datasets/cached/hf-datasets-----FakeJourney"
export UNET_CONFIG_PATH="/home/ayushman/code/custom/train_mini_sd_poc/configs"
export TIMESTAMP=$(date +%Y%m%d-%H%M%S)
export OUTPUT_DIR="/home/ayushman/code/custom/train_mini_sd_poc/outputs/20231026-130149"

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=2 main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --unet_config_path=$UNET_CONFIG_PATH \
  --image_column="image" \
  --caption_column="text" \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=32 \
  --num_train_epochs=10000000000 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --use_ema \
  --dataloader_num_workers=16 \
  --report_to="wandb" \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=1 \
  --validation_epochs=4 \
  --tracker_project_name="Stable Diffusion Better Prompts POC" \
  --resume_from_checkpoint="latest" \
  --validation_prompts "A painting of a vase filled with flowers on a table." "A digital painting of an Asian woman in a traditional dress and tiger makeup holding a tiger cub" "Baby Yoda sitting in a brown chair holding a cup of coffee" "A close up of a pink flower with a dark background"
