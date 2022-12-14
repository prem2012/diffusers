export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/prem/dev/data/prem_512_selected/"
export CLASS_DIR="/home/prem/dev/data/man/"
export OUTPUT_DIR="/home/prem/dev/models/prem-512-v0"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="a photo of pknpknpknat man" \
  --class_prompt="a photo of a man" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --gradient_accumulation_steps=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=7e-7 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --num_class_images=378 \
  --max_train_steps=3000 \
  --mixed_precision=no
