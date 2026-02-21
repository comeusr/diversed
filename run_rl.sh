#!/bin/bash

OPTIMIZER=AdamW
LR=5e-4
# DATA=wmt
# DATA=xsum
DATA=cnndm
LOSS=reinforce
MODEL=gemma # llama
# MODEL=llama
EPOCH=2
LAMBDA=10
TARGET_DRAFT_W=0.2
CLASS=google # meta-llama
# CLASS=meta-llama
# DRAFT_MODEL=Llama-3.2-1B-Instruct
# TARGET_MODEL=Llama-3.1-8B-Instruct
DRAFT_MODEL=gemma-3-4b-it # Llama-3.2-1B-Instruct
TARGET_MODEL=gemma-3-12b-it # Llama-3.1-8B-Instruct
TEMP=0.0

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found. Please create it with your API tokens."
  exit 1
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

    # model.max_prompt_length=512 \ # max length of the prompt, if less than this you do a left padding otherwise truncate
    # model.max_length=640 \ prompt + generation < max_length
    # model.max_tokens=128 \ generation alone 

python rl_train.py loss=$LOSS model=$MODEL datasets=[$DATA] optimizer=$OPTIMIZER \
    exp_name=${TARGET_MODEL}_${DRAFT_MODEL}_26Aug_TEMP${TEMP}_${LOSS}_${OPTIMIZER}_reg_scale${LAMBDA}_target${TARGET_DRAFT_W}_${LR} \
    lr=${LR} \
    global_epochs=$EPOCH \
    n_examples=400 \
    wandb.project=Ensemble_${DATA} \
    model.draft_name_or_path=${CLASS}/${DRAFT_MODEL} \
    model.target_name_or_path=${CLASS}/${TARGET_MODEL} \
    model.max_prompt_length=512 \
    model.max_length=640 \
    model.max_tokens=128 \
    model.do_sample=false \
    model.temperature=${TEMP} \
    cache_dir=../data/${DATA}/model \
    model.use_peft=false model.batch_size=8 \
    model.gradient_accumulation_steps=1 \
    model.reg_scale=${LAMBDA} \
    model.target_w_draft=${TARGET_DRAFT_W} \
    model.save_freqs=4
