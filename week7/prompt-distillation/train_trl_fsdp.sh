#!/bin/bash
# FSDP (Fully Sharded Data Parallel) training for 30B model
# FSDP shards the model across GPUs - much more memory efficient than DDP!
# Each GPU only holds ~1/8 of the model instead of the full model

set -x

# Configuration
MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
OUTPUT_DIR=${2:-"./models/prompt_distillation_trl"}
TRAIN_FILE=${3:-"./data/prompt_distillation_lang.jsonl"}

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "============================================"
echo "FSDP Training - Memory Efficient!"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Train file: $TRAIN_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Mode: FSDP (Fully Sharded Data Parallel)"
echo "============================================"
echo ""

# Check if training file exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Training file not found: $TRAIN_FILE"
    echo "Please run data generation first:"
    echo "  python create_data.py"
    exit 1
fi

# FSDP training with OpenAI-style hyperparameters
# When using LoRA, PEFT handles the auto-wrap policy automatically
accelerate launch \
    --mixed_precision bf16 \
    --num_processes=$NUM_GPUS \
    --use_fsdp \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_state_dict_type FULL_STATE_DICT \
    train_sft_trl.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr

echo ""
echo "============================================"
echo "✅ Training Complete!"
echo "============================================"
echo "Model saved to: $OUTPUT_DIR"
echo "Number of GPUs used: $NUM_GPUS"
echo "Mode: FSDP (each GPU held ~1/$NUM_GPUS of the model)"
echo ""
echo "Effective batch size: $((4 * 4 * NUM_GPUS))"
echo "  = per_device_batch_size (4)"
echo "  × gradient_accumulation_steps (4)"
echo "  × num_gpus ($NUM_GPUS)"
echo ""
echo "Memory Advantage:"
echo "  DDP: Each GPU holds 100% of model (~70GB)"
echo "  FSDP: Each GPU holds ~12.5% of model (~10-15GB)"
echo ""
echo "To test the model, run:"
echo "  python evaluate_trl.py --model_path $OUTPUT_DIR"
echo "============================================"

