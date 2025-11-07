#!/bin/bash

# Quick Start Script for Transformer LM Training
# This script demonstrates the complete pipeline from data preparation to training

set -e  # Exit on error

echo "================================================================"
echo "Transformer LM Training - Quick Start"
echo "================================================================"

# Configuration
BPE_MODEL="/mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/BPE/bpe_model_TinyStoriesV2-GPT4-train.pkl"
TRAIN_TEXT="/mnt/d/Stanford_LLM/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
VAL_TEXT="/mnt/d/Stanford_LLM/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_DATA="/mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/train.npy"
VAL_DATA="/mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/val.npy"
CHECKPOINT_DIR="./checkpoints"

# ================================================================
# Step 1: Train BPE Tokenizer (if not already trained)
# ================================================================

if [ ! -f "$BPE_MODEL" ]; then
    echo ""
    echo "Step 1: Training BPE tokenizer..."
    echo "----------------------------------------------------------------"
    
    python -c "
from cs336_basics.BPE.BPE import run_train_bpe

vocab, merges = run_train_bpe(
    input_path='$TRAIN_TEXT',
    vocab_size=$VOCAB_SIZE,
    special_tokens=['<|endoftext|>'],
    verbose=True,
    save_path='$BPE_MODEL'
)
"
    echo "✓ BPE tokenizer trained and saved to $BPE_MODEL"
else
    echo ""
    echo "Step 1: BPE tokenizer found at $BPE_MODEL (skipping training)"
fi

# ================================================================
# Step 2: Prepare Training Data
# ================================================================

if [ ! -f "$TRAIN_DATA" ]; then
    echo ""
    echo "Step 2: Preparing training data..."
    echo "----------------------------------------------------------------"
    
    python prepare_data.py \
        --input "$TRAIN_TEXT" \
        --output "$TRAIN_DATA" \
        --tokenizer-model "$BPE_MODEL" \
        --dtype uint16
    
    echo "✓ Training data prepared: $TRAIN_DATA"
else
    echo ""
    echo "Step 2: Training data found at $TRAIN_DATA (skipping preparation)"
fi

# ================================================================
# Step 3: Prepare Validation Data
# ================================================================

if [ ! -f "$VAL_DATA" ]; then
    echo ""
    echo "Step 3: Preparing validation data..."
    echo "----------------------------------------------------------------"
    
    python prepare_data.py \
        --input "$VAL_TEXT" \
        --output "$VAL_DATA" \
        --tokenizer-model "$BPE_MODEL" \
        --dtype uint16
    
    echo "✓ Validation data prepared: $VAL_DATA"
else
    echo ""
    echo "Step 3: Validation data found at $VAL_DATA (skipping preparation)"
fi

# ================================================================
# Step 4: Train Model
# ================================================================

SUMMARY_FILE="test(lr_and_min_lr).json"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "[]" > "$SUMMARY_FILE"
    echo "Created summary file: $SUMMARY_FILE"
fi

echo ""
echo "Step 4: Training model..."
echo "----------------------------------------------------------------"

uv run train.py \
    --config ./config.json

echo ""
echo "================================================================"
echo "Training Complete!"
echo "================================================================"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "  - best_model.pt     : Best validation loss"
echo "  - final_model.pt    : Final model"
echo "  - checkpoint_*.pt   : Periodic checkpoints"
echo ""
echo "To resume training:"
echo "  python train.py --config $CHECKPOINT_DIR/config.json --resume-from $CHECKPOINT_DIR/checkpoint_iter_5000.pt"
echo ""
echo "To use the model for inference:"
echo "  See inference_example.py"
echo ""


# python prepare_data.py --input /mnt/d/Stanford_LLM/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt --output train.npy --tokenizer-model /mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/BPE/bpe_model_TinyStoriesV2-GPT4-train.pkl
# python prepare_data.py --input /mnt/d/Stanford_LLM/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt --output val.npy --tokenizer-model /mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/BPE/bpe_model_TinyStoriesV2-GPT4-train.pkl