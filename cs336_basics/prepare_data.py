import argparse
import numpy as np
import pickle
from tqdm import tqdm
import os


def load_tokenizer(model_path: str):
    from cs336_basics.BPE.Tokenizer import Tokenizer
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    tokenizer = Tokenizer(
        vocab=data['vocab'],
        merges=data['merges'],
        special_tokens=data.get('special_tokens', [])
    )
    
    print(f"Loaded tokenizer from {model_path}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Special tokens: {tokenizer.special_tokens}")
    
    return tokenizer


def estimate_tokens(input_path: str, tokenizer, sample_size: int = 10000):
    file_size = os.path.getsize(input_path)
    
    # Read a sample
    with open(input_path, 'r', encoding='utf-8') as f:
        sample = f.read(sample_size)
    
    # Tokenize sample
    sample_tokens = tokenizer.encode(sample)
    
    # Estimate total tokens
    chars_per_token = len(sample) / len(sample_tokens)
    estimated_tokens = int(file_size / chars_per_token)
    
    print(f"File size: {file_size:,} bytes")
    print(f"Sample size: {len(sample)} chars -> {len(sample_tokens)} tokens")
    print(f"Estimated chars per token: {chars_per_token:.2f}")
    print(f"Estimated total tokens: {estimated_tokens:,}")
    
    return estimated_tokens


def tokenize_file_streaming(
    input_path: str,
    output_path: str,
    tokenizer,
    dtype: str = 'uint16',
    chunk_size_mb: int = 10
):
    dtype_map = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'uint32': np.uint32,
        'int32': np.int32,
        'int64': np.int64,
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. Choose from {list(dtype_map.keys())}")
    
    np_dtype = dtype_map[dtype]
    
    # Get file size for progress bar
    file_size = os.path.getsize(input_path)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # First pass: tokenize and save to temporary file
    print(f"\nTokenizing {input_path}...")
    print(f"Processing in {chunk_size_mb}MB chunks")
    
    tokens = []
    total_tokens = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar:
            buffer = ""
            
            while True:
                chunk = f.read(chunk_size_bytes)
                if not chunk:
                    # Process remaining buffer
                    if buffer:
                        chunk_tokens = tokenizer.encode(buffer)
                        tokens.extend(chunk_tokens)
                        total_tokens += len(chunk_tokens)
                    break
                
                # Add to buffer
                buffer += chunk
                pbar.update(len(chunk.encode('utf-8')))
                
                # Find last complete line in buffer
                last_newline = buffer.rfind('\n')
                if last_newline != -1:
                    # Process complete lines
                    complete_text = buffer[:last_newline + 1]
                    buffer = buffer[last_newline + 1:]
                    
                    # Tokenize
                    chunk_tokens = tokenizer.encode(complete_text)
                    tokens.extend(chunk_tokens)
                    total_tokens += len(chunk_tokens)
    
    print(f"Total tokens: {total_tokens:,}")
    
    # Convert to numpy array and save
    print(f"Converting to {dtype} array...")
    token_array = np.array(tokens, dtype=np_dtype)
    
    print(f"Saving to {output_path}...")
    np.save(output_path, token_array)
    
    # Verify the saved file
    print("\nVerifying saved file...")
    loaded = np.load(output_path, mmap_mode='r')
    print(f"Saved shape: {loaded.shape}")
    print(f"Saved dtype: {loaded.dtype}")
    print(f"First 20 tokens: {loaded[:20]}")
    print(f"Memory-mapped file size: {os.path.getsize(output_path):,} bytes")
    
    return token_array


def tokenize_file_inmemory(
    input_path: str,
    output_path: str,
    tokenizer,
    dtype: str = 'uint16'
):
    """
    Tokenize a text file that fits in memory.
    Faster than streaming but requires enough RAM.
    """
    dtype_map = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'uint32': np.uint32,
        'int32': np.int32,
        'int64': np.int64,
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. Choose from {list(dtype_map.keys())}")
    
    np_dtype = dtype_map[dtype]
    
    # Read entire file
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File size: {len(text):,} characters")
    
    # Tokenize
    print("Tokenizing...")
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    # Convert to numpy array
    print(f"Converting to {dtype} array...")
    token_array = np.array(tokens, dtype=np_dtype)
    
    # Save
    print(f"Saving to {output_path}...")
    np.save(output_path, token_array)
    
    # Verify
    print("\nVerifying saved file...")
    loaded = np.load(output_path, mmap_mode='r')
    print(f"Saved shape: {loaded.shape}")
    print(f"Saved dtype: {loaded.dtype}")
    print(f"First 20 tokens: {loaded[:20]}")
    
    return token_array


def get_vocab_range(tokenizer):
    token_ids = list(tokenizer.vocab.keys())
    return min(token_ids), max(token_ids)


def check_dtype_compatibility(tokenizer, dtype: str):
    min_id, max_id = get_vocab_range(tokenizer)
    
    dtype_ranges = {
        'uint8': (0, 255),
        'uint16': (0, 65535),
        'uint32': (0, 4294967295),
        'int32': (-2147483648, 2147483647),
        'int64': (-9223372036854775808, 9223372036854775807),
    }
    
    if dtype not in dtype_ranges:
        return False, f"Unknown dtype: {dtype}"
    
    min_val, max_val = dtype_ranges[dtype]
    
    if min_id < min_val or max_id > max_val:
        return False, (
            f"Vocabulary range [{min_id}, {max_id}] exceeds {dtype} range [{min_val}, {max_val}]. "
            f"Use a larger dtype."
        )
    
    return True, f"Vocabulary range [{min_id}, {max_id}] fits in {dtype} range [{min_val}, {max_val}]"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare tokenized data for Transformer LM training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output numpy file path (.npy)')
    parser.add_argument('--tokenizer-model', type=str, required=True,
                       help='Path to BPE model pickle file')
    parser.add_argument('--dtype', type=str, default='uint16',
                       choices=['uint8', 'uint16', 'uint32', 'int32', 'int64'],
                       help='Data type for token IDs')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode for large files (slower but less memory)')
    parser.add_argument('--chunk-size-mb', type=int, default=10,
                       help='Chunk size in MB for streaming mode')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("Data Preparation for Transformer LM Training")
    print("="*70)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_model)
    
    # Check dtype compatibility
    print("\n2. Checking dtype compatibility...")
    compatible, message = check_dtype_compatibility(tokenizer, args.dtype)
    print(message)
    if not compatible:
        print("ERROR: Dtype incompatible with vocabulary!")
        return
    
    # Estimate tokens (for progress tracking)
    if not args.streaming:
        print("\n3. Estimating token count...")
        try:
            estimate_tokens(args.input, tokenizer)
        except Exception as e:
            print(f"Could not estimate tokens: {e}")
    
    # Tokenize file
    print(f"\n4. Tokenizing file (streaming={args.streaming})...")
    try:
        if args.streaming:
            tokens = tokenize_file_streaming(
                args.input,
                args.output,
                tokenizer,
                args.dtype,
                args.chunk_size_mb
            )
        else:
            tokens = tokenize_file_inmemory(
                args.input,
                args.output,
                tokenizer,
                args.dtype
            )
        
        print("\n" + "="*70)
        print("SUCCESS! Data preparation complete.")
        print("="*70)
        print(f"\nOutput file: {args.output}")
        print(f"Total tokens: {len(tokens):,}")
        print(f"Dtype: {args.dtype}")
        print("\nYou can now use this file for training:")
        print(f"  python train.py --train-data-path {args.output} ...")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()