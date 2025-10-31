import numpy as np
import torch
import pickle

def get_batch(x, batch_size, context_length, device='cpu'):
    """
    Sample a batch of input sequences and corresponding targets from the dataset.
    
    Args:
        x: numpy array (or memmap) containing token IDs with shape (n_tokens,)
        batch_size: number of sequences to sample
        context_length: length of each sequence
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')
    
    Returns:
        inputs: tensor of shape (batch_size, context_length) containing input sequences
        targets: tensor of shape (batch_size, context_length) containing next-token targets
    """
    # We need context_length + 1 tokens to get both inputs and targets
    max_start_idx = len(x) - context_length - 1
    
    if max_start_idx < 0:
        raise ValueError(f"Dataset too small: needs at least {context_length + 1} tokens")
    
    # Sample random starting indices for each sequence in the batch
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Collect sequences
    inputs = np.array([x[i:i + context_length] for i in start_indices])
    targets = np.array([x[i + 1:i + context_length + 1] for i in start_indices])
    
    # Convert to PyTorch tensors and move to device
    inputs = torch.from_numpy(inputs.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets.astype(np.int64)).to(device)
    
    return inputs, targets


# Example usage with memory-mapped data
def load_data_memmap(filepath, dtype=np.uint16):
    """
    Load data in memory-mapped mode for efficient access to large files.
    
    Args:
        filepath: path to the .npy file
        dtype: data type of the array (should match what was saved)
    
    Returns:
        memory-mapped numpy array
    """
    # Method 1: Using np.load with mmap_mode
    data = np.load(filepath, mmap_mode='r')
    
    # Method 2 (alternative): Using np.memmap directly if you know the shape
    # data = np.memmap(filepath, dtype=dtype, mode='r')
    
    return data


# Verification function
def verify_data(data, vocab_size):
    """
    Verify that memory-mapped data looks correct.
    
    Args:
        data: numpy array or memmap
        vocab_size: expected vocabulary size
    
    Returns:
        bool: True if data passes checks
    """
    print(f"Data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    print(f"First 20 tokens: {data[:20]}")
    
    # Check for values outside vocabulary
    min_val = np.min(data[:10000])  # Sample check to avoid loading entire array
    max_val = np.max(data[:10000])
    
    print(f"Min value (sample): {min_val}")
    print(f"Max value (sample): {max_val}")
    
    if min_val < 0 or max_val >= vocab_size:
        print(f"WARNING: Found values outside expected range [0, {vocab_size})")
        return False
    
    print("Data verification passed!")
    return True


def load_bpe_model(filepath):
    """
    Load BPE model from pickle file.
    
    Args:
        filepath: path to the .pkl file
    
    Returns:
        dict containing 'vocab' and other BPE model information
    """
    with open(filepath, 'rb') as f:
        bpe_model = pickle.load(f)
    
    print(f"Loaded BPE model from {filepath}")
    print(f"Keys in model: {bpe_model.keys()}")
    
    if 'vocab' in bpe_model:
        vocab = bpe_model['vocab']
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Show some sample tokens
        print("\nSample vocabulary entries (first 10):")
        for i, (token, idx) in enumerate(list(vocab.items())[:10]):
            print(f"  {idx}: {repr(token)}")
    
    return bpe_model


# Complete example
if __name__ == "__main__":
    # Example 1: Load BPE model
    print("=== Example 1: Loading BPE model ===")
    bpe_model = load_bpe_model('/mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/BPE/bpe_model_TinyStoriesV2-GPT4-train.pkl')
    vocab_size = len(bpe_model['vocab'])
    
    # Example 3: Load data with memory mapping
    print("\n=== Example 3: Loading with memory mapping ===")
    data = load_data_memmap('sample_tokens.npy', dtype=np.uint16)
    
    # Example 4: Verify the data
    print("\n=== Example 4: Verifying data ===")
    verify_data(data, vocab_size=vocab_size)
    
    # Example 5: Sample a batch
    print("\n=== Example 5: Sampling a batch ===")
    batch_size = 4
    context_length = 8
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    inputs, targets = get_batch(data, batch_size, context_length, device)
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Device: {inputs.device}")
    print(f"\nSample input sequence:\n{inputs[0]}")
    print(f"\nCorresponding targets:\n{targets[0]}")
    
    # Verify that targets are shifted by 1
    print("\n=== Verifying target shift ===")
    idx = np.random.randint(0, len(data) - context_length - 1)
    print(f"Original sequence at index {idx}: {data[idx:idx + context_length + 1]}")
    
    # Sample a single batch to verify the shift
    inputs_check, targets_check = get_batch(data, 1, context_length, 'cpu')
    print(f"Input:  {inputs_check[0]}")
    print(f"Target: {targets_check[0]}")
    print(f"Targets are inputs shifted by 1: {torch.all(inputs_check[0, 1:] == targets_check[0, :-1])}")