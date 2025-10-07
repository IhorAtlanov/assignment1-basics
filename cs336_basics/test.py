import regex as re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional
from multiprocessing import Pool, cpu_count
import time
import pickle
from tqdm import tqdm

# Pattern for pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_chunk(args):
    """Helper for multiprocessing.Pool.map (must be top-level)."""
    chunk, special_tokens_set = args
    out_tokens = []
    for part in chunk:
        if part in special_tokens_set:
            out_tokens.append(part)
        elif part:
            tokens = re.findall(PAT, part)
            out_tokens.extend(tokens)
    return out_tokens


def parallel_pretokenize(parts: List[str], special_tokens: List[str], workers: Optional[int] = None) -> List[str]:
    """Pre-tokenize a list of 'parts' in parallel. Special tokens are preserved."""
    if workers is None:
        workers = max(1, cpu_count() - 1)

    if len(parts) == 0:
        return []

    # Sequential processing for small inputs
    if workers <= 1 or len(parts) < workers * 2:
        result = []
        sset = set(special_tokens)
        for part in parts:
            if part in sset:
                result.append(part)
            elif part:
                result.extend(re.findall(PAT, part))
        return result

    # Create chunks
    chunk_size = max(1, (len(parts) + workers - 1) // workers)
    chunks = [parts[i:i + chunk_size] for i in range(0, len(parts), chunk_size)]
    args_list = [(chunk, set(special_tokens)) for chunk in chunks]

    with Pool(processes=min(workers, len(chunks))) as pool:
        results = pool.map(_pretokenize_chunk, args_list)

    return [token for result in results for token in result]


def build_initial_corpus(all_pre_tokens: List[str], special_tokens_set: Set[str]):
    """
    Build initial corpus and compute pair frequencies efficiently.
    Returns: corpus, pair_frequencies, non_empty_indices
    """
    corpus: List[List[bytes]] = []
    pair_counter = Counter()
    non_empty_indices = []  # Track which sequences are non-empty

    for idx, pre_token in enumerate(all_pre_tokens):
        if pre_token in special_tokens_set:
            corpus.append([])  # Empty for special tokens
            continue

        token_bytes = pre_token.encode("utf-8")
        seq = [bytes([b]) for b in token_bytes]
        corpus.append(seq)
        
        if len(seq) > 1:
            non_empty_indices.append(idx)
            # Count pairs in this sequence
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counter[pair] += 1

    return corpus, pair_counter, non_empty_indices


def merge_and_update_optimized(
    corpus: List[List[bytes]], 
    non_empty_indices: List[int],
    pair_to_merge: Tuple[bytes, bytes],
    new_token: bytes,
    pair_freqs: Counter
) -> List[int]:
    """
    Optimized merge: only process sequences that might contain the pair.
    Returns updated non_empty_indices.
    """
    a, b = pair_to_merge
    new_non_empty = []
    
    # Only check sequences that have more than 1 token
    for seq_idx in non_empty_indices:
        seq = corpus[seq_idx]
        
        if len(seq) < 2:
            continue
        
        # Quick check: does this sequence contain the pair?
        # This is O(n) but with early exit
        has_pair = any(seq[i] == a and seq[i + 1] == b for i in range(len(seq) - 1))
        
        if not has_pair:
            if len(seq) > 1:
                new_non_empty.append(seq_idx)
            continue
        
        # Compute old pairs efficiently using a single pass
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_freqs[pair] -= 1
            if pair_freqs[pair] == 0:
                del pair_freqs[pair]
        
        # Merge the pair in-place style
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        
        corpus[seq_idx] = new_seq
        
        # Add new pairs
        for i in range(len(new_seq) - 1):
            pair = (new_seq[i], new_seq[i + 1])
            pair_freqs[pair] += 1
        
        if len(new_seq) > 1:
            new_non_empty.append(seq_idx)
    
    return new_non_empty


def run_train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2, 
    verbose: bool = False, 
    workers: Optional[int] = None,
    save_path: Optional[str] = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer with optimized parallel pre-tokenization.

    Args:
        input_path: Path to text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
        min_frequency: Minimum frequency for merge (default 2)
        verbose: Whether to print detailed progress
        workers: Number of processes for pre-tokenization (None = auto)
        save_path: Optional path to save vocab and merges
    Returns:
        (vocab, merges) - vocabulary {id: bytes} and list of merge operations
    """
    if special_tokens is None:
        special_tokens = []

    # 1. Read text
    if verbose:
        print(f"Reading file {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Split on special tokens
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]

    # 3. Pre-tokenize (parallel)
    if verbose:
        print("Pre-tokenizing...")
    all_pre_tokens = parallel_pretokenize(parts, special_tokens, workers=workers)
    if verbose:
        print(f"Generated {len(all_pre_tokens):,} pre-tokens")

    # 4. Build corpus and initial pair frequencies
    if verbose:
        print("Building corpus...")
    special_tokens_set = set(special_tokens)
    corpus, pair_freqs, non_empty_indices = build_initial_corpus(all_pre_tokens, special_tokens_set)

    # 5. Initialize vocabulary
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_id = 256
    
    for st in special_tokens:
        vocab[current_id] = st.encode("utf-8")
        current_id += 1

    merges: List[Tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    # Main BPE loop with tqdm
    pbar = tqdm(total=num_merges, desc="BPE merges", unit="merge")
    
    try:
        while len(vocab) < vocab_size and pair_freqs:
            # Find best pair
            best_pair, best_freq = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))

            if best_freq < min_frequency:
                if verbose:
                    print(f"\nStopping: best pair frequency {best_freq} < {min_frequency}")
                break

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            vocab[current_id] = new_token
            merges.append(best_pair)

            # Update corpus and pair frequencies (optimized)
            non_empty_indices = merge_and_update_optimized(
                corpus, non_empty_indices, best_pair, new_token, pair_freqs
            )

            # Update progress bar
            pbar.update(1)
            if verbose and len(vocab) % 500 == 0:
                pbar.set_postfix({
                    'freq': best_freq, 
                    'active_seqs': len(non_empty_indices),
                    'unique_pairs': len(pair_freqs)
                })

            current_id += 1
    finally:
        pbar.close()

    if verbose:
        print(f"Final vocab size: {len(vocab)}, merges: {len(merges)}")

    # Save if requested
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'vocab': vocab, 
                'merges': merges, 
                'special_tokens': special_tokens
            }, f)
        if verbose:
            print(f"Saved to {save_path}")

    return vocab, merges


def load_bpe(load_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]], List[str]]:
    """Load saved BPE model."""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    return data['vocab'], data['merges'], data['special_tokens']


def encode(text: str, merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None) -> List[int]:
    """
    Encode text using trained BPE merges.
    Returns list of token IDs.
    """
    if special_tokens is None:
        special_tokens = []
    
    # Build merge priority dict
    merge_priority = {pair: i for i, pair in enumerate(merges)}
    
    # Split on special tokens
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]
    
    # Pre-tokenize
    special_set = set(special_tokens)
    pre_tokens = []
    for part in parts:
        if part in special_set:
            pre_tokens.append(part)
        elif part:
            pre_tokens.extend(re.findall(PAT, part))
    
    # Apply merges to each pre-token
    result = []
    for pre_token in pre_tokens:
        if pre_token in special_set:
            # Map special token to its ID (256 + index in special_tokens list)
            result.append(256 + special_tokens.index(pre_token))
            continue
        
        # Convert to byte tokens
        tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
        
        # Apply merges
        while len(tokens) > 1:
            # Find the pair with highest priority (lowest merge index)
            best_pair = None
            best_idx = None
            best_priority = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in merge_priority and merge_priority[pair] < best_priority:
                    best_pair = pair
                    best_idx = i
                    best_priority = merge_priority[pair]
            
            if best_pair is None:
                break
            
            # Merge the best pair
            tokens = tokens[:best_idx] + [best_pair[0] + best_pair[1]] + tokens[best_idx + 2:]
        
        # Convert bytes to IDs
        for token in tokens:
            # For single bytes, ID is the byte value
            if len(token) == 1:
                result.append(token[0])
            else:
                # For merged tokens, find in merges list
                # ID = 256 + len(special_tokens) + merge_index
                for i, (a, b) in enumerate(merges):
                    if a + b == token:
                        result.append(256 + len(special_tokens) + i)
                        break
    
    return result


# Example usage
if __name__ == "__main__":
    start_time = time.time()
    
    data = "/mnt/d/Stanford_LLM/assignment1-basics/data/owt_valid.txt"
    vocab, merges = run_train_bpe(
        input_path=data, 
        vocab_size=10000, 
        special_tokens=["<|endoftext|>"], 
        verbose=True,
        workers=None,
        save_path="bpe_model.pkl"
    )
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f}s")
    
    # Test encoding
    test_text = "Hello, world! This is a test."
    token_ids = encode(test_text, merges, special_tokens=["<|endoftext|>"])
    print(f"\nEncoded '{test_text}' to {len(token_ids)} tokens")
    print(f"Token IDs: {token_ids}")