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
    """
    Pre-tokenize a list of 'parts' in parallel.
    Special tokens are preserved (not tokenized).
    """
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

    # Flatten results
    return [token for result in results for token in result]


def build_initial_corpus(all_pre_tokens: List[str], special_tokens_set: Set[str]):
    """
    Build initial corpus and compute pair frequencies efficiently.
    Returns: corpus, pair_frequencies
    """
    corpus: List[List[bytes]] = []
    pair_counter = Counter()

    for pre_token in all_pre_tokens:
        if pre_token in special_tokens_set:
            corpus.append([])  # Empty for special tokens
            continue

        token_bytes = pre_token.encode("utf-8")
        seq = [bytes([b]) for b in token_bytes]
        corpus.append(seq)

        # Count pairs in this sequence
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counter[pair] += 1

    return corpus, pair_counter


def merge_pair_in_sequence(token_list: List[bytes], pair: Tuple[bytes, bytes], new_token: bytes) -> List[bytes]:
    """
    Replace all non-overlapping occurrences of pair with new_token (left-to-right).
    """
    if len(token_list) < 2:
        return token_list

    new_list = []
    i = 0
    a, b = pair
    while i < len(token_list):
        if i < len(token_list) - 1 and token_list[i] == a and token_list[i + 1] == b:
            new_list.append(new_token)
            i += 2
        else:
            new_list.append(token_list[i])
            i += 1

    return new_list


def update_pair_frequencies_after_merge(
    corpus: List[List[bytes]], 
    pair_to_merge: Tuple[bytes, bytes],
    new_token: bytes,
    pair_freqs: Counter
) -> None:
    """
    Update corpus in-place by merging pair_to_merge, and update pair_freqs accordingly.
    More efficient approach: update frequencies based on local changes.
    """
    a, b = pair_to_merge
    
    for seq_idx, seq in enumerate(corpus):
        if len(seq) < 2:
            continue
        
        # Check if this sequence contains the pair
        has_pair = False
        for i in range(len(seq) - 1):
            if seq[i] == a and seq[i + 1] == b:
                has_pair = True
                break
        
        if not has_pair:
            continue
        
        # Store old pairs before merge
        old_pairs = []
        for i in range(len(seq) - 1):
            old_pairs.append((seq[i], seq[i + 1]))
        
        # Merge the pair
        new_seq = merge_pair_in_sequence(seq, pair_to_merge, new_token)
        corpus[seq_idx] = new_seq
        
        # Get new pairs
        new_pairs = []
        for i in range(len(new_seq) - 1):
            new_pairs.append((new_seq[i], new_seq[i + 1]))
        
        # Update frequencies: remove old, add new
        for p in old_pairs:
            pair_freqs[p] -= 1
            if pair_freqs[p] <= 0:
                del pair_freqs[p]
        
        for p in new_pairs:
            pair_freqs[p] += 1


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
    Train BPE tokenizer with parallel pre-tokenization.

    Args:
        input_path: Path to text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
        min_frequency: Minimum frequency for merge (default 2)
        verbose: Whether to print progress
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
        print("Pre-tokenization (parallel)...")
        start = time.time()
    all_pre_tokens = parallel_pretokenize(parts, special_tokens, workers=workers)
    if verbose:
        print(f"Pre-tokenization completed in {time.time() - start:.2f}s")
        print(f"Total pre-tokens: {len(all_pre_tokens)}")

    # 4. Build corpus and initial pair frequencies
    if verbose:
        print("Building corpus and counting pairs...")
    special_tokens_set = set(special_tokens)
    corpus, pair_freqs = build_initial_corpus(all_pre_tokens, special_tokens_set)

    # 5. Initialize vocabulary
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_id = 256
    
    for st in special_tokens:
      vocab[current_id] = st.encode("utf-8")
      current_id += 1

    merges: List[Tuple[bytes, bytes]] = []

    if verbose:
      print(f"Initial vocab size: {len(vocab)}, target: {vocab_size}")
      print("Starting BPE merges...")
    
    num_merges = vocab_size - len(vocab)
    pbar = tqdm(total=num_merges, desc="BPE merges", disable=not verbose)
    
    # Main BPE loop
    while len(vocab) < vocab_size and pair_freqs:
        # Find best pair
        best_pair, best_freq = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))

        if best_freq < min_frequency:
            if verbose:
                print(f"Best pair has frequency {best_freq} < {min_frequency}. Stopping.")
            break

        # Create new token
        new_token = best_pair[0] + best_pair[1]

        # Add to vocab and merges
        vocab[current_id] = new_token
        merges.append(best_pair)

        # Update corpus and pair frequencies
        update_pair_frequencies_after_merge(corpus, best_pair, new_token, pair_freqs)

        # Update progress bar with latest merge info
        pbar.set_postfix({
            'freq': best_freq,
            'vocab': len(vocab)
        })
        pbar.update(1)

        current_id += 1
        
    pbar.close()

    if verbose:
        print(f"✓ Training complete: {len(vocab)} tokens, {len(merges)} merges")
        print(f"Final vocab size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")

    # Save if requested
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({'vocab': vocab, 'merges': merges, 'special_tokens': special_tokens}, f)
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

def decode(token_ids: List[int], vocab: Dict[int, bytes], special_tokens: Optional[List[str]] = None) -> str:
    """
    Decode token IDs back to text using the vocabulary.
    
    Args:
        token_ids: List of token IDs to decode
        vocab: Vocabulary mapping {id: bytes}
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
    
    Returns:
        Decoded text string
    """
    if special_tokens is None:
        special_tokens = []
    
    # Collect all byte sequences
    byte_sequences = []
    for token_id in token_ids:
        if token_id in vocab:
            byte_sequences.append(vocab[token_id])
        else:
            # Handle unknown token ID gracefully
            byte_sequences.append(b'')
    
    # Concatenate all bytes
    all_bytes = b''.join(byte_sequences)
    
    # Decode to string, handling potential decoding errors
    try:
        text = all_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = all_bytes.decode("utf-8", errors="ignore")
    
    return text
 
# Example usage
if __name__ == "__main__":
    start_time = time.time()
    
    # Example with small vocab size for testing
    data = "/mnt/d/Stanford_LLM/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = run_train_bpe(
        input_path=data, 
        vocab_size=500, 
        special_tokens=["<|endoftext|>"], 
        verbose=True,
        workers=None,  # Auto-detect
        save_path="bpe_model.pkl"
    )
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f}s")     
    
    test_text = "One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun. Then, something unexpected happened."
    token_ids = encode(test_text, merges, special_tokens=["<|endoftext|>"])
    print(f"\nEncoded '{test_text}' to {len(token_ids)} tokens")
    print(f"Token IDs: {token_ids}")
    
    decode_text = decode(token_ids, vocab, special_tokens=["<|endoftext|>"])
    print(f"Decode text: {decode_text}")                                                                         