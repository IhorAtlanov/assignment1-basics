import os
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


def get_word_freqs(input_path: str, special_tokens: List[str], workers: Optional[int], chunk_size_mb: int = 10) -> Counter:
    """
    Reads a large file in chunks, pre-tokenizes, and returns word frequency counts.
    This is the first pass over the data.
    """
    special_tokens_set = set(special_tokens)
    word_freqs = Counter()
    
    file_size = os.path.getsize(input_path)
    
    with open(input_path, "r", encoding="utf-8") as f, tqdm(total=file_size, unit='B', unit_scale=True, desc="1/2: Counting word frequencies") as pbar:
        while True:
            chunk = f.readlines(chunk_size_mb * 1024 * 1024)
            if not chunk:
                break
            
            chunk_text = "".join(chunk)
            
            # Split on special tokens
            if special_tokens:
                pattern = "|".join(re.escape(st) for st in special_tokens)
                parts = re.split(f"({pattern})", chunk_text)
            else:
                parts = [chunk_text]
            
            # Pre-tokenize the chunk
            pre_tokens = parallel_pretokenize(parts, special_tokens, workers=workers)
            
            # Update word frequencies
            word_freqs.update(pre_tokens)
            
            pbar.update(len(chunk_text.encode('utf-8')))
            
    # Remove special tokens from word_freqs as they are handled separately
    for st in special_tokens_set:
        if st in word_freqs:
            del word_freqs[st]
            
    return word_freqs

def get_initial_pair_freqs(word_freqs: Counter) -> Tuple[Dict[Tuple[bytes, ...], int], Counter]:
    """
    Calculates initial pair frequencies from word frequency counts.
    The 'corpus' is now a dictionary mapping tokenized words to their frequency.
    """
    corpus = {}
    pair_freqs = Counter()
    
    for word, freq in word_freqs.items():
        tokens = tuple(bytes([b]) for b in word.encode("utf-8"))
        corpus[tokens] = freq
        
        for i in range(len(tokens) - 1):
            pair_freqs[(tokens[i], tokens[i+1])] += freq
            
    return corpus, pair_freqs

def merge_pair(tokens: Tuple[bytes, ...], pair: Tuple[bytes, bytes], new_token: bytes) -> Tuple[bytes, ...]:
    """
    Helper to merge a pair within a single tokenized word (represented as a tuple).
    """
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2,
    verbose: bool = False,
    workers: Optional[int] = None,
    save_path: Optional[str] = None,
    chunk_size_mb: int = 10
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer efficiently on large files by streaming.
    """
    if special_tokens is None: special_tokens = []
    
    # 1. First pass: Get word frequencies by reading file in chunks
    if verbose: print("Starting Pass 1: Reading file and counting word frequencies...")
    word_freqs = get_word_freqs(input_path, special_tokens, workers, chunk_size_mb)
    if verbose: print(f"✓ Pass 1 complete. Found {len(word_freqs)} unique words.")

    # 2. Initialize our vocabulary representation and initial pair frequencies
    if verbose: print("Building initial vocabulary and pair frequencies...")
    corpus, pair_freqs = get_initial_pair_freqs(word_freqs)

    # 3. Initialize vocabulary with base tokens and special tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_id = 256
    for st in special_tokens:
        vocab[current_id] = st.encode("utf-8")
        current_id += 1

    merges: List[Tuple[bytes, bytes]] = []

    # 4. Main BPE merge loop
    if verbose:
        print(f"Initial vocab size: {len(vocab)}, target: {vocab_size}")
        print("Starting Pass 2: BPE merges...")
    
    num_merges = vocab_size - len(vocab)
    pbar = tqdm(total=num_merges, desc="2/2: BPE merges", disable=not verbose)

    while len(vocab) < vocab_size and pair_freqs:
        # Find the best pair to merge
        try:
            best_pair, best_freq = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))
        except ValueError:
            break # No pairs left

        if best_freq < min_frequency:
            if verbose: print(f"Best pair frequency {best_freq} < {min_frequency}. Stopping.")
            break

        # Add the new merged token to our vocabulary
        new_token = best_pair[0] + best_pair[1]
        vocab[current_id] = new_token
        merges.append(best_pair)
        
        # This is the new, efficient update step. We don't touch the original file.
        # We update our in-memory 'corpus' (word frequencies) and pair counts.
        new_corpus = {}
        for word_tokens, freq in corpus.items():
            if len(word_tokens) < 2:
                new_corpus[word_tokens] = freq
                continue

            # Find pairs in this word that will be affected by the merge
            merged_word_tokens = merge_pair(word_tokens, best_pair, new_token)
            
            if merged_word_tokens != word_tokens:
                # The word was changed, so we need to update pair frequencies
                # Decrement counts for old pairs that were destroyed by the merge
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i+1])
                    if pair_freqs.get(pair): pair_freqs[pair] -= freq
                
                # Increment counts for new pairs created by the merge
                for i in range(len(merged_word_tokens) - 1):
                    pair = (merged_word_tokens[i], merged_word_tokens[i+1])
                    pair_freqs[pair] += freq
            
            new_corpus[merged_word_tokens] = freq

        corpus = new_corpus
        current_id += 1
        pbar.update(1)
        pbar.set_postfix({'freq': best_freq, 'vocab': len(vocab)})

    pbar.close()

    if verbose:
        print(f"✓ Training complete: {len(vocab)} tokens, {len(merges)} merges")

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

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    
    # Example with small vocab size for testing
    data = "/mnt/d/Stanford_LLM/assignment1-basics/data/owt_valid.txt"
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