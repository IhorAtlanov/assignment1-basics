import regex as re  # Використовуємо regex замість re для підтримки \p{L} та \p{N}
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from multiprocessing import Pool, cpu_count
import time

# Шаблон для pre-tokenization regex
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


def parallel_pretokenize(parts: List[str], special_tokens: List[str], workers: int = None) -> List[str]:
    """
    Pre-tokenize a list of 'parts' where parts already contains special tokens as separate elements.
    We chunk `parts` into `workers` nearly-equal chunks and run pre-tokenization in parallel.
    Special tokens are preserved (not tokenized).
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)

    if len(parts) == 0:
        return []

    # If only one worker or small number of parts, do sequentially (avoids multiprocessing overhead)
    if workers <= 1 or len(parts) < workers:
        result = []
        sset = set(special_tokens)
        for part in parts:
            if part in sset:
                result.append(part)
            elif part:
                result.extend(re.findall(PAT, part))
        return result

    # Create approximately equal chunks of parts
    chunk_size = (len(parts) + workers - 1) // workers
    chunks = [parts[i:i + chunk_size] for i in range(0, len(parts), chunk_size)]

    args_list = [(chunk, set(special_tokens)) for chunk in chunks]

    with Pool(processes=min(workers, len(chunks))) as pool:
        results = pool.map(_pretokenize_chunk, args_list)

    # Flatten
    flattened = []
    for r in results:
        flattened.extend(r)
    return flattened


def get_pair_frequencies_from_positions(pair_positions: Dict[Tuple[bytes, bytes], Set[Tuple[int, int]]]) -> Counter:
    """Compute frequencies from pair_positions mapping."""
    return Counter({pair: len(posset) for pair, posset in pair_positions.items()})


def build_initial_corpus_and_pair_index(all_pre_tokens: List[str], special_tokens_set: Set[str]):
    """
    Build initial corpus (list of byte-token lists) and pair index structures:
      - corpus: List[List[bytes]]
      - seq_pair_positions: List[Dict[pair, set(positions_in_seq)]]
      - pair_positions: Dict[pair, set((seq_idx, pos_in_seq))]
    """
    corpus: List[List[bytes]] = []
    seq_pair_positions: List[Dict[Tuple[bytes, bytes], Set[int]]] = []
    pair_positions: Dict[Tuple[bytes, bytes], Set[Tuple[int, int]]] = defaultdict(set)

    for seq_idx, pre_token in enumerate(all_pre_tokens):
        if pre_token in special_tokens_set:
            # We skip adding special tokens to corpus (they are in vocab separately)
            # Represent them as an empty sequence (or we could skip entirely)
            corpus.append([])
            seq_pair_positions.append({})
            continue

        token_bytes = pre_token.encode("utf-8")
        seq = [bytes([b]) for b in token_bytes]
        corpus.append(seq)

        seq_pairs: Dict[Tuple[bytes, bytes], Set[int]] = defaultdict(set)
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            seq_pairs[pair].add(i)
            pair_positions[pair].add((seq_idx, i))
        seq_pair_positions.append(seq_pairs)

    return corpus, seq_pair_positions, pair_positions


def merge_pair_in_sequence(token_list: List[bytes], pair: Tuple[bytes, bytes], new_token: bytes) -> List[bytes]:
    """Замінити всі входження пари на новий токен в одному списку токенів (non-overlapping, left-to-right)."""
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


def scan_sequence_pairs(seq: List[bytes]) -> Dict[Tuple[bytes, bytes], Set[int]]:
    """Return dict mapping pair -> set(positions) for given sequence."""
    d: Dict[Tuple[bytes, bytes], Set[int]] = defaultdict(set)
    for i in range(len(seq) - 1):
        p = (seq[i], seq[i + 1])
        d[p].add(i)
    return d


def run_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str],
                  min_frequency: int = 2, verbose: bool = False, workers: int = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Тренування BPE токенізатора з паралельною pre-tokenization та інкрементним оновленням частот пар.

    Args:
        input_path: шлях до текстового файлу
        vocab_size: цільовий розмір словника
        special_tokens: список спеціальних токенів (наприклад, ["<|endoftext|>"])
        min_frequency: мінімальна частота для merge (за замовчуванням 2)
        verbose: чи виводити прогрес
        workers: кількість процесів для pre-tokenization (None -> autodetect)
    Returns:
        (vocab, merges) - словник {id: bytes} та список merge операцій
    """

    # 1. Зчитати весь текст
    if verbose:
        print(f"Читаємо файл {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Split on special tokens (they will appear as standalone parts)
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]

    # 3. Pre-tokenize each fragment (parallel)
    if verbose:
        print("Pre-tokenization (parallel) ...")
    all_pre_tokens = parallel_pretokenize(parts, special_tokens, workers=workers)

    if verbose:
        print(f"Total pre-tokens: {len(all_pre_tokens)}")

    # 4. Build corpus and pair indices (skipping special tokens sequences)
    special_tokens_set = set(special_tokens)
    corpus, seq_pair_positions, pair_positions = build_initial_corpus_and_pair_index(all_pre_tokens, special_tokens_set)

    # 5. Initialize vocab with single bytes and special tokens
    vocab: Dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    current_id = 256
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        vocab[current_id] = st_bytes
        current_id += 1

    merges: List[Tuple[bytes, bytes]] = []

    if verbose:
        print(f"Початковий розмір vocab: {len(vocab)}, ціль: {vocab_size}")
        print("Початок BPE merges...")

    iteration = 0
    # Compute initial pair frequencies
    pair_freqs = get_pair_frequencies_from_positions(pair_positions)

    while len(vocab) < vocab_size and pair_freqs:
        # Pick best pair (highest frequency, tie-breaker by pair)
        best_pair, best_freq = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))

        if best_freq < min_frequency:
            if verbose:
                print(f"Найкраща пара має частоту {best_freq} < {min_frequency}. Зупиняємо.")
            break

        # New token value
        new_token = best_pair[0] + best_pair[1]

        # Add to vocab and merges
        vocab[current_id] = new_token
        merges.append(best_pair)
        if verbose:
            print(f"Iteration {iteration}: vocab_size={len(vocab)}, merged {best_pair[0]!r} + {best_pair[1]!r} -> {new_token!r} (freq={best_freq})")

        # Find affected sequences
        occurrences = pair_positions.get(best_pair, set())
        # Group by sequence index
        seqs_to_update = defaultdict(set)
        for seq_idx, pos in occurrences:
            seqs_to_update[seq_idx].add(pos)

        # Track which pairs changed (for updating pair_positions & pair_freqs)
        changed_pairs = set()

        # For each affected sequence: remove its old pair entries, rebuild sequence, rescan pairs, update maps
        for seq_idx in seqs_to_update.keys():
            old_seq_pairs = seq_pair_positions[seq_idx]  # dict pair -> set(positions)
            # Remove all old pairs from global pair_positions that belong to this sequence
            for pair, posset in old_seq_pairs.items():
                for pos in posset:
                    if (seq_idx, pos) in pair_positions.get(pair, set()):
                        pair_positions[pair].discard((seq_idx, pos))
                        changed_pairs.add(pair)
                        if not pair_positions[pair]:
                            del pair_positions[pair]

            # Merge the best_pair occurrences inside this sequence
            seq = corpus[seq_idx]
            if not seq:
                # nothing to do (empty sequence; e.g., was a special token)
                seq_pair_positions[seq_idx] = {}
                continue

            new_seq = merge_pair_in_sequence(seq, best_pair, new_token)
            corpus[seq_idx] = new_seq

            # Rescan pairs for this sequence
            new_pairs = scan_sequence_pairs(new_seq)
            seq_pair_positions[seq_idx] = new_pairs

            # Add new pairs to global pair_positions
            for pair, posset in new_pairs.items():
                for pos in posset:
                    pair_positions[pair].add((seq_idx, pos))
                changed_pairs.add(pair)

        # Recompute frequencies only for changed pairs
        for pair in changed_pairs:
            pair_freqs[pair] = len(pair_positions.get(pair, ()))
            if pair_freqs[pair] == 0:
                del pair_freqs[pair]

        # Remove the merged pair from pair_freqs (it may persist if overlapping merges created new ones, but usually becomes 0)
        if best_pair in pair_freqs:
            del pair_freqs[best_pair]

        current_id += 1
        iteration += 1

    if verbose:
        print("\nТренування завершено!")
        print(f"Фінальний розмір vocab: {len(vocab)}")
        print(f"Кількість merges: {len(merges)}")

    return vocab, merges


# Приклад використання
if __name__ == "__main__":
    start_time = time.time()
    data = "/mnt/d/Stanford_LLM/assignment1-basics/data/owt_valid.txt"
    run_train_bpe(input_path=data, vocab_size=10000, special_tokens=["<|endoftext|>"], verbose=True)
    end_time = time.time()
    print(f"TIME_SPEND = {end_time-start_time}")
