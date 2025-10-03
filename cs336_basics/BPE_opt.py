import regex as re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from multiprocessing import Pool, cpu_count
import heapq
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_initial_pair_frequencies(corpus: List[List[bytes]]) -> Counter:
    """Підрахувати частоти всіх суміжних пар токенів у корпусі (тільки для ініціалізації)."""
    pair_freqs = Counter()
    for token_list in corpus:
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            pair_freqs[pair] += 1
    return pair_freqs

def merge_pair_and_update_frequencies(
    corpus: List[List[bytes]], 
    pair: Tuple[bytes, bytes], 
    new_token: bytes,
    pair_freqs: Dict[Tuple[bytes, bytes], int],
    heap: List,
    heap_entry: Dict
) -> None:
    """
    Замінити пару в корпусі та інкрементально оновити частоти пар з heap.
    Модифікує corpus, pair_freqs, та heap in-place для ефективності.
    """
    changed_pairs = set()
    
    for seq_idx in range(len(corpus)):
        token_list = corpus[seq_idx]
        if len(token_list) < 2:
            continue
        
        new_list = []
        i = 0
        
        while i < len(token_list):
            # Якщо знайшли пару для злиття
            if i < len(token_list) - 1 and token_list[i] == pair[0] and token_list[i + 1] == pair[1]:
                # Оновити частоти для пар, які зникають
                # Пара зліва від merged pair
                if new_list:  # Є токен зліва
                    left_pair = (new_list[-1], pair[0])
                    if left_pair in pair_freqs:
                        pair_freqs[left_pair] -= 1
                        changed_pairs.add(left_pair)
                        if pair_freqs[left_pair] <= 0:
                            del pair_freqs[left_pair]
                
                # Пара справа від merged pair
                if i + 2 < len(token_list):
                    right_pair = (pair[1], token_list[i + 2])
                    if right_pair in pair_freqs:
                        pair_freqs[right_pair] -= 1
                        changed_pairs.add(right_pair)
                        if pair_freqs[right_pair] <= 0:
                            del pair_freqs[right_pair]
                
                # Додати новий токен
                new_list.append(new_token)
                
                # Оновити частоти для нових пар
                # Нова пара зліва
                if len(new_list) > 1:
                    new_left_pair = (new_list[-2], new_token)
                    pair_freqs[new_left_pair] = pair_freqs.get(new_left_pair, 0) + 1
                    changed_pairs.add(new_left_pair)
                
                # Нова пара справа
                if i + 2 < len(token_list):
                    new_right_pair = (new_token, token_list[i + 2])
                    pair_freqs[new_right_pair] = pair_freqs.get(new_right_pair, 0) + 1
                    changed_pairs.add(new_right_pair)
                
                i += 2
            else:
                new_list.append(token_list[i])
                i += 1
        
        corpus[seq_idx] = new_list
    
    # Видалити саму merged pair з частот
    if pair in pair_freqs:
        del pair_freqs[pair]
        changed_pairs.add(pair)
    
    # Оновити heap для змінених пар
    for changed_pair in changed_pairs:
        # Позначити старий запис як недійсний
        if changed_pair in heap_entry:
            heap_entry[changed_pair][-1] = False  # Mark as invalid
        
        # Додати новий запис, якщо пара ще існує
        if changed_pair in pair_freqs:
            freq = pair_freqs[changed_pair]
            entry = [-freq, changed_pair, True]  # negative for max heap, valid flag
            heap_entry[changed_pair] = entry
            heapq.heappush(heap, entry)

def get_chunk_boundaries(text: str, special_tokens: List[str], num_chunks: int) -> List[int]:
    """
    Get chunk boundaries that align with special token positions.
    Returns list of indices where chunks should start.
    """
    if not special_tokens:
        # If no special tokens, divide text evenly
        chunk_size = len(text) // num_chunks
        return [i * chunk_size for i in range(num_chunks)] + [len(text)]
    
    # Find all special token positions
    pattern = "|".join(re.escape(st) for st in special_tokens)
    special_token_positions = [0]
    
    for match in re.finditer(pattern, text):
        special_token_positions.append(match.start())
    
    special_token_positions.append(len(text))
    
    # If we have fewer special tokens than desired chunks, use what we have
    if len(special_token_positions) <= num_chunks + 1:
        return special_token_positions
    
    # Otherwise, select evenly spaced positions
    step = len(special_token_positions) // (num_chunks + 1)
    boundaries = [special_token_positions[i * step] for i in range(num_chunks)]
    boundaries.append(len(text))
    
    return boundaries

def process_chunk(args):
    """
    Process a single chunk of text for pre-tokenization.
    This function will be run in parallel.
    """
    text_chunk, special_tokens = args
    special_tokens_set = set(special_tokens)
    
    # Split by special tokens
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({pattern})", text_chunk)
    else:
        parts = [text_chunk]
    
    # Pre-tokenize each part
    chunk_pre_tokens = []
    for part in parts:
        if part in special_tokens_set:
            chunk_pre_tokens.append(part)
        elif part:  # ignore empty strings
            tokens = re.findall(PAT, part)
            chunk_pre_tokens.extend(tokens)
    
    # Convert to bytes representation
    corpus_chunk = []
    for pre_token in chunk_pre_tokens:
        if pre_token not in special_tokens_set:
            token_as_bytes = pre_token.encode("utf-8")
            corpus_chunk.append([bytes([b]) for b in token_as_bytes])
    
    return corpus_chunk

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], 
                  min_frequency: int = 2, verbose: bool = False, num_processes: int = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Тренування BPE токенізатора з паралелізацією та оптимізацією merging.
    
    Args:
        input_path: шлях до текстового файлу
        vocab_size: цільовий розмір словника
        special_tokens: список спеціальних токенів (наприклад, ["<|endoftext|>"])
        min_frequency: мінімальна частота для merge (за замовчуванням 2)
        verbose: чи виводити прогрес
        num_processes: кількість процесів (за замовчуванням - кількість CPU)
    
    Returns:
        (vocab, merges) - словник {id: bytes} та список merge операцій
    """
    
    if num_processes is None:
        num_processes = cpu_count()
    
    # 1. Зчитати весь текст
    if verbose:
        print(f"Читаємо файл {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 2. Get chunk boundaries aligned with special tokens
    if verbose:
        print(f"Розділяємо текст на {num_processes} чанків...")
    
    boundaries = get_chunk_boundaries(text, special_tokens, num_processes)
    
    # Create chunks
    chunks = []
    for i in range(len(boundaries) - 1):
        chunk = text[boundaries[i]:boundaries[i + 1]]
        chunks.append((chunk, special_tokens))
    
    # 3. Parallelize pre-tokenization
    if verbose:
        print(f"Паралельна pre-tokenization з {num_processes} процесами...")
    
    with Pool(processes=num_processes) as pool:
        corpus_chunks = pool.map(process_chunk, chunks)
    
    # Flatten the results
    corpus = []
    for chunk in corpus_chunks:
        corpus.extend(chunk)
    
    if verbose:
        print(f"Pre-tokenization завершено. Створено {len(corpus)} токенів.")
    
    # 4. Ініціалізувати словник з одиночних байтів (0-255)
    vocab: Dict[int, bytes] = {}
    
    # Додати всі можливі байти
    for i in range(256):
        vocab[i] = bytes([i])
    
    current_id = 256
    
    # Додати спеціальні токени до словника
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        vocab[current_id] = st_bytes
        current_id += 1
    
    # 5. Ініціалізувати частоти пар та heap (тільки один раз!)
    if verbose:
        print("Ініціалізація частот пар та heap...")
    
    pair_freqs = {}
    for token_list in corpus:
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + 1
    
    # Створити max heap (використовуємо негативні частоти для max heap)
    heap = []
    heap_entry = {}  # Мапа pair -> heap entry для оновлень
    
    for pair, freq in pair_freqs.items():
        entry = [-freq, pair, True]  # [priority, pair, valid_flag]
        heap_entry[pair] = entry
        heapq.heappush(heap, entry)
    
    # 6. Ітеративно виконувати BPE merges з heap-based selection
    merges: List[Tuple[bytes, bytes]] = []
    
    if verbose:
        print(f"Початковий розмір vocab: {len(vocab)}, ціль: {vocab_size}")
        print(f"Початкова кількість унікальних пар: {len(pair_freqs)}")
        print("Початок BPE merges з heap оптимізацією...")
    
    iteration = 0
    while len(vocab) < vocab_size:
        # Знайти найчастішу пару з heap (O(log n))
        while heap:
            neg_freq, best_pair, valid = heapq.heappop(heap)
            if valid:  # Перевірити, чи запис ще дійсний
                best_freq = -neg_freq
                break
        else:
            # Heap порожній
            if verbose:
                print("Більше немає пар для об'єднання.")
            break
        
        # Перевірка мінімальної частоти
        if best_freq < min_frequency:
            if verbose:
                print(f"Найкраща пара має частоту {best_freq} < {min_frequency}. Зупиняємо.")
            break
        
        # Створити новий токен
        new_token = best_pair[0] + best_pair[1]
        
        # Додати до словника
        vocab[current_id] = new_token
        merges.append(best_pair)
        
        if verbose and iteration % 50 == 0:
            print(f"Iteration {iteration}: vocab_size={len(vocab)}, "
                  f"merged {best_pair[0]!r} + {best_pair[1]!r} -> {new_token!r} (freq={best_freq}), "
                  f"unique_pairs={len(pair_freqs)}")
        
        # Замінити пару в корпусі та інкрементально оновити частоти та heap
        merge_pair_and_update_frequencies(corpus, best_pair, new_token, pair_freqs, heap, heap_entry)
        
        current_id += 1
        iteration += 1
    
    if verbose:
        print(f"\nТренування завершено!")
        print(f"Фінальний розмір vocab: {len(vocab)}")
        print(f"Кількість merges: {len(merges)}")
    
    return vocab, merges


# Приклад використання
if __name__ == "__main__":
    # Тестовий приклад
    start_time = time.time()
    data = "/mnt/c/Users/Ihor/Desktop/Stanford_LLM/assignment1-basics/data/corpus.en"
    run_train_bpe(input_path=data, vocab_size=500, special_tokens=["<|endoftext|>"], verbose=True)
    end_time = time.time()
    
    print(f"Time: {end_time-start_time}")
    #print(f"Time: {(end_time-start_time)/60}")