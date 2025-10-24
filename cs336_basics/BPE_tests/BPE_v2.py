import regex as re  # Використовуємо regex замість re для підтримки \p{L} та \p{N}
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from multiprocessing import Process

# Шаблон для pre-tokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_pair_frequencies(corpus: List[List[bytes]]) -> Counter:
    """Підрахувати частоти всіх суміжних пар токенів у корпусі."""
    pair_freqs = Counter()
    for token_list in corpus:
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            pair_freqs[pair] += 1
    return pair_freqs

def merge_pair_in_sequence(token_list: List[bytes], pair: Tuple[bytes, bytes], new_token: bytes) -> List[bytes]:
    """Замінити всі входження пари на новий токен в одному списку токенів."""
    if len(token_list) < 2:
        return token_list
    
    new_list = []
    i = 0
    while i < len(token_list):
        # Якщо знайшли пару - замінюємо
        if i < len(token_list) - 1 and token_list[i] == pair[0] and token_list[i + 1] == pair[1]:
            new_list.append(new_token)
            i += 2
        else:
            new_list.append(token_list[i])
            i += 1
    
    return new_list

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], 
                  min_frequency: int = 2, verbose: bool = False) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Тренування BPE токенізатора.
    
    Args:
        input_path: шлях до текстового файлу
        vocab_size: цільовий розмір словника
        special_tokens: список спеціальних токенів (наприклад, ["<|endoftext|>"])
        min_frequency: мінімальна частота для merge (за замовчуванням 2)
        verbose: чи виводити прогрес
    
    Returns:
        (vocab, merges) - словник {id: bytes} та список merge операцій
    """
    
    # 1. Зчитати весь текст
    if verbose:
        print(f"Читаємо файл {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 2. Розділити текст на частини за спеціальними токенами
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]
    
    # 3. Pre-tokenize кожний фрагмент
    if verbose:
        print("Pre-tokenization...")
    all_pre_tokens: List[str] = []
    for part in parts:
        if part in special_tokens:
            all_pre_tokens.append(part)
        elif part:  # ігноруємо порожні рядки
            tokens = re.findall(PAT, part)
            all_pre_tokens.extend(tokens)
    
    # 4. Перетворити кожен pre-token у список байтів
    if verbose:
        print(f"Конвертуємо {len(all_pre_tokens)} pre-tokens у байти...")
    
    # Створюємо corpus як список списків байтів
    corpus: List[List[bytes]] = []
    special_tokens_set = set(special_tokens) # Use a set for efficient lookups
    for pre_token in all_pre_tokens:
        # Only process tokens that are NOT in the special tokens list
        if pre_token not in special_tokens_set:
            token_as_bytes = pre_token.encode("utf-8")
            corpus.append([bytes([b]) for b in token_as_bytes])

    
    # 5. Ініціалізувати словник з одиночних байтів (0-255)
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
    
    # 6. Ітеративно виконувати BPE merges
    merges: List[Tuple[bytes, bytes]] = []
    
    if verbose:
        print(f"Початковий розмір vocab: {len(vocab)}, ціль: {vocab_size}")
        print("Початок BPE merges...")
    
    iteration = 0
    while len(vocab) < vocab_size:
        # Підрахувати частоти пар
        pair_freqs = get_pair_frequencies(corpus)
        
        if not pair_freqs:
            if verbose:
                print("Більше немає пар для об'єднання.")
            break
        
        # Знайти найчастішу пару
        # Виправлена версія
        best_pair = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))[0]
        best_freq = pair_freqs[best_pair]
        
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
        
        if verbose:
            print(f"Iteration {iteration}: vocab_size={len(vocab)}, "
                  f"merged {best_pair[0]!r} + {best_pair[1]!r} -> {new_token!r} (freq={best_freq})")
        
        # Замінити пару в корпусі
        corpus = [merge_pair_in_sequence(seq, best_pair, new_token) for seq in corpus]
        
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
    data = "/mnt/c/Users/Ihor/Desktop/Stanford_LLM/assignment1-basics/data/corpus.en"
    run_train_bpe(input_path=data, vocab_size=500, special_tokens=["<|endoftext|>"], verbose=True)
    pass