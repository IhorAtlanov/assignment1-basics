import regex as re  # Використовуємо regex замість re для підтримки \p{L} та \p{N}
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Iterable
from multiprocessing import Pool, cpu_count
from cs336_basics.BPE.BPE_v3 import _pretokenize_chunk

# Шаблон для pre-tokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def stream_pretokenize_lines(input_path: str, special_tokens: List[str], workers: int = None,
                             lines_per_batch: int = 2000) -> Iterable[str]:
    """
    Стрімінгово читає файл по рядках, робить split по special tokens (якщо є),
    і повертає претокени по одному. Паралелізує обробку пакетів рядків, якщо
    workers > 1.
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)
    special_set = set(special_tokens)

    pattern = None
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)

    if workers <= 1:
        # Послідовна (менше overhead)
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if pattern:
                    parts = re.split(f"({pattern})", line)
                else:
                    parts = [line]
                for part in parts:
                    if part in special_set:
                        yield part
                    elif part:
                        for tok in re.findall(PAT, part):
                            yield tok
    else:
        # Паралельна обробка пакетів рядків у невеликих батчах
        pool = Pool(processes=workers)
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                batch_parts_args = []
                while True:
                    # Зчитаємо lines_per_batch * workers рядків і розділимо їх на workers частин
                    raw_lines = []
                    for _ in range(lines_per_batch * workers):
                        line = f.readline()
                        if not line:
                            break
                        raw_lines.append(line)
                    if not raw_lines:
                        break

                    # розбиваємо на workers кусочків (напр., приблизно рівних)
                    chunk_size = (len(raw_lines) + workers - 1) // workers
                    args_list = []
                    for i in range(0, len(raw_lines), chunk_size):
                        chunk_lines = raw_lines[i:i + chunk_size]
                        chunk_parts = []
                        for line in chunk_lines:
                            if pattern:
                                parts = re.split(f"({pattern})", line)
                            else:
                                parts = [line]
                            chunk_parts.extend(parts)
                        args_list.append((chunk_parts, special_set))

                    # Викликаємо паралельно
                    results = pool.map(_pretokenize_chunk, args_list)
                    for res in results:
                        for tok in res:
                            yield tok
        finally:
            pool.close()
            pool.join()