"""
BPE Tokenizer Module

This module provides a Tokenizer class for encoding text into token IDs
and decoding token IDs back to text using Byte Pair Encoding (BPE).

Dependencies:
    - regex: For advanced pattern matching (install: pip install regex)
    - pickle: For loading serialized models (standard library)
    - typing: For type hints (standard library)
"""

import regex as re
import pickle
from typing import List, Dict, Tuple, Optional, Iterable, Iterator


class Tokenizer:
    """
    BPE Tokenizer that encodes text into token IDs and decodes token IDs back to text.
    Supports special tokens that are preserved during tokenization.
    
    Attributes:
        vocab (Dict[int, bytes]): Mapping from token IDs to byte representations
        merges (List[Tuple[bytes, bytes]]): List of BPE merge operations
        special_tokens (List[str]): List of special tokens to preserve
    
    Example:
        >>> # Load tokenizer from trained BPE model
        >>> with open("bpe_model.pkl", 'rb') as f:
        ...     data = pickle.load(f)
        >>> tokenizer = Tokenizer(data['vocab'], data['merges'], data['special_tokens'])
        >>> 
        >>> # Encode text
        >>> token_ids = tokenizer.encode("Hello world")
        >>> 
        >>> # Decode back to text
        >>> text = tokenizer.decode(token_ids)
    """
    
    # Pattern for pre-tokenization (matches contractions, words, numbers, punctuation, whitespace)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and special tokens.
        
        Args:
            vocab: Dictionary mapping token IDs to their byte representations (from BPE training)
            merges: List of merge operations (byte_pair_1, byte_pair_2) (from BPE training)
            special_tokens: Optional list of special tokens to add/preserve
        
        Example:
            >>> # vocab and merges should come from BPE training
            >>> tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        # Build reverse mapping: bytes -> token_id for encoding
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        # Build merge priority dict for encoding
        self.merge_priority = {pair: i for i, pair in enumerate(self.merges)}
        
        # Build special token pattern for splitting
        if self.special_tokens:
            # Sort special tokens by length, longest first, to handle overlaps correctly.
            # This ensures that the regex engine tries to match the longest token first.
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = "|".join(re.escape(st) for st in sorted_special_tokens)
        else:
            self.special_pattern = None
        
        self.special_set = set(self.special_tokens)
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: Optional[List[str]] = None
    ) -> 'Tokenizer':
        """
        Class method to construct a Tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to pickled vocabulary file
            merges_filepath: Path to pickled merges file
            special_tokens: Optional list of special tokens
        
        Returns:
            Tokenizer instance
        
        Example:
            >>> tokenizer = Tokenizer.from_files(
            ...     "vocab.pkl", 
            ...     "merges.pkl", 
            ...     special_tokens=["<|endoftext|>"]
            ... )
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode input text into a sequence of token IDs.
        
        Args:
            text: Input text string to encode
        
        Returns:
            List of token IDs
        
        Example:
            >>> token_ids = tokenizer.encode("Hello world!")
        """
        if not text:
            return []
        
        # Split on special tokens
        if self.special_pattern:
            parts = re.split(f"({self.special_pattern})", text)
        else:
            parts = [text]
        
        # Pre-tokenize each part
        pre_tokens = []
        for part in parts:
            if part in self.special_set:
                pre_tokens.append(part)
            elif part:
                pre_tokens.extend(re.findall(self.PAT, part))
        
        # Apply BPE merges to each pre-token
        result = []
        for pre_token in pre_tokens:
            # Handle special tokens
            if pre_token in self.special_set:
                # Find the ID for this special token in vocab
                special_bytes = pre_token.encode("utf-8")
                if special_bytes in self.token_to_id:
                    result.append(self.token_to_id[special_bytes])
                continue
            
            # Convert to byte tokens
            tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
            
            # Apply merges iteratively
            while len(tokens) > 1:
                # Find the pair with highest priority (lowest merge index)
                best_pair = None
                best_idx = None
                best_priority = float('inf')
                
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_priority and self.merge_priority[pair] < best_priority:
                        best_pair = pair
                        best_idx = i
                        best_priority = self.merge_priority[pair]
                
                if best_pair is None:
                    break
                
                # Merge the best pair
                tokens = tokens[:best_idx] + [best_pair[0] + best_pair[1]] + tokens[best_idx + 2:]
            
            # Convert tokens to IDs using token_to_id mapping
            for token in tokens:
                if token in self.token_to_id:
                    result.append(self.token_to_id[token])
        
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (e.g., file handle) into token IDs.
        Memory-efficient for large files that cannot be loaded into memory.
        
        Args:
            iterable: Iterable of strings (e.g., file handle opened for reading)
        
        Yields:
            Token IDs one at a time
        
        Example:
            >>> with open("large_file.txt", "r") as f:
            ...     for token_id in tokenizer.encode_iterable(f):
            ...         # Process token_id
            ...         pass
        """
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into text.
        
        Args:
            ids: List of token IDs to decode
        
        Returns:
            Decoded text string
        
        Example:
            >>> text = tokenizer.decode([72, 101, 108, 108, 111])
        """
        if not ids:
            return ""
        
        # Collect byte sequences for all token IDs
        byte_sequences = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_sequences)
        
        # Decode to string
        try:
            text = all_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = all_bytes.decode("utf-8", errors="ignore")
        
        return text
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        return (f"Tokenizer(vocab_size={len(self.vocab)}, "
                f"num_merges={len(self.merges)}, "
                f"special_tokens={self.special_tokens})")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("BPE Tokenizer - Example Usage")
    print("=" * 60)
    
    # Example 1: Load from trained BPE model
    print("\n=== Example 1: Load from Trained BPE Model ===")
    try:
        # Load from BPE training output
        with open("bpe_model.pkl", 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = Tokenizer(
            vocab=data['vocab'], 
            merges=data['merges'], 
            special_tokens=data['special_tokens']
        )
        print(tokenizer)
        
        test_text = "One day, a little boy named Tim went to the park."
        encoded = tokenizer.encode(test_text)
        print(f"\nEncoded: '{test_text}'")
        print(f"Token IDs: {encoded}")
        print(f"Number of tokens: {len(encoded)}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        print(f"Match: {decoded == test_text}")
        
    except FileNotFoundError:
        print("bpe_model.pkl not found. Run BPE_v4.py first to train a model.")