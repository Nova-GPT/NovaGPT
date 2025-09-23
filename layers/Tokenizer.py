from collections import Counter, defaultdict
import json
import re

class BPETokenizer:
    def __init__(self):
        self.vocab = {}  # id -> token mapping
        self.merges = {}  # (token1, token2) -> merged_token_id
        self.inverse_vocab = {}  # token -> id mapping
        
    def get_pairs_count(self, word_tokens):
        """Count frequency of adjacent token pairs"""
        pairs = defaultdict(int)
        for word in word_tokens:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs
    
    def merge_vocab(self, pair, word_tokens):
        """Merge all occurrences of the most frequent pair"""
        new_word_tokens = []
        for word in word_tokens:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tokens.append(new_word)
        return new_word_tokens
    
    def train(self, text : str, vocab_size : int, verbose= False):
        """Train BPE tokenizer on text"""
        # Initialize with byte-level tokens (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Convert text to bytes and split into words
        words = text.split()
        word_tokens = []
        for word in words:
            word_bytes = word.encode('utf-8')
            word_tokens.append([bytes([b]) for b in word_bytes])
        
        # Number of merge operations needed
        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            # Get pair frequencies
            pairs = self.get_pairs_count(word_tokens)
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if verbose:
                print(f"Merge {i+1}: {best_pair} -> {pairs[best_pair]} occurrences")
            
            # Merge the pair
            word_tokens = self.merge_vocab(best_pair, word_tokens)
            
            # Add to vocabulary and merges
            new_token_id = 256 + i
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[new_token_id] = merged_token
            self.merges[best_pair] = new_token_id
        
        # Build inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode_word(self, word):
        """Encode a single word using learned merges"""
        # Convert to bytes
        word_bytes = word.encode('utf-8')
        tokens = [bytes([b]) for b in word_bytes]
        
        # Apply merges in order they were learned
        while len(tokens) > 1:
            # Find pairs that can be merged
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # Get the earliest merge (lowest ID) that applies
            merge_indices = []
            for i, pair in enumerate(pairs):
                if pair in self.merges:
                    merge_indices.append((i, self.merges[pair]))
            
            if not merge_indices:
                break
                
            # Apply the merge with the lowest ID (earliest learned)
            merge_idx, _ = min(merge_indices, key=lambda x: x[1])
            
            # Perform the merge
            new_tokens = tokens[:merge_idx]
            new_tokens.append(tokens[merge_idx] + tokens[merge_idx + 1])
            new_tokens.extend(tokens[merge_idx + 2:])
            tokens = new_tokens
        
        # Convert to IDs
        return [self.inverse_vocab[token] for token in tokens]
    
    def encode(self, text):
        """Encode text to token IDs"""
        words = text.split()
        token_ids = []
        for word in words:
            token_ids.extend(self.encode_word(word))
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        tokens = [self.vocab[id] for id in token_ids]
        text_bytes = b''.join(tokens)
        return text_bytes.decode('utf-8', errors='replace')
    
    def save(self, filename):
        """Save tokenizer to disk"""
        data = {
            'vocab': {k: v.decode('utf-8', errors='replace') for k, v in self.vocab.items()},
            'merges': {f"{k[0].decode('utf-8', errors='replace')}||{k[1].decode('utf-8', errors='replace')}": v 
                      for k, v in self.merges.items()}
        }
        with open(f"{filename}.json", 'w') as f:
            json.dump(data, f, indent=2)
