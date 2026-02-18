"""
Field-Aware Tokenizer for Field LLM

This tokenizer is specifically designed for Field LLM's attention mechanism:
1. Optimizes for 1D convolution in field space
2. Groups semantically related tokens together
3. Preserves co-occurrence patterns
4. Matches vocabulary size to field_size for optimal attention
"""

from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple
import math


class FieldAwareTokenizer:
    def __init__(self, field_size=256):
        """
        Initialize Field-Aware Tokenizer.
        
        Args:
            field_size: The field dimension used in Field LLM attention.
                       Vocabulary will be optimized for this size.
        """
        self.field_size = field_size
        self.vocab_size = field_size  # Match vocab to field size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        self.special_tokens = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        
        self.vocab = {}
        self.id_to_token = {}
        self.token_clusters = {}  # Maps tokens to semantic clusters
        
    def build_vocab(self, texts: List[str], min_freq=1):
        """
        Build field-aware vocabulary optimized for Field LLM.
        
        Strategy:
        1. Extract n-grams with co-occurrence scoring
        2. Cluster by semantic similarity (position + context)
        3. Prioritize chunks that preserve field locality
        4. Optimize vocab size for field_size
        """
        print(f"Building field-aware vocabulary (field_size={self.field_size})...")
        
        # Step 1: Extract and score n-grams
        ngram_stats = self._extract_ngrams(texts, min_freq)
        
        # Step 2: Score n-grams by field-awareness
        scored_ngrams = self._score_for_field_attention(ngram_stats, texts)
        
        # Step 3: Select best tokens for vocabulary
        vocab_items = self._select_vocab_items(scored_ngrams)
        
        # Step 4: Build vocabulary with semantic clustering
        self._build_vocab_dict(vocab_items)
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
        print(f"  - Trigrams: {sum(1 for t in vocab_items if t.count(' ') == 2)}")
        print(f"  - Bigrams: {sum(1 for t in vocab_items if t.count(' ') == 1)}")
        print(f"  - Words: {sum(1 for t in vocab_items if ' ' not in t)}")
        
    def _extract_ngrams(self, texts: List[str], min_freq: int) -> Dict:
        """Extract n-grams with frequency and co-occurrence info."""
        word_counts = Counter()
        bigram_counts = Counter()
        trigram_counts = Counter()
        
        # Co-occurrence matrix: which tokens appear near each other
        # OPTIMIZED: Only track adjacent pairs (not full window)
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for text in texts:
            words = self._basic_tokenize(text)
            
            # Count words
            word_counts.update(words)
            
            # Count bigrams and track co-occurrence (single pass)
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_counts[bigram] += 1
                # Only track adjacent co-occurrence (much faster!)
                cooccurrence[words[i]][words[i+1]] += 1
            
            # Count trigrams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                trigram_counts[trigram] += 1
        
        return {
            'words': word_counts,
            'bigrams': bigram_counts,
            'trigrams': trigram_counts,
            'cooccurrence': cooccurrence,
        }
    
    def _score_for_field_attention(self, ngram_stats: Dict, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Score n-grams based on how well they work with field attention.
        
        Field attention uses 1D convolution, so we want:
        1. High frequency (common patterns)
        2. Strong co-occurrence (tokens that appear together)
        3. Semantic coherence (meaningful chunks)
        4. Compression ratio (fewer tokens = better field utilization)
        """
        scored = []
        
        # Score trigrams
        for trigram, freq in ngram_stats['trigrams'].items():
            if freq < 1:
                continue
            
            words = trigram.split()
            
            # Field-aware score components:
            # 1. Frequency score (log scale)
            freq_score = math.log(freq + 1)
            
            # 2. Co-occurrence strength (how often these words appear together)
            cooccur_score = 0
            for i in range(len(words)-1):
                cooccur_score += ngram_stats['cooccurrence'][words[i]].get(words[i+1], 0)
            cooccur_score = math.log(cooccur_score + 1)
            
            # 3. Compression benefit (3 words → 1 token)
            compression_score = 3.0
            
            # 4. Length penalty (prefer not too long)
            length_penalty = 1.0 / (len(trigram) + 1)
            
            # Combined score
            score = (freq_score * 2.0 + cooccur_score * 3.0 + 
                    compression_score * 1.5) * length_penalty
            
            scored.append((trigram, score))
        
        # Score bigrams
        for bigram, freq in ngram_stats['bigrams'].items():
            if freq < 1:
                continue
            
            words = bigram.split()
            
            freq_score = math.log(freq + 1)
            cooccur_score = math.log(
                ngram_stats['cooccurrence'][words[0]].get(words[1], 0) + 1
            )
            compression_score = 2.0
            length_penalty = 1.0 / (len(bigram) + 1)
            
            score = (freq_score * 2.0 + cooccur_score * 3.0 + 
                    compression_score * 1.5) * length_penalty
            
            scored.append((bigram, score))
        
        # Score individual words
        for word, freq in ngram_stats['words'].items():
            if freq < 1:
                continue
            
            freq_score = math.log(freq + 1)
            
            # Co-occurrence diversity (how many different contexts)
            context_diversity = len(ngram_stats['cooccurrence'][word])
            diversity_score = math.log(context_diversity + 1)
            
            compression_score = 1.0
            length_penalty = 1.0 / (len(word) + 1)
            
            score = (freq_score * 1.5 + diversity_score * 1.0 + 
                    compression_score * 1.0) * length_penalty
            
            scored.append((word, score))
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def _select_vocab_items(self, scored_ngrams: List[Tuple[str, float]]) -> List[str]:
        """
        Select best tokens for vocabulary, avoiding redundancy.
        """
        vocab_items = []
        seen_words = set()
        
        # Reserve space for special tokens
        max_items = self.vocab_size - len(self.special_tokens)
        
        for token, score in scored_ngrams:
            if len(vocab_items) >= max_items:
                break
            
            # Check if this token is redundant
            words = set(token.split())
            
            # For trigrams/bigrams: only add if not all words are already covered
            if ' ' in token:
                # Add if it provides new coverage or is high-scoring
                if not words.issubset(seen_words) or score > 5.0:
                    vocab_items.append(token)
                    seen_words.update(words)
            else:
                # Single word: always add if not seen
                if token not in seen_words:
                    vocab_items.append(token)
                    seen_words.add(token)
        
        return vocab_items
    
    def _build_vocab_dict(self, vocab_items: List[str]):
        """Build vocabulary dictionary with semantic clustering."""
        self.vocab = {**self.special_tokens}
        
        # Group tokens by type for better field locality
        trigrams = [t for t in vocab_items if t.count(' ') == 2]
        bigrams = [t for t in vocab_items if t.count(' ') == 1]
        words = [t for t in vocab_items if ' ' not in t]
        
        # Assign IDs in clusters (helps field attention)
        idx = len(self.special_tokens)
        
        # Trigrams first (most semantic)
        for token in trigrams:
            self.vocab[token] = idx
            self.token_clusters[token] = 'trigram'
            idx += 1
        
        # Then bigrams
        for token in bigrams:
            self.vocab[token] = idx
            self.token_clusters[token] = 'bigram'
            idx += 1
        
        # Then words
        for token in words:
            self.vocab[token] = idx
            self.token_clusters[token] = 'word'
            idx += 1
        
        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic word tokenization."""
        text = text.lower().strip()
        words = re.findall(r'\w+|[^\w\s]', text)
        return words
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using greedy matching (longest first).
        """
        words = self._basic_tokenize(text)
        tokens = []
        i = 0
        
        while i < len(words):
            matched = False
            
            # Try trigram
            if i + 2 < len(words):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if trigram in self.vocab:
                    tokens.append(trigram)
                    i += 3
                    matched = True
                    continue
            
            # Try bigram
            if not matched and i + 1 < len(words):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in self.vocab:
                    tokens.append(bigram)
                    i += 2
                    matched = True
                    continue
            
            # Try word
            if not matched:
                word = words[i]
                if word in self.vocab:
                    tokens.append(word)
                else:
                    tokens.append(self.unk_token)
                i += 1
        
        return tokens
    
    def encode(self, text: str, add_special_tokens=False) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens=True) -> str:
        """Decode token IDs to text."""
        tokens = []
        
        for id in ids:
            token = self.id_to_token.get(id, self.unk_token)
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Join and clean up spacing
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        
        return text.strip()
    
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.vocab)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        import json
        
        data = {
            'vocab': self.vocab,
            'field_size': self.field_size,
            'token_clusters': self.token_clusters,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.field_size = data['field_size']
        self.token_clusters = data.get('token_clusters', {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Tokenizer loaded from {path}")


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing Field-Aware Tokenizer")
    print("="*70)
    
    # Sample texts
    texts = [
        "the cat sat on the mat",
        "the dog sat on the floor",
        "the cat ran to the door",
        "the dog ran to the mat",
        "a bird flew over the tree",
        "a bird sat on the tree",
    ]
    
    # Build tokenizer
    tokenizer = FieldAwareTokenizer(field_size=256)
    tokenizer.build_vocab(texts, min_freq=1)
    
    print(f"\nVocabulary size: {tokenizer.vocab_size_actual()}")
    
    # Test tokenization
    test_text = "the cat sat on the mat"
    tokens = tokenizer.tokenize(test_text)
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    
    print(f"\nTest: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Decoded: '{decoded}'")
    
    # Show some vocabulary
    print(f"\nSample vocabulary (first 20):")
    for i, (token, id) in enumerate(list(tokenizer.vocab.items())[:20]):
        cluster = tokenizer.token_clusters.get(token, 'special')
        print(f"  {id:3d}: '{token}' ({cluster})")
