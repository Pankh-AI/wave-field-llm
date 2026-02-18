"""
Field Tokenizer V3 - High-Coverage Tokenizer for Field LLM
============================================================

Keeps the field-aware co-design from V2, but closes the coverage gap:

V2 problem: field_size=1024 → 969 words → 86.8% coverage → 13% char fallback
V3 solution: morphological subwords + smarter vocab allocation → 95%+ coverage

Key improvements over V2:
1. Morphological subwords: common prefixes (-un, -re) and suffixes (-ing, -tion, -ed)
2. Stem extraction: "walking" → "walk" + "-ing" (2 tokens, not 7 characters)
3. Smarter char budget: only characters that actually appear, not full alphabet
4. Field-aware ordering preserved: stems near their suffixes in the field
5. Backward compatible: same API as V2
"""

from collections import Counter, defaultdict
import re
import json
import math
from typing import List, Dict, Tuple, Optional, Set


class FieldTokenizerV3:
    """
    High-coverage field-aware tokenizer.

    Closes the coverage gap by decomposing unknown words into
    stem + suffix instead of individual characters.
    """

    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
    }
    NUM_SPECIAL = len(SPECIAL_TOKENS)

    # Common English suffixes (ordered by length, longest first for greedy match)
    SUFFIXES = [
        'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ious', 'eous',
        'ling', 'ful', 'ous', 'ing', 'ity', 'ism', 'ist', 'ive',
        'ant', 'ent', 'ard', 'dom', 'ure', 'age',
        'ly', 'er', 'ed', 'es', 'en', 'al', 'le', 'ty',
        'th', 'st', 'nd', 'rd',
        's', 'd',
    ]

    # Common English prefixes
    PREFIXES = [
        'over', 'under', 'fore', 'with', 'mis', 'out', 'dis', 'pre',
        'un', 're',
    ]

    def __init__(self, field_size: int = 1024):
        self.field_size = field_size
        self.vocab: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.token_type: Dict[str, str] = {}
        self._built = False

        # Subword lookup for fast tokenization
        self._suffix_set: Set[str] = set()
        self._prefix_set: Set[str] = set()
        self._stem_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Morphological Decomposition
    # ------------------------------------------------------------------

    def _decompose_word(self, word: str) -> List[str]:
        """
        Decompose a word into morphological subwords.

        "unhappiness" → ["un-", "happi", "-ness"]
        "walking"     → ["walk", "-ing"]
        "replied"     → ["repli", "-ed"]
        "dogs"        → ["dog", "-s"]

        Returns the word unchanged if no decomposition improves coverage.
        """
        if len(word) <= 2:
            return [word]

        # Check cache
        if word in self._stem_cache:
            return self._stem_cache[word]

        best_parts = [word]  # default: no decomposition
        best_char_fallback = self._count_char_fallback([word])

        # Try all combinations of prefix + stem + suffix
        prefix_options = [('', word)]
        for prefix in self.PREFIXES:
            pfx_token = prefix + '-'
            if word.startswith(prefix) and len(word) > len(prefix) + 1 and pfx_token in self.vocab:
                prefix_options.append((pfx_token, word[len(prefix):]))

        for pfx_token, after_prefix in prefix_options:
            suffix_options = [('', after_prefix)]
            for suffix in self.SUFFIXES:
                sfx_token = '-' + suffix
                if after_prefix.endswith(suffix) and len(after_prefix) > len(suffix) and sfx_token in self.vocab:
                    suffix_options.append((sfx_token, after_prefix[:-len(suffix)]))

            for sfx_token, stem in suffix_options:
                if not stem:
                    continue

                # Try the stem directly and with common spelling adjustments
                stem_candidates = [stem]
                if len(stem) >= 2:
                    stem_candidates.append(stem + 'e')           # walk+ing → walke? no, but lov+ing → love
                    stem_candidates.append(stem + stem[-1])      # run+ing → runn? no, but stop+ed → stopp → stop
                    if len(stem) >= 3 and stem[-1] == stem[-2]:
                        stem_candidates.append(stem[:-1])        # runn → run, stopp → stop
                    if stem.endswith('i') and len(stem) >= 3:
                        stem_candidates.append(stem[:-1] + 'y')  # happi → happy, easi → easy

                for stem_try in stem_candidates:
                    parts = []
                    if pfx_token:
                        parts.append(pfx_token)

                    if stem_try in self.vocab:
                        parts.append(stem_try)
                    else:
                        parts.append(stem_try)  # will be char-fallback

                    if sfx_token:
                        parts.append(sfx_token)

                    if len(parts) <= 1:
                        continue

                    fallback = self._count_char_fallback(parts)
                    if fallback < best_char_fallback:
                        best_char_fallback = fallback
                        best_parts = parts

        if len(best_parts) > 1:
            self._stem_cache[word] = best_parts
            return best_parts

        return [word]

    def _count_char_fallback(self, parts: List[str]) -> int:
        """Count how many characters would need char fallback for these parts."""
        count = 0
        for part in parts:
            if part not in self.vocab:
                count += len(part)
        return count

    # ------------------------------------------------------------------
    # Vocabulary Building
    # ------------------------------------------------------------------

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary with morphological subwords for high coverage.
        """
        print(f"Building Field Tokenizer V3 (field_size={self.field_size})...")

        # Step 1: Tokenize all texts
        all_words_per_text = []
        for text in texts:
            words = self._word_tokenize(text)
            all_words_per_text.append(words)

        # Step 2: Count everything
        word_freq = Counter()
        char_freq = Counter()
        bigram_freq = Counter()
        adjacency = defaultdict(Counter)

        for words in all_words_per_text:
            word_freq.update(words)
            for w in words:
                char_freq.update(w)
            for i in range(len(words) - 1):
                bigram_freq[f"{words[i]} {words[i+1]}"] += 1
                adjacency[words[i]][words[i+1]] += 1

        total_unique_words = len(word_freq)
        total_word_occurrences = sum(word_freq.values())

        # Step 3: Analyze which suffixes/prefixes would help most
        suffix_value = Counter()
        prefix_value = Counter()
        words_needing_subword = []

        for word, freq in word_freq.items():
            if len(word) <= 3:
                continue
            for suffix in self.SUFFIXES:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    stem = word[:-len(suffix)]
                    suffix_value['-' + suffix] += freq
                    break
            for prefix in self.PREFIXES:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    prefix_value[prefix + '-'] += freq
                    break

        # Step 4: Allocate vocab budget
        budget = self.field_size - self.NUM_SPECIAL

        # 4a: Characters (only those that appear in data -- minimal set)
        data_chars = set(char_freq.keys())
        for ch in ".,!?;:'-\"()[] ":
            data_chars.add(ch)
        char_tokens = sorted(data_chars)
        char_budget = len(char_tokens)

        remaining = budget - char_budget

        # 4b: Subword budget -- top suffixes and prefixes
        useful_suffixes = [sfx for sfx, cnt in suffix_value.most_common(25) if cnt >= 3]
        useful_prefixes = [pfx for pfx, cnt in prefix_value.most_common(10) if cnt >= 3]
        subword_tokens = useful_suffixes + useful_prefixes
        subword_budget = len(subword_tokens)
        remaining -= subword_budget

        # 4c: Words -- fill the rest with most frequent words
        seen = set(char_tokens) | set(subword_tokens)
        selected_words = []

        for word, freq in word_freq.most_common():
            if len(selected_words) >= remaining:
                break
            if word not in seen and len(word) > 1:
                selected_words.append(word)
                seen.add(word)

        words_added = len(selected_words)

        # 4d: Fill any leftover slots with bigrams
        leftover = remaining - words_added
        selected_bigrams = []
        if leftover > 0:
            for bigram, freq in bigram_freq.most_common():
                if len(selected_bigrams) >= leftover:
                    break
                if bigram not in seen and freq >= 2:
                    selected_bigrams.append(bigram)
                    seen.add(bigram)

        # Step 5: Build vocabulary with field-aware ordering
        # Order: special → chars → subwords (prefixes then suffixes) → words (by frequency)
        # This keeps morphologically related tokens near each other in the field

        self.vocab = dict(self.SPECIAL_TOKENS)
        self.token_type = {}
        idx = self.NUM_SPECIAL

        # Characters
        for ch in char_tokens:
            self.vocab[ch] = idx
            self.token_type[ch] = 'char'
            idx += 1

        # Prefixes (grouped together in field)
        for pfx in useful_prefixes:
            if idx >= self.field_size:
                break
            self.vocab[pfx] = idx
            self.token_type[pfx] = 'prefix'
            idx += 1

        # Suffixes (grouped together in field, near prefixes)
        for sfx in useful_suffixes:
            if idx >= self.field_size:
                break
            self.vocab[sfx] = idx
            self.token_type[sfx] = 'suffix'
            idx += 1

        # Bigrams
        for bigram in selected_bigrams:
            if idx >= self.field_size:
                break
            self.vocab[bigram] = idx
            self.token_type[bigram] = 'bigram'
            idx += 1

        # Words (most frequent first)
        for word in selected_words:
            if idx >= self.field_size:
                break
            self.vocab[word] = idx
            self.token_type[word] = 'word'
            idx += 1

        # Build reverse mapping and subword sets
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._suffix_set = set(sfx for sfx in useful_suffixes if sfx in self.vocab)
        self._prefix_set = set(pfx for pfx in useful_prefixes if pfx in self.vocab)
        self._stem_cache = {}
        self._built = True

        # Step 6: Compute actual coverage
        direct_match = sum(1 for w in word_freq if w in self.vocab)
        decomposable = 0
        for word in word_freq:
            if word not in self.vocab:
                parts = self._decompose_word(word)
                if len(parts) > 1 and all(p in self.vocab for p in parts):
                    decomposable += 1

        # Stats
        type_counts = Counter(self.token_type.values())
        print(f"Vocabulary built: {len(self.vocab)} tokens (target: {self.field_size})")
        print(f"  Characters : {type_counts.get('char', 0)}")
        print(f"  Prefixes   : {type_counts.get('prefix', 0)}")
        print(f"  Suffixes   : {type_counts.get('suffix', 0)}")
        print(f"  Bigrams    : {type_counts.get('bigram', 0)}")
        print(f"  Words      : {type_counts.get('word', 0)}")
        print(f"  Special    : {self.NUM_SPECIAL}")
        print(f"  UNK tokens : ZERO (subword + character fallback)")
        print(f"  Unique words in data    : {total_unique_words:,}")
        print(f"  Direct vocab matches    : {direct_match:,} ({direct_match/total_unique_words*100:.1f}%)")
        print(f"  Subword decomposable    : {decomposable:,}")
        print(f"  Need char fallback      : {total_unique_words - direct_match - decomposable:,}")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _word_tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        return re.findall(r'\w+|[^\w\s]', text)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize with: trigram → bigram → word → subword decomposition → char fallback.
        """
        words = self._word_tokenize(text)
        tokens = []
        i = 0

        while i < len(words):
            # Try bigram match
            if i + 1 < len(words):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in self.vocab:
                    tokens.append(bigram)
                    i += 2
                    continue

            # Try direct word match
            if words[i] in self.vocab:
                tokens.append(words[i])
                i += 1
                continue

            # Try morphological decomposition
            parts = self._decompose_word(words[i])
            if len(parts) > 1:
                all_in_vocab = all(p in self.vocab for p in parts)
                if all_in_vocab:
                    tokens.extend(parts)
                    i += 1
                    continue

            # Character fallback (ZERO UNK)
            for ch in words[i]:
                if ch in self.vocab:
                    tokens.append(ch)
            i += 1

        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        parts = []
        prev_type = None

        for token_id in ids:
            token = self.id_to_token.get(token_id, '')
            if skip_special and token in self.SPECIAL_TOKENS:
                continue
            if not token:
                continue

            ttype = self.token_type.get(token, 'word')

            if ttype == 'suffix':
                # Suffixes glue to the previous token (no space)
                parts.append(token[1:])  # Remove the leading '-'
            elif ttype == 'prefix':
                # Prefixes: add space before, glue to next
                parts.append((' ' if parts else '') + token[:-1])  # Remove trailing '-'
            elif ttype == 'char':
                if prev_type == 'char':
                    parts.append(token)
                elif prev_type == 'prefix':
                    parts.append(token)
                else:
                    parts.append((' ' if parts else '') + token)
            else:
                # Word or bigram
                if prev_type == 'prefix':
                    parts.append(token)
                else:
                    parts.append((' ' if parts else '') + token)

            prev_type = ttype

        text = ''.join(parts)
        text = re.sub(r'\s+([.,!?;:\'"])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    def save(self, path: str):
        data = {
            'field_size': self.field_size,
            'vocab': self.vocab,
            'token_type': self.token_type,
            'version': 3,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved tokenizer to {path}")

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.field_size = data['field_size']
        self.vocab = data['vocab']
        self.token_type = data.get('token_type', {})
        self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        self._suffix_set = set(t for t, tt in self.token_type.items() if tt == 'suffix')
        self._prefix_set = set(t for t, tt in self.token_type.items() if tt == 'prefix')
        self._stem_cache = {}
        self._built = True
        print(f"Loaded tokenizer from {path} ({len(self.vocab)} tokens)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_report(self, texts: List[str]) -> Dict:
        """Report coverage including subword decomposition."""
        total_words = 0
        direct_match = 0
        subword_match = 0
        char_fallback = 0

        for text in texts:
            words = self._word_tokenize(text)
            for w in words:
                total_words += 1
                if w in self.vocab:
                    direct_match += 1
                else:
                    parts = self._decompose_word(w)
                    if len(parts) > 1 and all(p in self.vocab for p in parts):
                        subword_match += 1
                    else:
                        char_fallback += 1

        effective_coverage = (direct_match + subword_match) / max(total_words, 1) * 100

        return {
            'total_words': total_words,
            'direct_matches': direct_match,
            'subword_matches': subword_match,
            'char_fallback': char_fallback,
            'coverage_pct': direct_match / max(total_words, 1) * 100,
            'effective_coverage_pct': effective_coverage,
        }


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIELD TOKENIZER V3 - SELF TEST")
    print("=" * 70)

    # Load Shakespeare for a real test
    try:
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"Loaded Shakespeare: {len(text):,} chars, {len(lines):,} lines")
    except FileNotFoundError:
        lines = [
            "the cat sat on the mat walking slowly",
            "the dog was running and jumping over fences",
            "unhappiness is not something we desire",
            "the kingdom was ruled by a powerful king",
            "she replied with great kindness and understanding",
        ]
        print(f"Using test data: {len(lines)} lines")

    # Split
    split = int(len(lines) * 0.9)
    train_lines = lines[:split]
    val_lines = lines[split:]

    # Test V3 at different field sizes
    for fs in [1024, 2048]:
        print(f"\n{'='*70}")
        print(f"FIELD SIZE: {fs}")
        print(f"{'='*70}")

        tok = FieldTokenizerV3(field_size=fs)
        tok.build_vocab(train_lines)

        train_report = tok.coverage_report(train_lines)
        val_report = tok.coverage_report(val_lines)

        print(f"\n  Train coverage:")
        print(f"    Direct word match    : {train_report['coverage_pct']:.1f}%")
        print(f"    + Subword decomp     : {train_report['effective_coverage_pct']:.1f}%")
        print(f"    Char fallback words  : {train_report['char_fallback']:,}")

        print(f"\n  Val coverage:")
        print(f"    Direct word match    : {val_report['coverage_pct']:.1f}%")
        print(f"    + Subword decomp     : {val_report['effective_coverage_pct']:.1f}%")
        print(f"    Char fallback words  : {val_report['char_fallback']:,}")

        # Test tokenization examples
        examples = [
            "First Citizen: unhappiness and walking",
            "The kingdom was overthrown by rebellion",
            "She replied with kindness and understanding",
        ]
        print(f"\n  Tokenization examples:")
        for ex in examples:
            tokens = tok.tokenize(ex)
            decoded = tok.decode(tok.encode(ex))
            print(f"    Input:   '{ex}'")
            print(f"    Tokens:  {tokens}")
            print(f"    Decoded: '{decoded}'")
            print()

    # Compare V2 vs V3
    print("=" * 70)
    print("V2 vs V3 COMPARISON (field_size=1024)")
    print("=" * 70)

    from field_tokenizer_v2 import FieldTokenizerV2

    v2 = FieldTokenizerV2(field_size=1024)
    v2.build_vocab(train_lines)
    v2_train = v2.coverage_report(train_lines)
    v2_val = v2.coverage_report(val_lines)

    v3 = FieldTokenizerV3(field_size=1024)
    v3.build_vocab(train_lines)
    v3_train = v3.coverage_report(train_lines)
    v3_val = v3.coverage_report(val_lines)

    print(f"\n  {'Metric':<30} {'V2':>10} {'V3':>10} {'Diff':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Train coverage (direct)':<30} {v2_train['coverage_pct']:>9.1f}% {v3_train['coverage_pct']:>9.1f}% {v3_train['coverage_pct']-v2_train['coverage_pct']:>+9.1f}%")
    print(f"  {'Train coverage (effective)':<30} {v2_train['coverage_pct']:>9.1f}% {v3_train['effective_coverage_pct']:>9.1f}% {v3_train['effective_coverage_pct']-v2_train['coverage_pct']:>+9.1f}%")
    print(f"  {'Val coverage (direct)':<30} {v2_val['coverage_pct']:>9.1f}% {v3_val['coverage_pct']:>9.1f}% {v3_val['coverage_pct']-v2_val['coverage_pct']:>9.1f}%")
    print(f"  {'Val coverage (effective)':<30} {v2_val['coverage_pct']:>9.1f}% {v3_val['effective_coverage_pct']:>9.1f}% {v3_val['effective_coverage_pct']-v2_val['coverage_pct']:>+9.1f}%")
    print(f"  {'Train char fallback':<30} {v2_train['char_fallback']:>10,} {v3_train['char_fallback']:>10,}")
    print(f"  {'Val char fallback':<30} {v2_val['char_fallback']:>10,} {v3_val['char_fallback']:>10,}")

    print("\n  Done!")
