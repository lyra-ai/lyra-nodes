"""
Lyra Character Search
=====================

A fuzzy character search node using RapidFuzz for matching against
a Danbooru character database. Returns trigger words and core tags
for matched characters, or empty strings if no confident match is found.
"""

import os
import folder_paths
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from typing import Dict, Tuple, Optional

class LyraCharacterSearch:
    CATEGORY = "Lyra/Search"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("trigger_words", "core_tags")
    FUNCTION = "search_character"

    # Class-level cache for the dataframe
    _df: Optional[pd.DataFrame] = None
    _csv_path: str = ""

    # Filler words to filter out
    FILLER_WORDS = frozenset({'from', 'the', 'of', 'in', 'at', 'by', 'and', 'or', 'to', 'a', 'an'})

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "query": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Character name to search for (e.g., 'hatsune miku', 'rem re:zero')",
                }),
                "min_score": ("INT", {
                    "default": 80,
                    "min": 50,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Minimum fuzzy match score (higher = stricter matching)",
                }),
                "validation_threshold": ("INT", {
                    "default": 60,
                    "min": 30,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Cross-validation threshold for result confidence",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed value to prevent caching (change to force re-execution)",
                }),
            },
            "optional": {
                "csv_path": ("STRING", {
                    "default": "data/danbooru_character_webui.csv",
                    "multiline": False,
                    "tooltip": "Path to the character CSV file (relative to ComfyUI base or absolute)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, query, min_score, validation_threshold, seed, csv_path=""):
        """Force re-execution when seed changes."""
        return seed

    @classmethod
    def _resolve_csv_path(cls, csv_path: str) -> Optional[str]:
        """
        Resolve the CSV path by checking multiple possible locations.
        Returns the first valid path found, or None if not found.
        """
        paths_to_try = []

        # 1. Absolute path (if provided)
        if os.path.isabs(csv_path):
            paths_to_try.append(csv_path)

        # 2. Relative to ComfyUI base directory
        try:
            comfy_base = folder_paths.base_path
            paths_to_try.append(os.path.join(comfy_base, csv_path))
        except:
            pass

        # 3. Relative to this node's directory
        node_dir = os.path.dirname(os.path.abspath(__file__))
        paths_to_try.append(os.path.join(node_dir, csv_path))

        # 4. Relative to parent of node directory (custom_nodes/YourNodePack/)
        parent_dir = os.path.dirname(node_dir)
        paths_to_try.append(os.path.join(parent_dir, csv_path))

        # 5. Relative to custom_nodes directory
        custom_nodes_dir = os.path.dirname(parent_dir)
        paths_to_try.append(os.path.join(custom_nodes_dir, csv_path))

        # 6. Current working directory
        paths_to_try.append(os.path.join(os.getcwd(), csv_path))

        # 7. Direct relative path
        paths_to_try.append(csv_path)

        # Try each path
        for path in paths_to_try:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path) and os.path.isfile(normalized_path):
                print(f"[Lyra CharacterSearch] Found CSV at: '{normalized_path}'")
                return normalized_path

        # Log all attempted paths for debugging
        print(f"[Lyra CharacterSearch] CSV not found. Tried the following paths:")
        for path in paths_to_try:
            print(f"  - {os.path.normpath(path)}")

        return None

    @classmethod
    def _load_dataframe(cls, csv_path: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """Load and cache the character database."""
        resolved_path = cls._resolve_csv_path(csv_path)

        if resolved_path is None:
            print(f"[Lyra CharacterSearch] Error: Could not find CSV file '{csv_path}'")
            return None

        if cls._df is None or cls._csv_path != resolved_path or force_reload:
            try:
                cls._df = pd.read_csv(resolved_path)
                cls._csv_path = resolved_path
                print(f"[Lyra CharacterSearch] Loaded {len(cls._df)} characters from '{resolved_path}'")
            except Exception as e:
                print(f"[Lyra CharacterSearch] Error loading CSV: {e}")
                return None

        return cls._df

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by removing special characters and extra whitespace."""
        text = text.replace("_", " ").replace("(", "").replace(")", "").replace(":", "")
        return " ".join(text.strip().lower().split())

    def _generate_permutations(self, words: list) -> list:
        """Generate word order permutations for multi-word queries."""
        if len(words) <= 1:
            return [" ".join(words)] if words else []

        permutations = [" ".join(words)]

        for i in range(len(words)):
            # Word moved to front
            perm = " ".join([words[i]] + words[:i] + words[i+1:])
            if perm not in permutations:
                permutations.append(perm)

            # Word moved to end
            perm = " ".join(words[:i] + words[i+1:] + [words[i]])
            if perm not in permutations:
                permutations.append(perm)

        return permutations

    def _check_keyword_overlap(self, query_words: list, target_text: str) -> bool:
        """
        Check if there's meaningful keyword overlap between query and target.
        Returns False if no significant words from the query appear in the target.
        """
        if not query_words:
            return False

        target_words = set(self._normalize_text(target_text).split())

        # Check for exact word matches
        for word in query_words:
            if len(word) >= 3 and word in target_words:
                return True
            # Also check if query word is a substring of any target word (for partial names)
            for target_word in target_words:
                if len(word) >= 3 and (word in target_word or target_word in word):
                    return True

        return False

    def search_character(
        self,
        query: str,
        min_score: int = 80,
        validation_threshold: int = 60,
        seed: int = 0,
        csv_path: str = "data/danbooru_character_webui.csv"
    ) -> Tuple[str, str]:
        """
        Search for a character and return trigger words and core tags.
        Returns empty strings if no confident match is found.
        """
        print(f"[Lyra CharacterSearch] Searching for: '{query}' (seed: {seed})")

        # Handle empty query
        if not query or not query.strip():
            print("[Lyra CharacterSearch] Empty query provided")
            return ("", "")

        # Load dataframe
        df = self._load_dataframe(csv_path)
        if df is None or df.empty:
            print("[Lyra CharacterSearch] No data available")
            return ("", "")

        # Normalize query and filter filler words
        normalized_query = self._normalize_text(query)
        query_words = [w for w in normalized_query.split() if w not in self.FILLER_WORDS]

        if not query_words:
            print("[Lyra CharacterSearch] Query contains only filler words")
            return ("", "")

        filtered_query = " ".join(query_words)
        query_permutations = self._generate_permutations(query_words)

        print(f"[Lyra CharacterSearch] Normalized query: '{filtered_query}'")
        print(f"[Lyra CharacterSearch] Generated {len(query_permutations)} permutations")

        # Prepare character data
        characters = df['character'].apply(self._normalize_text).to_numpy()
        triggers = df['trigger'].str.lower().to_numpy()
        combined_texts = np.char.add(np.char.add(characters, " "), triggers)

        # Try exact match first
        for perm in query_permutations:
            exact_matches = np.where(characters == perm)[0]
            if len(exact_matches) > 0:
                best_match = df.iloc[exact_matches[0]]
                trigger = str(best_match['trigger']) if pd.notna(best_match['trigger']) else ""
                core_tags = str(best_match['core_tags']) if pd.notna(best_match['core_tags']) else ""
                print(f"[Lyra CharacterSearch] Exact match found: '{best_match['character']}'")
                return (trigger, core_tags)

        # Fuzzy matching with all permutations
        best_score = 0
        best_idx = -1
        best_perm = ""

        for perm in query_permutations:
            scores = np.array([
                max(
                    fuzz.token_sort_ratio(perm, text),
                    fuzz.partial_ratio(perm, text)
                ) for text in combined_texts
            ])

            perm_best_idx = np.argmax(scores)
            perm_best_score = scores[perm_best_idx]

            if perm_best_score > best_score:
                best_score = perm_best_score
                best_idx = perm_best_idx
                best_perm = perm

        print(f"[Lyra CharacterSearch] Best fuzzy score: {best_score}")

        # Strict check: minimum score threshold
        if best_score < min_score:
            print(f"[Lyra CharacterSearch] Score {best_score} below minimum {min_score}")
            return ("", "")

        best_match = df.iloc[best_idx]
        result_trigger = str(best_match['trigger']).lower() if pd.notna(best_match['trigger']) else ""
        result_character = str(best_match['character']).lower() if pd.notna(best_match['character']) else ""

        # Strict check: keyword overlap validation
        combined_result = f"{result_character} {result_trigger}"
        if not self._check_keyword_overlap(query_words, combined_result):
            print(f"[Lyra CharacterSearch] No keyword overlap between '{filtered_query}' and '{combined_result}'")
            return ("", "")

        # Cross-reference validation
        validation_score = max(
            fuzz.token_sort_ratio(perm, result_trigger)
            for perm in query_permutations
        )

        print(f"[Lyra CharacterSearch] Validation score: {validation_score}")

        if validation_score < validation_threshold:
            # Also try validating against character name
            char_validation = max(
                fuzz.token_sort_ratio(perm, result_character)
                for perm in query_permutations
            )
            if char_validation < validation_threshold:
                print(f"[Lyra CharacterSearch] Validation failed (trigger: {validation_score}, char: {char_validation})")
                return ("", "")

        # Success - return results
        trigger = str(best_match['trigger']) if pd.notna(best_match['trigger']) else ""
        core_tags = str(best_match['core_tags']) if pd.notna(best_match['core_tags']) else ""

        print(f"[Lyra CharacterSearch] Match found: '{best_match['character']}' (score: {best_score})")

        return (trigger, core_tags)