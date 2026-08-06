from __future__ import annotations

import re
import unicodedata
from typing import Any


class TextPreprocessor:
    def __init__(self, config: dict[str, bool] | None = None) -> None:
        self.config = config or {
            "remove_emojis": True,
            "expand_abbreviations": True,
            "normalize_unicode": True,
            "remove_extra_spaces": True,
            "lowercase": True,
        }

        self.abbreviations = {
            "ur": "your",
            "u r": "you are",
            "pls": "please",
            "plz": "please",
            "thx": "thanks",
            "ty": "thank you",
            "btw": "by the way",
            "imo": "in my opinion",
            "imho": "in my humble opinion",
            "idk": "i don't know",
            "idc": "i don't care",
            "tbh": "to be honest",
            "fyi": "for your information",
            "gtg": "got to go",
            "ttyl": "talk to you later",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "kinda": "kind of",
            "sorta": "sort of",
            "outta": "out of",
            "cuz": "because",
            "cos": "because",
            "'cause": "because",
        }

        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def preprocess(self, text: str) -> str:
        if not text:
            return text

        processed = text

        if self.config.get("normalize_unicode", True):
            processed = self._normalize_unicode(processed)

        if self.config.get("remove_emojis", True):
            processed = self._remove_emojis(processed)

        if self.config.get("expand_abbreviations", True):
            processed = self._expand_abbreviations(processed)

        if self.config.get("lowercase", True):
            processed = processed.lower()

        if self.config.get("remove_extra_spaces", True):
            processed = self._clean_spaces(processed)

        return processed.strip()

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _remove_emojis(self, text: str) -> str:
        return self._emoji_pattern.sub("", text)

    def _expand_abbreviations(self, text: str) -> str:
        words = text.split()
        expanded: list[str] = []

        for word in words:
            lower_word = word.lower()
            clean_word = re.sub(r"[^\w\s]", "", lower_word)

            if clean_word in self.abbreviations:
                if len(clean_word) == 1 and len(word) > 1:
                    expanded.append(word)
                else:
                    expanded.append(self.abbreviations[clean_word])
            else:
                expanded.append(word)

        return " ".join(expanded)

    def _clean_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


class TextPreprocessorPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.preprocessor = TextPreprocessor(config=self.config.get("preprocessor_config", {}))
        self.filters: list[dict[str, Any]] = self.config.get("filters", [])

    def process(self, text: str) -> dict[str, Any]:
        original = text
        processed = self.preprocessor.preprocess(text)

        return {
            "original": original,
            "processed": processed,
            "changed": original != processed,
            "word_count": len(processed.split()),
            "char_count": len(processed),
        }

    def process_for_routing(self, text: str) -> str:
        return self.preprocessor.preprocess(text)

    def process_for_rag(self, text: str) -> str:
        return self.preprocessor.preprocess(text)
