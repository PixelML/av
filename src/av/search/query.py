"""Bounded lexical candidate queries for natural-language questions.

This is a recall-oriented lookup, not a query rewrite or answer model. It does
not change the original question passed to embeddings, Jev, or answer synthesis.
"""

from __future__ import annotations

import re
import unicodedata

MAX_CANDIDATE_TERMS = 24
MAX_TERM_CHARACTERS = 80
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
# Generic English question/function words, not video-domain vocabulary.
_QUESTION_WORDS = frozenset("""
a an the am is are was were be been being do does did can could would should
shall will may might must have has had having i me my we us our you your he
him his she her it its they them their this that these those there here then
to of for from in on at by with as about and or but if so what which who whom
whose when where why how please tell show s t
""".split())


def natural_language_fts_query(question: str) -> str:
    """Return a safe OR of distinct content terms, or empty when none remain.

    Unicode letters/digits are retained; punctuation and FTS operators are never
    interpreted as syntax. Apply the term budget after removing question words
    and duplicates so a long question preamble cannot consume the candidate cap.
    """
    terms: list[str] = []
    seen: set[str] = set()
    normalized = unicodedata.normalize("NFC", question).casefold()
    for match in _WORDS.finditer(normalized):
        term = match.group()
        if term in _QUESTION_WORDS or term in seen or len(term) > MAX_TERM_CHARACTERS:
            continue
        seen.add(term)
        terms.append('"' + term.replace('"', '""') + '"')
        if len(terms) >= MAX_CANDIDATE_TERMS:
            break
    return " OR ".join(terms)
