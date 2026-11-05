"""

features.py is the part of our project that converts a headline and the article body into 
four measurements so that our model can understand it (i.e. X = [x1, x2, x3, x4]).


implemented with the help of spaCy (evidence ent types) and numpy 
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import spacy

FEATURE_NAMES = ["headline_charge", "headline_attribution", "body_evidence", "tone_gap"]

"""
manually created list of words we made that indicate attribution 
(i.e. it attributes that the headline is reporting or attributing information to a source - e.g. "Critics claim x")
"""
ATTRIBUTION_CUES = {
    "say", "claim", "according", "critic", "allege", "accuse", "tell",
    "announce", "report", "argue", "insist", "state", "warn", "deny",
}
# Double quotes only; single quotes collide with apostrophes. Just for data checking.
QUOTE_CHARS = ('"', "\u201c", "\u201d")
# entities spaCy should count as evidence in the article body
# e.g. "PERSON" refers to people's names (e.g. Mark Carney)
# "ORG" refers to organizations such as the United Nations
# "GPE" - countries, cities, states
# "LOC" - geographic locations
# "NORP" - nationalities, religious or political groups 
EVIDENCE_ENT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP"}


def load_nlp(model: str = "en_core_web_sm"):
    """Load the spaCy pipeline (tokenizer, lemmatizer, NER).
    en_core_web_sm is a model used by spaCy so that we can tokenize text, lemmatize, and NER (identify named entities)
    """
    return spacy.load(model)


class Lexicon:
    """Word bank L. `lemma in lexicon` is True for an exact entry or a stem match. Using subjclues.tff (MPQA Subjectivity Lexicon - University of Pittsburgh)"""

    def __init__(self, exact, prefixes=()):
        self.exact = set(exact)
        self.prefixes = tuple(sorted(set(prefixes)))

    def __contains__(self, lemma: str) -> bool:
        return lemma in self.exact or (bool(self.prefixes) and lemma.startswith(self.prefixes))

    def __len__(self) -> int:
        return len(self.exact) + len(self.prefixes)


def _parse_mpqa(lines, include_weak: bool) -> Lexicon:
    """Parse MPQA subjclues lines, e.g.
    type=strongsubj len=1 word1=boundless pos1=adj stemmed1=n priorpolarity=positive
    """
    exact, prefixes = set(), set()
    for line in lines:
        fields = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        word = fields.get("word1", "").lower()
        if not word or fields.get("len", "1") != "1":
            continue  # single-word entries only
        if fields.get("type") == "weaksubj" and not include_weak:
            continue  # keep strongly subjective clues only, unless asked
        # stemmed1=y marks a stem: it matches any lemma that starts with it
        (prefixes if fields.get("stemmed1") == "y" else exact).add(word)
    return Lexicon(exact, prefixes)


def load_lexicon(path: str | Path, include_weak: bool = False) -> Lexicon:
    """Load the word bank L.

    Goal of this function is to load subjclues.tff into our program for feature extraction
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if lines and "word1=" in lines[0]:
        return _parse_mpqa(lines, include_weak)
    return Lexicon({ln.lower() for ln in lines})


def matched_words(doc, lexicon) -> list[str]:
    """Words in a spaCy doc that hit the lexicon. Use it to check what the
    charge feature actually sees before trusting it.
    
    
    Purpose: Shows us which words in a piece of text match the subjective word bank
    """
    return [t.text for t in content_tokens(doc) if t.lemma_.lower() in lexicon]


def content_tokens(doc):
    """Tokens excluding punctuation and whitespace because we don't want puncuation to count toward our total number of words when calculating 
    subjective-word ratio.
    
    
    Purpose: Removes puncuation and whitespace from doc
    """
    return [t for t in doc if not t.is_punct and not t.is_space]


def charge(doc, lexicon: "Lexicon | set[str]") -> float:
    """Fraction of content tokens whose lemma is in the lexicon.
    
    Purpose: Calculates the proportion of words in the text that match our lexicon of subjective words. 
    """
    toks = content_tokens(doc)
    if not toks:
        return 0.0 # edge case: if text empty, return 0 instead of trying to divide by zero
    return sum(t.lemma_.lower() in lexicon for t in toks) / len(toks)


def headline_attribution(doc) -> float:
    """1.0 if the headline has an attribution cue or a quotation mark, else 0.0.
    
    Purpose: Checks whether headline attributes a statement or claim to someone. 
    """
    has_quote = any(q in doc.text for q in QUOTE_CHARS) # checks if headline has quote
    has_cue = any(t.lemma_.lower() in ATTRIBUTION_CUES for t in doc) # checks whether any word in the headline matches ATTRIBUTEION_CUES
    return 1.0 if (has_quote or has_cue) else 0.0 # we 1.0 if headline contaisn a quote or attribution cue, 0 if neither


def evidence_rate(doc) -> float:
    """Concrete grounding per 100 content tokens in the body:
    numeric tokens + named entities (person/org/place/group) + attribution cues.
    
    Purpose: Measures how much concrete info or attribution language appears in the article body. 
    """
    toks = content_tokens(doc)
    if not toks:
        return 0.0 # return 0 to avoid dividing by 0
    n_num = sum(t.like_num for t in toks)
    n_ent = sum(ent.label_ in EVIDENCE_ENT_TYPES for ent in doc.ents)
    n_attr = sum(t.lemma_.lower() in ATTRIBUTION_CUES for t in toks)
    return 100.0 * (n_num + n_ent + n_attr) / len(toks)


def featurize(headline: str, body: str, nlp, lexicon: "Lexicon | set[str]") -> np.ndarray:
    """phi(h, a): one (headline, body) pair -> feature vector of length 4.
    
    Purpose: Takes one headline and the article body and converts them into the four numerical features. (Basically all features are done here)
    """
    h_doc, a_doc = nlp(headline), nlp(body)
    x1 = charge(h_doc, lexicon)                 # headline charge
    x2 = headline_attribution(h_doc)            # attribution cue or quote in headline
    x3 = evidence_rate(a_doc)                   # body evidence
    x4 = x1 - charge(a_doc, lexicon)            # tone gap: headline minus body
    return np.array([x1, x2, x3, x4], dtype=float) # given the headline and body we'll get an array of scores


def featurize_many(pairs, nlp, lexicon: "Lexicon | set[str]") -> np.ndarray:
    """pairs: iterable of (headline, body) -> array of shape (n, 4).
    
    Purpose: Same thing as featurize() but for multiple articles - key for processing an entire dataset
    """
    return np.vstack([featurize(h, a, nlp, lexicon) for h, a in pairs])


if __name__ == "__main__":
    # Little test with a made-up lexicon and invented text
    toy_lexicon = {"authoritarian", "backlash", "chaos"} 
    nlp = load_nlp()
    h = "China's authoritarian health mandate sparks backlash"
    a = "The ministry said the quarantine rule takes effect Monday, according to a statement."
    print(dict(zip(FEATURE_NAMES, featurize(h, a, nlp, toy_lexicon).round(3))))
