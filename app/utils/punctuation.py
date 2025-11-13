"""Punctuation restoration using Stanza NLP pipeline."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PunctuatedText:
    """Result of punctuation restoration."""
    
    text: str
    is_question: bool
    has_commas: bool
    confidence: float = 1.0


@lru_cache(maxsize=1)
def _load_stanza_pipeline(lang: str = "nb"):
    """Load and cache the Stanza NLP pipeline.
    
    Args:
        lang: Language code (nb=Norwegian Bokmål, nn=Norwegian Nynorsk, sme=Northern Sámi)
    
    Note: For Northern Sámi, we'll fall back to Norwegian as Stanza doesn't have
    native support for Sámi languages yet. The punctuation patterns are similar enough.
    """
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError(
            "stanza is required for punctuation restoration. "
            "Install it with: pip install stanza"
        ) from exc
    
    # Map sme to Norwegian as fallback
    if lang == "sme":
        lang = "nb"
        logger.info("Using Norwegian (nb) Stanza pipeline for Northern Sámi text")
    
    try:
        # Download the model if not present
        stanza.download(lang, verbose=False, logging_level='ERROR')
    except Exception:
        # If download fails, try to use existing model
        logger.debug("Could not download Stanza model, attempting to use existing", exc_info=True)
    
    try:
        # Initialize pipeline with tokenization, POS tagging, and dependency parsing
        # These features help with punctuation placement
        nlp = stanza.Pipeline(
            lang=lang,
            processors='tokenize,pos,lemma',
            tokenize_no_ssplit=False,  # Allow sentence splitting
            verbose=False,
            download_method=None,  # Don't auto-download during initialization
        )
        
        logger.info("Stanza pipeline loaded for language: %s", lang)
        return nlp
        
    except Exception as exc:
        logger.error("Failed to load Stanza pipeline for %s: %s", lang, exc)
        raise RuntimeError(f"Could not initialize Stanza pipeline for {lang}") from exc


def restore_punctuation(
    text: str,
    *,
    lang: str = "nb",
    audio_is_question: bool = False,
) -> PunctuatedText:
    """
    Restore punctuation to text using Stanza NLP.
    
    Args:
        text: Raw transcription text without proper punctuation
        lang: Language code (nb, nn, sme)
        audio_is_question: Whether audio analysis suggests this is a question
        
    Returns:
        PunctuatedText with restored punctuation and metadata
    """
    text = text.strip()
    if not text:
        return PunctuatedText(text="", is_question=False, has_commas=False)
    
    try:
        nlp = _load_stanza_pipeline(lang)
        
        # Process the text
        doc = nlp(text)
        
        # Extract sentences with improved punctuation
        sentences = []
        is_question = audio_is_question
        has_commas = False
        
        for sent in doc.sentences:
            sent_text = _reconstruct_sentence(sent)
            
            # Check if this looks like a question based on linguistic features
            if _is_question_sentence(sent):
                is_question = True
            
            # Check for commas
            if ',' in sent_text:
                has_commas = True
            
            sentences.append(sent_text)
        
        # Join sentences
        if not sentences:
            # Fallback to basic formatting
            return _fallback_punctuation(text, audio_is_question)
        
        result_text = ' '.join(sentences)
        
        # Ensure proper capitalization
        if result_text and result_text[0].islower():
            result_text = result_text[0].upper() + result_text[1:]
        
        # Normalize spacing around punctuation
        result_text = _normalize_punctuation_spacing(result_text)
        
        return PunctuatedText(
            text=result_text,
            is_question=is_question,
            has_commas=has_commas,
            confidence=0.9,
        )
        
    except Exception as exc:
        logger.warning("Stanza punctuation restoration failed: %s, using fallback", exc)
        return _fallback_punctuation(text, audio_is_question)


def _reconstruct_sentence(sentence) -> str:
    """Reconstruct a sentence from Stanza tokens with proper punctuation."""
    tokens = []
    
    for word in sentence.words:
        token_text = word.text
        
        # POS-based punctuation hints
        # PUNCT tokens should be kept as-is
        if word.upos == 'PUNCT':
            tokens.append(token_text)
        else:
            tokens.append(token_text)
    
    # Join tokens
    text = ' '.join(tokens)
    
    # Add comma before coordinating conjunctions if appropriate
    text = _add_clause_commas(text, sentence)
    
    # Ensure sentence ends with punctuation
    if not text or text[-1] not in '.!?,;:':
        # Check if it's a question
        if _is_question_sentence(sentence):
            text = text + '?'
        else:
            text = text + '.'
    
    return text


def _is_question_sentence(sentence) -> bool:
    """Determine if a sentence is a question based on Stanza analysis."""
    if not sentence.words:
        return False
    
    # Check for question words at the start
    first_word = sentence.words[0]
    question_lemmas = {
        'hva', 'hvem', 'hvor', 'hvordan', 'hvorfor', 'hvilken', 'når',  # Norwegian
        'mii', 'gii', 'gos', 'gosa', 'mo', 'man', 'goas', 'makkár',  # Northern Sámi
        'what', 'who', 'where', 'when', 'why', 'how', 'which',  # English (for testing)
    }
    
    if first_word.lemma and first_word.lemma.lower() in question_lemmas:
        return True
    
    # Check for auxiliary verb at start (inverted question structure)
    if first_word.upos in ['AUX', 'VERB'] and len(sentence.words) > 1:
        second_word = sentence.words[1]
        if second_word.upos in ['PRON', 'NOUN', 'PROPN']:
            return True
    
    # Check if sentence already ends with '?'
    if sentence.words and sentence.words[-1].text == '?':
        return True
    
    return False


def _add_clause_commas(text: str, sentence) -> str:
    """Add commas before coordinating conjunctions where appropriate."""
    # Common coordinating conjunctions
    conjunctions = {'og', 'men', 'eller', 'for', 'så', 'and', 'but', 'or', 'ja', 'muhto'}
    
    # Simple heuristic: add comma before conjunction if sentence is long enough
    words = text.split()
    if len(words) < 5:
        return text
    
    result = []
    for i, word in enumerate(words):
        # Check if this word is a conjunction and not at the start
        if i > 0 and word.lower() in conjunctions:
            # Don't add comma if there's already one
            if result and not result[-1].endswith(','):
                # Add comma before the conjunction
                result[-1] = result[-1] + ','
        result.append(word)
    
    return ' '.join(result)


def _normalize_punctuation_spacing(text: str) -> str:
    """Normalize spacing around punctuation marks."""
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Ensure space after punctuation (except at end)
    text = re.sub(r'([,.!?;:])([^\s\d])', r'\1 \2', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove space at start/end
    return text.strip()


def _fallback_punctuation(text: str, audio_is_question: bool) -> PunctuatedText:
    """Fallback punctuation when Stanza is unavailable or fails."""
    text = text.strip()
    if not text:
        return PunctuatedText(text="", is_question=False, has_commas=False)
    
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)
    
    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Check if it looks like a question
    is_question = audio_is_question or _looks_like_question_heuristic(text)
    
    # Add ending punctuation if missing
    if text[-1] not in '.!?,;:':
        text = text + ('?' if is_question else '.')
    elif text[-1] == '?' and not audio_is_question:
        # Trust the existing question mark
        is_question = True
    
    return PunctuatedText(
        text=text,
        is_question=is_question,
        has_commas=',' in text,
        confidence=0.5,  # Lower confidence for fallback
    )


def _looks_like_question_heuristic(text: str) -> bool:
    """Simple heuristic to detect questions when Stanza is unavailable."""
    lowered = text.lower()
    
    # Check for question mark
    if '?' in lowered:
        return True
    
    # Check for question words at start
    question_words = {
        'hva', 'hvem', 'hvor', 'hvordan', 'hvorfor', 'hvilken', 'når',  # Norwegian
        'mii', 'gii', 'gos', 'gosa', 'mo', 'man', 'goas',  # Northern Sámi
        'what', 'who', 'where', 'when', 'why', 'how', 'which',  # English
        'kan', 'vil', 'skal', 'er', 'har',  # Norwegian auxiliary verbs
        'sáhtt', 'fert', 'lea', 'leat',  # Northern Sámi auxiliary verbs
    }
    
    tokens = lowered.split()
    if tokens and tokens[0] in question_words:
        return True
    
    # Check for inverted structure (verb-subject)
    if len(tokens) >= 2:
        # Very simple heuristic
        if tokens[0] in {'er', 'kan', 'vil', 'skal', 'har', 'lea', 'sáhtt'}:
            return True
    
    return False


def split_into_sentences(text: str, lang: str = "nb") -> List[str]:
    """
    Split text into sentences using Stanza.
    
    Args:
        text: Text to split
        lang: Language code
        
    Returns:
        List of sentence strings
    """
    try:
        nlp = _load_stanza_pipeline(lang)
        doc = nlp(text)
        return [' '.join(w.text for w in sent.words) for sent in doc.sentences]
    except Exception as exc:
        logger.warning("Stanza sentence splitting failed: %s", exc)
        # Simple fallback
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
