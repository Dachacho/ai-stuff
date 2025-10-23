from spellchecker import SpellChecker
import string

def build_custom_spellchecker(texts):
    spell = SpellChecker()
    vocab = set()
    for text in texts:
        for word in text.lower().split():
            clean_word = word.strip(string.punctuation)
            if clean_word: 
                vocab.add(clean_word)
    spell.word_frequency.load_words(vocab)
    return spell

def spellcheck(query, spell):
    corrected = []
    for word in query.split():
        clean_word = word.strip(string.punctuation)
        if clean_word:
            corrected_word = spell.correction(clean_word)
            corrected.append(corrected_word)
    return ' '.join(corrected)