"""spaCy provider -- en_core_web_sm."""


def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")
