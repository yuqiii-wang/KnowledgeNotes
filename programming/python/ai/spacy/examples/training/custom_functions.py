import spacy
from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
    
@spacy.registry.misc("custom_tokenizer")
def create_custom_tokenizer():
    def create_tokenizer(nlp):
        return WhitespaceTokenizer(nlp.vocab)
    return create_tokenizer
