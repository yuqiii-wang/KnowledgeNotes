import spacy
from spacy.pipeline.lemmatizer import Lemmatizer
from spacy.tokens import Doc, Token
from spacy.language import Language
from thinc.api import Model

from typing import List, Callable, Optional, overload


class CustomEnglish(Language):
    lang = "en"

@CustomEnglish.factory(
    "custom_lemmatizer",
    assigns=["token.lemma"],
    default_config={
        "model": None,
        "mode": "rule",
        "overwrite": False,
        "scorer": {"@scorers": "spacy.lemmatizer_scorer.v1"},
    },
    default_score_weights={"lemma_acc": 1.0},
)
def make_lemmatizer(
    nlp: Language,
    model: Optional[Model],
    name: str,
    mode: str,
    overwrite: bool,
    scorer: Optional[Callable],
):
    return CustomLemmatizer(
        nlp.vocab, model, name, mode=mode, overwrite=overwrite, scorer=scorer
    )

class CustomLemmatizer(Lemmatizer):
    def __init__(self, vocab):
        super().__init__(vocab, model=None, name="CustomLemmatizer", mode="lookup", overwrite=True)

        # Define a custom mapping of words to their lemmas
        self.custom_lemmas = {
            "ur": "your",
            "pls": "please",
            "ofr": "offer",
            "gv": "give",
            "thks": "thanks",
            "thx": "thanks",
            "thk": "thank",
            "bk": "back",
            "bak": "back",
            "abt": "about",
            "prod": "product",
        }

    @overload
    def __call__(self, doc: Doc) -> Doc:
        for token in doc:
            if self.overwrite or token.lemma == 0:
                lemma_ = self.lemmatize(token)[0]
                self.lemma_ = self.custom_lemmas.get(lemma_, lemma_)


# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Replace the default lemmatizer with the custom lemmatizer
nlp.replace_pipe("lemmatizer", "custom_lemmatizer")

# Test the custom lemmatizer
doc = nlp("Ur ofr abt this prod: US264938933, thx")
for token in doc:
    print(f"{token.text} -> {token.lemma_}")
