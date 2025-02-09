import spacy
from spacy.scorer import Scorer
from spacy.training import Example, Alignment
from spacy.tokens import Doc


# Provided scoring pipeline
nlp = spacy.load("en_core_web_md")

scorer = Scorer(nlp)

print(scorer)

pred1_sentence = "This is an amazing app"
tokens1_pred = pred1_sentence.split(" ")
pred1_doc = Doc(nlp.vocab, words=tokens1_pred)
ref1_sentence = "This is quite an amazing app"
token1_ref = ref1_sentence.split(" ")
tags1_ref = ["PRON", "VERB", "ADV", "DET", "ADJ", "NOUN"]
ref1_doc = Doc(nlp.vocab, words=token1_ref)
alignment1 = Alignment.from_strings(tokens1_pred, token1_ref)
example1 = Example(pred1_doc, ref1_doc, alignment1.x2y)

pred2_sentence = "This is amazing"
tokens2_pred = pred2_sentence.split(" ")
pred2_doc = Doc(nlp.vocab, words=tokens2_pred)
ref2_sentence = "This is just magnificent"
token2_ref = ref2_sentence.split(" ")
tags2_ref = ["PRON", "VERB", "ADV", "ADJ"]
ref2_doc = Doc(nlp.vocab, words=token2_ref)
alignment2 = Alignment.from_strings(tokens2_pred, token2_ref)
example2 = Example(pred2_doc, ref2_doc, alignment2.x2y)

examples = [example1, example2]

tok_scores = Scorer.score_tokenization(examples)
print(tok_scores)

scores = scorer.score(examples)
print(scores)