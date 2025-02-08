import spacy

# Load a blank spaCy model
nlp = spacy.blank("en")

# Add the attribute ruler to the pipeline
config = {"validate": True}
ruler = nlp.add_pipe("attribute_ruler", config=config)

# Example patterns and attributes
patterns = [
    [{"LOWER": "i"}, {"LOWER": "am"}],
    [{"LOWER": "you're"}],
    [{"TEXT": "New"}, {"TEXT": "York"}],
    [{"TEXT": "San"}, {"TEXT": "Francisco"}]
]

attrs = [
    {"LEMMA": "be", "POS": "AUX"},
    {"LEMMA": "be", "POS": "AUX"},
    {"LEMMA": "PLACE", "POS": "PROPN"},
    {"LEMMA": "PLACE", "POS": "PROPN"}
]

# Add patterns and attributes to the ruler
for pattern, attr in zip(patterns, attrs):
    ruler.add_patterns([{"patterns": [pattern], "attrs": attr}])


# Process a text
doc = nlp("I am a tourist, and I love New York and San Francisco.")
for token in doc:
    text_to_print = f"{token.text}: lemma {token.lemma_}, pos {token.pos_}" \
        if token.lemma_ is not "" else f"{token.text}"
    print(text_to_print)