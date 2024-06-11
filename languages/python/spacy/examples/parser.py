import spacy
from spacy import displacy

# Load a pre-trained spaCy model (e.g., English model)
nlp = spacy.load("en_core_web_md")

# Define the example sentence
sentence = "I am a tourist, and I love New York and San Francisco."

# Process the sentence using the spaCy pipeline
doc = nlp(sentence)

# Print the dependency information
print(f"{'Token':<12} {'Head':<12} {'Dep':<10} {'POS':<10}")
print("-" * 40)
for token in doc:
    print(f"{token.text:<12} {token.head.text:<12} {token.dep_:<10} {token.pos_:<10}")

# Visualize the dependency tree
displacy.render(doc, style="dep")
displacy.serve(doc, style="dep", port=12345)
