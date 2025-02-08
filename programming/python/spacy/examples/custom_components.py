import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc


@Language.component("info")
def info_component(doc: Doc) -> Doc:
    print("The part-of-speech tags are:", [{token.text: token.pos_} for token in doc])
    return doc

custom_abbrevs = dict()
custom_abbrevs["Ur"] = "your"
custom_abbrevs["ur"] = "your"
custom_abbrevs["pls"] = "please"
custom_abbrevs["ofr"] = "offer"
custom_abbrevs["thks"] = "thanks"
custom_abbrevs["thx"] = "thanks"
custom_abbrevs["thk"] = "thank"
custom_abbrevs["bk"] = "back"
custom_abbrevs["abt"] = "about"
custom_abbrevs["prod"] = "product"
custom_abbrevs["vd"] = "value date"

# Doc is immutable, can only apply mapping prior to sending doc into a spacy pipe
def map_abbrevs(doc_str: str) -> str:
    doc_ls = doc_str.split(" ")
    for idx, token in enumerate(doc_ls):
        doc_ls[idx] = custom_abbrevs.get(token, token)
    return " ".join(doc_ls)

nlp = spacy.load("en_core_web_md")

nlp.add_pipe("info", name="print_info", last=True)
print(nlp.pipe_names) 

TEST_SENTENCE = "Ur ofr abt this prod : US264938933, vd today, end 01/01/2030, thx"

doc = nlp(TEST_SENTENCE)

doc = nlp(map_abbrevs(TEST_SENTENCE))
