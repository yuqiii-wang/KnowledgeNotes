from transformers import BertTokenizer, BertModel
from transformers import pipeline
import torch
import torch.nn.functional.cosine_similarity as cosine_similarity

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
    torch.cuda.empty_cache()

### FIrst download bert-large-uncased
# huggingface-cli download --resume-download bert-large-uncased --local-dir bert-large-uncased
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

inputSentence = "Do you [MASK] me?"

# Manually 
inputs = tokenizer(inputSentence, return_tensors="pt").to(device)
model = model.to(device)
outputs = model(**inputs)
model
dist = F.cosine_similarity(x,outputs)
index_sorted = torch.argsort(dist)
top_1000 = index_sorted[:1000]

# build a pipeline work.
unmasker = pipeline('fill-mask', model='bert-large-uncased', tokenizer=tokenizer, device=device)

# make a prediction
pred = unmasker(inputSentence)
print(pred)