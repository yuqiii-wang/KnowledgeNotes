from transformers import BertTokenizer, BertModel, AutoModelForQuestionAnswering, EncoderDecoderModel

import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
    torch.cuda.empty_cache()
    
model_base_name = './bert-large-uncased'

max_new_tokens_len = 15

########### Loading 

tokenizer = BertTokenizer.from_pretrained(model_base_name)

model_qa = AutoModelForQuestionAnswering.from_pretrained(model_base_name)

########### Data Prepare

context_sentence = "Yuqi is a good guy. He loves AI."
question_sentence = "What does Yuqi like ?"

# automatically set token type for `question_sentence` and `context_sentence` if tokenizer takes two str inputs.
dataAll = tokenizer(
    question_sentence, context_sentence, add_special_tokens=True, return_tensors="pt"
)

########### Inference

with torch.no_grad():
    model_qa_outputs = model_qa( **dataAll )

answer_start_index = model_qa_outputs.start_logits.argmax()
answer_end_index = model_qa_outputs.end_logits.argmax()

predict_answer_tokens = dataAll.input_ids[0, answer_start_index : answer_end_index + 1]
predict_answer_decoded_tokens = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

print(predict_answer_decoded_tokens)