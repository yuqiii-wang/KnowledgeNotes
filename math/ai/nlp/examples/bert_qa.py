from transformers import BertTokenizer, BertModel, AutoModelForQuestionAnswering, EncoderDecoderModel

import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
    torch.cuda.empty_cache()
    
model_base_name = './bert-base-uncased'

max_new_tokens_len = 15

########### Loading 

tokenizer = BertTokenizer.from_pretrained(model_base_name)

model_qa = AutoModelForQuestionAnswering.from_pretrained(model_base_name)

########### Data Preparation for Test

context_sentence = "Yuqi's passion for technology can be traced back to his childhood, that he loves playing with toys that inspires curiosity and wisdom."
question_sentence = "What does Yuqi like ?"

dataTest = tokenizer(
    context_sentence, question_sentence, add_special_tokens=True, return_tensors="pt"
)

answer_id = tokenizer.convert_tokens_to_ids('technology')
answer_start_pos = (dataTest.input_ids[0] == answer_id).nonzero()
ansswer_start_pos, ansswer_end_pos = answer_start_pos[0], answer_start_pos[0] + 1


########### Inference Test

model_qa_outputs = model_qa(input_ids=dataTest.input_ids,
                            attention_mask=dataTest.attention_mask,
                            token_type_ids=dataTest.token_type_ids,
                            start_positions=ansswer_start_pos,
                            end_positions=ansswer_end_pos
                            )

answer_start_index = model_qa_outputs.start_logits.argmax()
answer_end_index = model_qa_outputs.end_logits.argmax()

predict_answer_tokens = dataTest.input_ids[0, answer_start_index : answer_end_index + 1]
predict_answer_decoded_tokens = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

print("model_qa_outputs.loss: " + str(model_qa_outputs.loss.item()))
print("predict_answer_decoded_tokens: " + predict_answer_decoded_tokens)

########### Fine-tuning

context_sentence_list = []
question_sentence_list = []

context_sentence_list.append("Yuqi studied electronics in China, then he moved to Australia. After 3 years, he moved back to China, and stayed there ever since.")
context_sentence_list.append("Yuqi's first encounter with AI happened in 2015, when his teacher introduced him this new technology. 2 years later in 2017, he onboarded a new journey to advance his knowledge in AI.")
context_sentence_list.append("After graduation from university with a master degree in AI, Yuqi started his career as a SLAM engineer specializing in computer vision recognition. After one year, he accepted another job offer as a c++ engineer.")
context_sentence_list.append("Later in 2020, Yuqi got tired of a programming life, then moved to the financial industry, which is a whole new area compared to the previous AI industry.")
context_sentence_list.append("One thing particularly interesting about Yuqi is that, Yuqi is a good looking guy.")
question_sentence_list.append("Where does Yuqi live now ?")
question_sentence_list.append("When did Yuqi start studying AI ?")
question_sentence_list.append("What is Yuqi's occupation after graduation?")
question_sentence_list.append("What industry did Yuqi stay before 2020 ?")
question_sentence_list.append("What is a fascinating thing about Yuqi ?")

assert(len(context_sentence_list) == len(question_sentence_list))

model_new_qa = model_qa.to(device)
dataContextQuestionSentenceTensor = tokenizer(context_sentence_list, question_sentence_list, padding=True, return_tensors="pt").to(device)

answerStartPosTensor = torch.zeros(len(context_sentence_list), dtype=torch.long).to(device)
answerStartPosTensor[0] = (dataContextQuestionSentenceTensor.input_ids[0] == tokenizer.convert_tokens_to_ids('China'.lower())).nonzero()[0]
answerStartPosTensor[1] = (dataContextQuestionSentenceTensor.input_ids[1] == tokenizer.convert_tokens_to_ids('2015'.lower())).nonzero()[0]
answerStartPosTensor[2] = (dataContextQuestionSentenceTensor.input_ids[2] == tokenizer.convert_tokens_to_ids('SLAM'.lower())).nonzero()[0]
answerStartPosTensor[3] = (dataContextQuestionSentenceTensor.input_ids[3] == tokenizer.convert_tokens_to_ids('AI'.lower())).nonzero()[0]
answerStartPosTensor[4] = (dataContextQuestionSentenceTensor.input_ids[4] == tokenizer.convert_tokens_to_ids('good'.lower())).nonzero()[0]

answerEndPosTensor = torch.zeros(len(context_sentence_list), dtype=torch.long).to(device)
answerEndPosTensor[0] = answerStartPosTensor[0] + 1
answerEndPosTensor[1] = answerStartPosTensor[1] + 1
answerEndPosTensor[2] = answerStartPosTensor[2] + 2
answerEndPosTensor[3] = answerStartPosTensor[3] + 1
answerEndPosTensor[4] = answerStartPosTensor[4] + 2

optim = torch.optim.SGD(model_new_qa.parameters(), lr=1e-4, momentum=0.5)

for i in range(50):
    loss = model_new_qa(input_ids=dataContextQuestionSentenceTensor.input_ids,
                       attention_mask=dataContextQuestionSentenceTensor.attention_mask,
                       token_type_ids=dataContextQuestionSentenceTensor.token_type_ids,
                       start_positions=answerStartPosTensor,
                       end_positions=answerEndPosTensor
                       ).loss
    last_loss = loss.item()
    loss.backward()
    optim.step()
    if i % 10 == 0:
        print("epoch " + str(i) + " training loss: " + str(loss.item()))

    if (last_loss * 1.1 < loss.item()):
        break

########### Inference after fine-tuning

dataTest = dataTest.to(device)
ansswer_start_pos, ansswer_end_pos = ansswer_start_pos.to(device), ansswer_end_pos.to(device)

model_new_qa_outputs = model_new_qa(input_ids=dataTest.input_ids,
                            attention_mask=dataTest.attention_mask,
                            token_type_ids=dataTest.token_type_ids,
                            start_positions=ansswer_start_pos,
                            end_positions=ansswer_end_pos
                            )

answer_new_start_index = model_new_qa_outputs.start_logits.argmax()
answer_new_end_index = model_new_qa_outputs.end_logits.argmax()

predict_new_answer_tokens = dataTest.input_ids[0, answer_new_start_index : answer_new_end_index + 1]
predict_new_answer_decoded_tokens = tokenizer.decode(predict_new_answer_tokens, skip_special_tokens=True)

print("predict_new_answer_decoded_tokens: " + predict_new_answer_decoded_tokens)
print("model_new_qa_outputs.loss: " + str(model_new_qa_outputs.loss.item()))