from transformers import BertTokenizer, BertModel, BertLMHeadModel, EncoderDecoderModel
from transformers import BertGenerationEncoder, BertGenerationDecoder
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationOnlyLMHead
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    torch.cuda.empty_cache()
    
model_base_name = './bert-base-uncased'
model_new_seq2seq_name = './model_new_enc_dec_bert2bert'

max_new_tokens_len = 15

########### Loading 

tokenizer = BertTokenizer.from_pretrained(model_base_name)

model_enc = BertGenerationEncoder.from_pretrained(model_base_name, 
                                            output_hidden_states=True,
                                            output_attentions=True,
                                            bos_token_id=tokenizer.get_vocab()["[CLS]"], 
                                            eos_token_id=tokenizer.get_vocab()["[SEP]"])
model_dec = BertGenerationDecoder.from_pretrained(model_base_name, 
                                            add_cross_attention=True, 
                                            is_decoder=True, 
                                            output_hidden_states=True,
                                            output_attentions=True,
                                            bos_token_id=tokenizer.get_vocab()["[CLS]"], 
                                            eos_token_id=tokenizer.get_vocab()["[SEP]"])

model_enc_dec = EncoderDecoderModel(encoder=model_enc, decoder=model_dec)

########### Prepare Data

dataInputSentence = "Yuqi's friends said Yuqi loves life and AI. Besides, Yuqi participated in many AI knowledge sharing seminars."
dataOutputSentence = "Yuqi is passionate about AI."

dataInput = tokenizer(
    dataInputSentence, add_special_tokens=True, return_tensors="pt"
)
dataOutput = tokenizer(
    dataOutputSentence, return_tensors="pt"
)

########### Manual encoder decoder

model_enc_outputs = model_enc(                
                input_ids=dataInput.input_ids,
                attention_mask=dataInput.attention_mask,
                )

model_dec_outputs = model_dec(    
            input_ids=dataOutput.input_ids,
            attention_mask=dataOutput.attention_mask,
            encoder_hidden_states=model_enc_outputs.last_hidden_state,
            encoder_attention_mask=dataInput.attention_mask,
            labels=None,
            )

model_manual_enc_dec_outputs = Seq2SeqLMOutput(
            loss=model_dec_outputs.loss,
            logits=model_dec_outputs.logits,
            past_key_values=None,
            decoder_hidden_states=model_dec_outputs.hidden_states,
            decoder_attentions=model_dec_outputs.attentions,
            encoder_last_hidden_state=model_enc_outputs.last_hidden_state,
            encoder_hidden_states=model_enc_outputs.hidden_states,
            encoder_attentions=model_enc_outputs.attentions,
            )

model_manual_enc_dec_forward_output_vec = []
for each_pred in model_manual_enc_dec_outputs.logits[0]:
    token_id = torch.argmax(torch.softmax(each_pred, dim=0))
    model_manual_enc_dec_forward_output_vec.append(token_id)
print("model_manual_enc_dec_forward_output_vec: ")
print(tokenizer.decode(model_manual_enc_dec_forward_output_vec))

############ Built-in Encoder Decoder (to verify the above results that they are identical)

model_enc_dec_forward_output = model_enc_dec(input_ids=dataInput.input_ids, 
                                              decoder_input_ids=dataOutput.input_ids)
model_enc_dec_forward_output_vec = []
for each_pred in model_enc_dec_forward_output.logits[0]:
    token_id = torch.argmax(torch.softmax(each_pred, dim=0))
    model_enc_dec_forward_output_vec.append(token_id)
print("model_enc_dec_forward_output_vec: ")
print(tokenizer.decode(model_enc_dec_forward_output_vec))

############ Fine-tuning
###
#    The above bert model is pre-trained, should be re-trained on task-specific data 
###

dataInputSentenceList = []
dataInputSentenceList.append(dataInputSentence)
dataInputSentenceList.append("Yuqi loves playing football with his friends, once a week.")
dataInputSentenceList.append("Yuqi climbed a mountain last week. The mountain's height is 1000 meter.")
dataInputSentenceList.append("Yuqi has been programming for a long time, studied c++, python and other computer technologies.")
dataInputSentenceList.append("Yuqi joined a computer knowledge sharing seminar, and spoke as a representative of his company.")
dataInputSentenceList.append("People in Yuqi's company said Yuqi was promoted to a senior position, taking role as a specialist last month.")
dataInputSentenceList.append("There has been rumors about Yuqi's resignation for he wants to be a full-time dad.")
dataInputSentenceList.append("Yuqi bought lots of gifts to his girlfriend, spent lots of money.")
dataInputSentenceList.append("Yuqi enjoys his days everyday.")
dataInputSentenceList.append("Yuqi was asked to participate in a computer skill challenge and he won a prize by hacking the host's computer system.")
dataInputSentenceList.append("Friends of Yuqi are attractive to many good looking boys and girls.")
dataOutputSentenceList = []
dataOutputSentenceList.append(dataOutputSentence)
dataOutputSentenceList.append("Yuqi loves sports.")
dataOutputSentenceList.append("Yuqi loves climbing.")
dataOutputSentenceList.append("Yuqi is a programmer .")
dataOutputSentenceList.append("Yuqi is a company representative .")
dataOutputSentenceList.append("Yuqi is a specialist.")
dataOutputSentenceList.append("Yuqi will be a full-time dad.")
dataOutputSentenceList.append("Yuqi loves his girlfriend.")
dataOutputSentenceList.append("Yuqi is a happy man.")
dataOutputSentenceList.append("Yuqi is a hacker.")
dataOutputSentenceList.append("Yuqi's friends received many attentions from boys and girls.")

model_new_enc_dec = model_enc_dec.to(device)
dataInputSentenceTensor = tokenizer(dataInputSentenceList, padding=True, return_tensors="pt").to(device)
dataOutputSentenceTensor = tokenizer(dataOutputSentenceList, padding=True, return_tensors="pt").to(device)

optim = torch.optim.SGD(model_new_enc_dec.parameters(), lr=3e-5, momentum=0.5)

last_loss = 0
for i in range(200):
    loss = 0
    loss += model_new_enc_dec(input_ids=dataInputSentenceTensor.input_ids, 
                             decoder_input_ids=dataOutputSentenceTensor.input_ids, 
                             labels=dataOutputSentenceTensor.input_ids,
                             encoder_attention_mask=dataInputSentenceTensor.attention_mask,
                             decoder_attention_mask=dataOutputSentenceTensor.attention_mask,
                             ).loss
    last_loss = loss.item()
    loss.backward()
    optim.step()
    if i % 10 == 0:
        print("epoch " + str(i) + " training loss: " + str(loss.item()))

    if (last_loss * 1.2 < loss.item()):
        break

###
#   Now test it
###

dataInputTestSentence = "Yuqi joined a jogging group that hosts a jogging activity every month."
dataOutputTestSentence = "Yuqi is a jogger."

dataInputTest = tokenizer(
    dataInputTestSentence, padding=True,  return_tensors="pt"
).to(device)
dataOutputTest = tokenizer(
    dataOutputTestSentence, padding=True, return_tensors="pt"
).to(device)

model_new_enc_dec_forward_output = model_new_enc_dec(input_ids=dataInputTest.input_ids, 
                                                     decoder_input_ids=dataOutputTest.input_ids, 
                                                     labels=dataOutputTest.input_ids,
                                                     )
model_new_enc_dec_forward_output_vec = []
for each_pred in model_new_enc_dec_forward_output.logits[0]:
    token_id = torch.argmax(torch.softmax(each_pred, dim=0))
    model_new_enc_dec_forward_output_vec.append(token_id)
print("model_new_enc_dec_forward_output_vec: ")
print(tokenizer.decode(model_new_enc_dec_forward_output_vec))

model_new_enc_dec_gen_outputs = model_new_enc_dec.generate(dataInputTest.input_ids)
print("model_new_enc_dec_gen_outputs: ")
print(tokenizer.decode(model_new_enc_dec_gen_outputs[0]))

###
#   Save this model
###

model_new_enc_dec.save_pretrained(model_new_seq2seq_name)