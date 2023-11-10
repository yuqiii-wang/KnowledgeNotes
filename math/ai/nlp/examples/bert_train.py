import transformers
from transformers import BertTokenizer, BertModel, AutoModelForQuestionAnswering
from transformers import pipeline

# from datasets import load_dataset, load_metric

print(transformers.__version__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/tokenizer.json')
model = BertModel.from_pretrained("bert-base-uncased")

question_answerer =  pipeline("question-answering", model="bert-base-uncased")

pred_1 = question_answerer(
    question="What is the novelty of the work.",
    context="In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.",
)

print(
    f"score: {round(pred_1['score'], 4)}, start: {pred_1['start']}, end: {pred_1['end']}, answer: {pred_1['answer']}"
)

