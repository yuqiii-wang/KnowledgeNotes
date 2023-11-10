# Large Language Model (LLM) Practices

model name: `gpt-3.5-turbo-16k-0613`

temperature: randomness

system: context, such as your role is a physicist,, response might be academic; the anime expert, response might be kawaii.

safety/vetting: 

## Preparation

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets transformers dill tqdm
```

## Hugging Face

Hugging Face, Inc. is a French-American company and open-source community that develops tools and resources to build, deploy, and train machine learning models.

Its most famous product is `pip install transformers`.

Hugging Face provides unified APIs of different AI tasks and model configs.

Go to HuggingFace website and find desired model.
then download the files.

<div style="display: flex; justify-content: center;">
      <img src="imgs/huggingface_download.png" width="55%" height="50%" alt="huggingface_download" />
</div>
</br>

### Pipeline

Pipeline is a convenient API that takes one argument `pipeline(task="<your-task-name>")` from the list https://huggingface.co/docs/transformers/task_summary then the underlying code helps build the relevant modules.

For example, below gives two examples, one for Q&A and one for summary.

```py
from transformers import pipeline

question_answerer = pipeline(task="question-answering")
qa_pred = question_answerer(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
print(qa_pred.answer) # print "huggingface"

summarizer = pipeline(task="summarization")
sum_pred = summarizer(
    "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
)
print(sum_pred.summary)
```


## LangChain (LC) vs Semantic Kernel (SK)

